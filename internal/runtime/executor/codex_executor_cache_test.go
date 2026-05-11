package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

type codexCacheTestHTTPResponse struct {
	status   int
	respBody string
	respCT   string
	err      error
}

type codexCacheTestRoundTripper struct {
	body       []byte
	headers    http.Header
	bodies     [][]byte
	headersLog []http.Header
	responses  []codexCacheTestHTTPResponse
}

func (rt *codexCacheTestRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	rt.body = body
	rt.headers = req.Header.Clone()
	rt.bodies = append(rt.bodies, body)
	rt.headersLog = append(rt.headersLog, req.Header.Clone())

	status := http.StatusOK
	contentType := "application/json"
	respBody := `{"ok":true}`
	var respErr error
	if idx := len(rt.bodies) - 1; idx >= 0 && idx < len(rt.responses) {
		response := rt.responses[idx]
		if response.status != 0 {
			status = response.status
		}
		if response.respCT != "" {
			contentType = response.respCT
		}
		respBody = response.respBody
		respErr = response.err
	}
	if respErr != nil {
		return nil, respErr
	}
	return &http.Response{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{contentType},
		},
		Body:    io.NopCloser(strings.NewReader(respBody)),
		Request: req,
	}, nil
}

func newCodexCacheTestAuth(id string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{
		ID:       id,
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":  "test-key-" + id,
			"base_url": "https://example.invalid/backend-api/codex",
		},
	}
}

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_ManagedPromptCacheKeyFromAPIKey(t *testing.T) {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("userApiKey", "test-api-key")

	ctx := context.WithValue(context.Background(), "gin", ginCtx)
	executor := &CodexExecutor{}
	rawJSON := []byte(`{"model":"gpt-5.3-codex","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.3-codex",
		Payload: []byte(`{"model":"gpt-5.3-codex"}`),
	}
	url := "https://example.com/responses"

	sourceFormat := sdktranslator.FromString("openai")
	httpReq, continuity, attemptBody, err := executor.cacheHelper(ctx, nil, sourceFormat, url, req, cliproxyexecutor.Options{
		SourceFormat: sourceFormat,
	}, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error: %v", err)
	}

	body, errRead := io.ReadAll(httpReq.Body)
	if errRead != nil {
		t.Fatalf("read request body: %v", errRead)
	}

	gotKey := gjson.GetBytes(body, "prompt_cache_key").String()
	if gotKey == "" {
		t.Fatal("prompt_cache_key should not be empty")
	}
	if _, errParse := uuid.Parse(gotKey); errParse != nil {
		t.Fatalf("prompt_cache_key should be a UUID, got %q: %v", gotKey, errParse)
	}
	if string(attemptBody) != string(body) {
		t.Fatalf("attempt body should match actual request body: attempt=%s body=%s", string(attemptBody), string(body))
	}
	expectedCacheKey := "api:" + uuid.NewSHA1(uuid.NameSpaceOID, []byte("cli-proxy-api:codex:session-cache:test-api-key")).String()
	if continuity.CacheKey != expectedCacheKey {
		t.Fatalf("cache key = %q, want %q", continuity.CacheKey, expectedCacheKey)
	}
	if gotConversation := httpReq.Header.Get("Conversation_id"); gotConversation != "" {
		t.Fatalf("Conversation_id = %q, want empty", gotConversation)
	}
	if gotSession := httpReq.Header.Get("Session_id"); gotSession != gotKey {
		t.Fatalf("Session_id = %q, want %q", gotSession, gotKey)
	}

	httpReq2, _, _, err := executor.cacheHelper(ctx, nil, sourceFormat, url, req, cliproxyexecutor.Options{
		SourceFormat: sourceFormat,
	}, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error (second call): %v", err)
	}
	body2, errRead2 := io.ReadAll(httpReq2.Body)
	if errRead2 != nil {
		t.Fatalf("read request body (second call): %v", errRead2)
	}
	gotKey2 := gjson.GetBytes(body2, "prompt_cache_key").String()
	if gotKey2 != gotKey {
		t.Fatalf("prompt_cache_key (second call) = %q, want cached %q", gotKey2, gotKey)
	}
}

func TestCodexHttpRequestBackfillsPromptCacheKeyFromSessionHeader(t *testing.T) {
	rt := &codexCacheTestRoundTripper{responses: []codexCacheTestHTTPResponse{{status: http.StatusOK, respBody: `{"ok":true}`}}}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	executor := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	httpReq, err := http.NewRequest(http.MethodPost, "https://example.invalid/backend-api/codex/responses", strings.NewReader(`{"model":"gpt-5","input":"hi"}`))
	if err != nil {
		t.Fatalf("new request failed: %v", err)
	}
	httpReq.Header.Set("Session_id", "session-from-header")

	resp, errReq := executor.HttpRequest(ctx, auth, httpReq)
	if errReq != nil {
		t.Fatalf("HttpRequest error: %v", errReq)
	}
	if resp != nil && resp.Body != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	if got := gjson.GetBytes(rt.body, "prompt_cache_key").String(); got != "session-from-header" {
		t.Fatalf("prompt_cache_key = %q, want %q", got, "session-from-header")
	}
	if got := rt.headers.Get("Session_id"); got != "session-from-header" {
		t.Fatalf("Session_id = %q, want %q", got, "session-from-header")
	}
}

func TestCodexExplicitPromptCacheKeyDoesNotOverwriteAuthScopedSessionCache(t *testing.T) {
	executor := NewCodexExecutor(&config.Config{})
	auth := newCodexCacheTestAuth("explicit-does-not-poison-main")
	url := "https://example.invalid/backend-api/codex/responses"
	sourceFormat := sdktranslator.FromString("openai-response")
	opts := cliproxyexecutor.Options{SourceFormat: sourceFormat}

	initialReq := cliproxyexecutor.Request{Model: "gpt-5", Payload: []byte(`{"model":"gpt-5","input":"first"}`)}
	httpReq1, continuity1, _, err := executor.cacheHelper(context.Background(), auth, sourceFormat, url, initialReq, opts, initialReq.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (initial): %v", err)
	}
	body1, err := io.ReadAll(httpReq1.Body)
	if err != nil {
		t.Fatalf("read request body (initial): %v", err)
	}
	fixedSessionID := gjson.GetBytes(body1, "prompt_cache_key").String()
	if fixedSessionID == "" {
		t.Fatal("fixed session should not be empty")
	}

	explicitReq := cliproxyexecutor.Request{Model: "gpt-5", Payload: []byte(`{"model":"gpt-5","input":"second","prompt_cache_key":"explicit-cache-key"}`)}
	httpReq2, _, _, err := executor.cacheHelper(context.Background(), auth, sourceFormat, url, explicitReq, opts, explicitReq.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (explicit): %v", err)
	}
	body2, err := io.ReadAll(httpReq2.Body)
	if err != nil {
		t.Fatalf("read request body (explicit): %v", err)
	}
	if got := gjson.GetBytes(body2, "prompt_cache_key").String(); got != "explicit-cache-key" {
		t.Fatalf("explicit prompt_cache_key = %q, want %q", got, "explicit-cache-key")
	}

	httpReq3, continuity3, _, err := executor.cacheHelper(context.Background(), auth, sourceFormat, url, initialReq, opts, initialReq.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (follow-up): %v", err)
	}
	body3, err := io.ReadAll(httpReq3.Body)
	if err != nil {
		t.Fatalf("read request body (follow-up): %v", err)
	}
	if got := gjson.GetBytes(body3, "prompt_cache_key").String(); got != fixedSessionID {
		t.Fatalf("follow-up prompt_cache_key = %q, want original fixed session %q", got, fixedSessionID)
	}
	if continuity3.CacheKey != continuity1.CacheKey {
		t.Fatalf("cacheKey mismatch after explicit override: %q != %q", continuity3.CacheKey, continuity1.CacheKey)
	}
}

func TestCodexExecuteRetriesOnceWithFreshSessionAfterManagedSessionError(t *testing.T) {
	rt := &codexCacheTestRoundTripper{responses: []codexCacheTestHTTPResponse{
		{status: http.StatusInternalServerError, respBody: `{"error":"first failure"}`},
		{status: http.StatusOK, respCT: "text/event-stream", respBody: "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"},
	}}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	executor := NewCodexExecutor(&config.Config{})
	auth := newCodexCacheTestAuth("retry-after-error-main")
	resp, err := executor.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":[{"role":"user","content":"retry me"}]}`),
	}, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response"), Stream: false})
	if err != nil {
		t.Fatalf("Execute should retry once and succeed, got error: %v", err)
	}
	if len(rt.headersLog) != 2 {
		t.Fatalf("round trip count = %d, want 2", len(rt.headersLog))
	}

	firstSessionID := rt.headersLog[0].Get("Session_id")
	secondSessionID := rt.headersLog[1].Get("Session_id")
	if firstSessionID == "" || secondSessionID == "" {
		t.Fatalf("session ids should be non-empty: first=%q second=%q", firstSessionID, secondSessionID)
	}
	if firstSessionID == secondSessionID {
		t.Fatalf("retry should use a fresh session id, still %q", firstSessionID)
	}
	if got := gjson.GetBytes(rt.bodies[0], "prompt_cache_key").String(); got != firstSessionID {
		t.Fatalf("first prompt_cache_key = %q, want %q", got, firstSessionID)
	}
	if got := gjson.GetBytes(rt.bodies[1], "prompt_cache_key").String(); got != secondSessionID {
		t.Fatalf("second prompt_cache_key = %q, want %q", got, secondSessionID)
	}
	if !gjson.ValidBytes(resp.Payload) {
		t.Fatalf("response payload should be valid JSON, got %q", string(resp.Payload))
	}
}
