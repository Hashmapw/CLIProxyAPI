package executor

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

type codexCaptureHTTPResponse struct {
	status   int
	respBody string
	respCT   string
	err      error
}

type codexCaptureRoundTripper struct {
	body       []byte
	headers    http.Header
	status     int
	respBody   string
	respCT     string
	bodies     [][]byte
	headersLog []http.Header
	responses  []codexCaptureHTTPResponse
}

func (rt *codexCaptureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	rt.body = body
	rt.headers = req.Header.Clone()
	rt.bodies = append(rt.bodies, body)
	rt.headersLog = append(rt.headersLog, req.Header.Clone())

	status := rt.status
	contentType := rt.respCT
	respBody := rt.respBody
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
	if status == 0 {
		status = http.StatusBadRequest
	}
	if strings.TrimSpace(contentType) == "" {
		contentType = "application/json"
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

func newCodexSessionTestAuth(id string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{
		ID:       id,
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":  "test-key-" + id,
			"base_url": "https://example.invalid/backend-api/codex",
		},
	}
}

func TestCodexHttpRequestBackfillsPromptCacheKeyFromSessionHeader(t *testing.T) {
	rt := &codexCaptureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, err := url.Parse("https://example.invalid/backend-api/codex/responses")
	if err != nil {
		t.Fatalf("parse url failed: %v", err)
	}
	httpReq := &http.Request{
		Method: http.MethodPost,
		URL:    reqURL,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
			"Session_id":   []string{"session-from-header"},
		},
		Body:          io.NopCloser(strings.NewReader(`{"model":"gpt-5","input":"hi"}`)),
		ContentLength: int64(len(`{"model":"gpt-5","input":"hi"}`)),
	}

	resp, errReq := exec.HttpRequest(ctx, auth, httpReq)
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
	if got := rt.headers.Get("Conversation_id"); got != "" {
		t.Fatalf("Conversation_id = %q, want empty", got)
	}
}

func TestCodexSessionCacheIsAuthScopedAcrossModels(t *testing.T) {
	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexSessionTestAuth("auth-scope-cross-model-main")
	url := "https://example.invalid/backend-api/codex/responses"

	req1 := cliproxyexecutor.Request{Model: "gpt-5", Payload: []byte(`{"model":"gpt-5","input":"hi-1"}`)}
	httpReq1, continuity1, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, req1, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")}, req1.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (first model): %v", err)
	}
	body1, err := io.ReadAll(httpReq1.Body)
	if err != nil {
		t.Fatalf("read request body (first model): %v", err)
	}

	req2 := cliproxyexecutor.Request{Model: "gpt-5-mini", Payload: []byte(`{"model":"gpt-5-mini","input":"hi-2"}`)}
	httpReq2, continuity2, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, req2, cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")}, req2.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (second model): %v", err)
	}
	body2, err := io.ReadAll(httpReq2.Body)
	if err != nil {
		t.Fatalf("read request body (second model): %v", err)
	}

	firstKey := gjson.GetBytes(body1, "prompt_cache_key").String()
	secondKey := gjson.GetBytes(body2, "prompt_cache_key").String()
	if firstKey == "" || secondKey == "" {
		t.Fatalf("prompt_cache_key values should be non-empty: first=%q second=%q", firstKey, secondKey)
	}
	if firstKey != secondKey {
		t.Fatalf("prompt_cache_key should reuse same auth-scoped session across models: %q != %q", firstKey, secondKey)
	}
	if continuity1.CacheKey != continuity2.CacheKey {
		t.Fatalf("cacheKey mismatch across models: %q != %q", continuity1.CacheKey, continuity2.CacheKey)
	}
}

func TestCodexExplicitPromptCacheKeyDoesNotOverwriteAuthScopedSessionCache(t *testing.T) {
	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexSessionTestAuth("explicit-does-not-poison-main")
	url := "https://example.invalid/backend-api/codex/responses"
	opts := cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")}

	initialReq := cliproxyexecutor.Request{Model: "gpt-5", Payload: []byte(`{"model":"gpt-5","input":"first"}`)}
	httpReq1, continuity1, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, initialReq, opts, initialReq.Payload)
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
	httpReq2, _, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, explicitReq, opts, explicitReq.Payload)
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

	httpReq3, continuity3, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, initialReq, opts, initialReq.Payload)
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
	rt := &codexCaptureRoundTripper{responses: []codexCaptureHTTPResponse{
		{status: http.StatusInternalServerError, respBody: `{"error":"first failure"}`},
		{status: http.StatusOK, respCT: "text/event-stream", respBody: "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"},
	}}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexSessionTestAuth("retry-after-error-main")
	resp, err := exec.Execute(ctx, auth, cliproxyexecutor.Request{
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

func TestApplyCodexPromptCacheHeadersSetsConversationSessionAndContexts(t *testing.T) {
	body, headers, continuity := applyCodexPromptCacheHeaders(
		context.Background(),
		nil,
		sdktranslator.FromString("openai-response"),
		cliproxyexecutor.Request{Model: "gpt-5", Payload: []byte(`{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}]}`)},
		cliproxyexecutor.Options{SourceFormat: sdktranslator.FromString("openai-response")},
		[]byte(`{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}]}`),
	)

	if continuity.CacheKey == "" {
		t.Fatal("cacheKey should not be empty")
	}
	sessionID := headers.Get("Session_id")
	if sessionID == "" {
		t.Fatal("Session_id should not be empty")
	}
	if headers.Get("Conversation_id") != sessionID {
		t.Fatalf("Conversation_id = %q, want %q", headers.Get("Conversation_id"), sessionID)
	}
	if got := gjson.GetBytes(body, "prompt_cache_key").String(); got != sessionID {
		t.Fatalf("prompt_cache_key = %q, want %q", got, sessionID)
	}
	if got := gjson.GetBytes(body, "contexts.0.prompt_cache_key").String(); got != sessionID {
		t.Fatalf("contexts.0.prompt_cache_key = %q, want %q", got, sessionID)
	}
	if got := gjson.GetBytes(body, "contexts.1.prompt_cache_key").String(); got != sessionID {
		t.Fatalf("contexts.1.prompt_cache_key = %q, want %q", got, sessionID)
	}
}
