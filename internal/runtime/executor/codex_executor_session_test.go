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

type captureHTTPResponse struct {
	status   int
	respBody string
	respCT   string
	err      error
}

type captureRoundTripper struct {
	body       []byte
	headers    http.Header
	status     int
	respBody   string
	respCT     string
	requestURL string
	bodies     [][]byte
	headersLog []http.Header
	urls       []string
	responses  []captureHTTPResponse
}

func (rt *captureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	rt.body = body
	rt.headers = req.Header.Clone()
	rt.requestURL = req.URL.String()
	rt.bodies = append(rt.bodies, body)
	rt.headersLog = append(rt.headersLog, req.Header.Clone())
	rt.urls = append(rt.urls, req.URL.String())

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

func newCodexTestAuth(id string) *cliproxyauth.Auth {
	return &cliproxyauth.Auth{
		ID:       id,
		Provider: "codex",
		Attributes: map[string]string{
			"api_key":  "test-key-" + id,
			"base_url": "https://example.invalid/backend-api/codex",
		},
	}
}

func TestCodexExecuteInjectsSessionTripleAndNormalizesTools(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusBadRequest, respBody: `{"error":"mock"}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"api_key":  "test-key",
		"base_url": "https://example.invalid/backend-api/codex",
	}}
	_, err := exec.Execute(ctx, auth, cliproxyexecutor.Request{
		Model: "gpt-5",
		Payload: []byte(`{
			"model":"gpt-5",
			"input":[{"role":"user","content":"hi"}],
			"tool_choice":"none",
			"tools":{"type":"function","name":"echo","description":"x","parameters":{"type":"object"}}
		}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
	})
	if err == nil {
		t.Fatalf("expected upstream error")
	}

	promptCacheKey := gjson.GetBytes(rt.body, "prompt_cache_key").String()
	if promptCacheKey == "" {
		t.Fatal("prompt_cache_key should not be empty")
	}
	if got := rt.headers.Get("Session_id"); got != promptCacheKey {
		t.Fatalf("Session_id = %q, want %q", got, promptCacheKey)
	}
	if got := rt.headers.Get("Conversation_id"); got != "" {
		t.Fatalf("Conversation_id = %q, want empty", got)
	}
	if got := gjson.GetBytes(rt.body, "tool_choice").String(); got != "auto" {
		t.Fatalf("tool_choice = %q, want %q", got, "auto")
	}
	tools := gjson.GetBytes(rt.body, "tools")
	if !tools.IsArray() || len(tools.Array()) != 2 {
		t.Fatalf("tools should be normalized array of length 2, got %s", tools.Raw)
	}
	if !gjson.GetBytes(rt.body, `tools.#(name=="echo")`).Exists() {
		t.Fatalf("echo function tool not found: %s", tools.Raw)
	}
	if !gjson.GetBytes(rt.body, `tools.#(type=="web_search")`).Exists() {
		t.Fatalf("web_search tool not found: %s", tools.Raw)
	}
}

func TestCodexHttpRequestBackfillsPromptCacheKeyFromSessionHeader(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
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

func TestResolveCodexSessionIDReusesCacheUntilDeletion(t *testing.T) {
	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexTestAuth("resolve-session-cache")
	request := cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"hi"}`),
	}

	sessionID1, cacheKey := exec.resolveCodexSessionID(context.Background(), auth, sdktranslator.FromString("openai-response"), request)
	if sessionID1 == "" {
		t.Fatal("first sessionID should not be empty")
	}
	if cacheKey == "" {
		t.Fatal("cacheKey should not be empty")
	}

	sessionID2, cacheKey2 := exec.resolveCodexSessionID(context.Background(), auth, sdktranslator.FromString("openai-response"), request)
	if cacheKey2 != cacheKey {
		t.Fatalf("cacheKey mismatch: %q != %q", cacheKey2, cacheKey)
	}
	if sessionID2 != sessionID1 {
		t.Fatalf("sessionID = %q, want %q", sessionID2, sessionID1)
	}

	deleteCodexCache(cacheKey)
	sessionID3, _ := exec.resolveCodexSessionID(context.Background(), auth, sdktranslator.FromString("openai-response"), request)
	if sessionID3 == sessionID1 {
		t.Fatalf("sessionID should rotate after cache deletion, still %q", sessionID3)
	}
}

func TestCodexSessionCacheIsAuthScopedAcrossModels(t *testing.T) {
	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexTestAuth("auth-scope-cross-model")

	req1 := cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"hi-1"}`),
	}
	httpReq1, continuity1, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), "https://example.invalid/backend-api/codex/responses", req1, cliproxyexecutor.Options{}, req1.Payload)
	if err != nil {
		t.Fatalf("cacheHelper error (first model): %v", err)
	}
	body1, err := io.ReadAll(httpReq1.Body)
	if err != nil {
		t.Fatalf("read request body (first model): %v", err)
	}

	req2 := cliproxyexecutor.Request{
		Model:   "gpt-5-mini",
		Payload: []byte(`{"model":"gpt-5-mini","input":"hi-2"}`),
	}
	httpReq2, continuity2, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), "https://example.invalid/backend-api/codex/responses", req2, cliproxyexecutor.Options{}, req2.Payload)
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
	auth := newCodexTestAuth("explicit-does-not-poison")
	url := "https://example.invalid/backend-api/codex/responses"

	initialReq := cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"first"}`),
	}
	httpReq1, continuity1, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, initialReq, cliproxyexecutor.Options{}, initialReq.Payload)
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

	explicitReq := cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"second","prompt_cache_key":"explicit-cache-key"}`),
	}
	httpReq2, _, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, explicitReq, cliproxyexecutor.Options{}, explicitReq.Payload)
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

	httpReq3, continuity3, err := exec.cacheHelper(context.Background(), auth, sdktranslator.FromString("openai-response"), url, initialReq, cliproxyexecutor.Options{}, initialReq.Payload)
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
	rt := &captureRoundTripper{
		responses: []captureHTTPResponse{
			{status: http.StatusInternalServerError, respBody: `{"error":"first failure"}`},
			{status: http.StatusOK, respCT: "text/event-stream", respBody: "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n"},
		},
	}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := newCodexTestAuth("retry-after-error")
	resp, err := exec.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":[{"role":"user","content":"retry me"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
	})
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
		cliproxyexecutor.Request{
			Model:   "gpt-5",
			Payload: []byte(`{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}]}`),
		},
		cliproxyexecutor.Options{},
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

func TestNormalizeCodexToolsListPreservesStrictAndInputSchema(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"tool_choice":{"type":"function","name":"exec_command"},
		"tools":[
			{
				"name":"Task",
				"description":"Launch task",
				"input_schema":{
					"$schema":"https://json-schema.org/draft/2020-12/schema",
					"type":"object",
					"properties":{"prompt":{"type":"string"}},
					"required":["prompt"]
				}
			},
			{
				"type":"function",
				"name":"write_stdin",
				"description":"write stdin",
				"strict":true,
				"parameters":{"type":"object","properties":{"session_id":{"type":"number"}}}
			}
		]
	}`)

	out := normalizeCodexToolsList(body)
	if got := gjson.GetBytes(out, "tool_choice").String(); got != "auto" {
		t.Fatalf("tool_choice = %q, want %q", got, "auto")
	}
	if got := gjson.GetBytes(out, `tools.#(name=="Task").parameters.properties.prompt.type`).String(); got != "string" {
		t.Fatalf("Task.parameters.properties.prompt.type = %q, want %q", got, "string")
	}
	if gjson.GetBytes(out, `tools.#(name=="Task").parameters.$schema`).Exists() {
		t.Fatal("Task.parameters.$schema should be removed")
	}
	if got := gjson.GetBytes(out, `tools.#(name=="write_stdin").strict`).Bool(); !got {
		t.Fatalf("write_stdin.strict = %v, want true", got)
	}
	if !gjson.GetBytes(out, `tools.#(type=="web_search")`).Exists() {
		t.Fatalf("web_search tool not found: %s", gjson.GetBytes(out, "tools").Raw)
	}
}
