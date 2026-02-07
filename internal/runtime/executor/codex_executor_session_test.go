package executor

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

type captureRoundTripper struct {
	body       []byte
	headers    http.Header
	status     int
	respBody   string
	respCT     string
	requestURL string
}

func (rt *captureRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	body, _ := io.ReadAll(req.Body)
	rt.body = body
	rt.headers = req.Header.Clone()
	rt.requestURL = req.URL.String()

	status := rt.status
	if status == 0 {
		status = http.StatusBadRequest
	}
	contentType := rt.respCT
	if strings.TrimSpace(contentType) == "" {
		contentType = "application/json"
	}
	return &http.Response{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{contentType},
		},
		Body:    io.NopCloser(strings.NewReader(rt.respBody)),
		Request: req,
	}, nil
}

func TestCodexExecuteInjectsPromptCacheKeyAndSessionHeaders(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusBadRequest, respBody: `{"error":"mock"}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"api_key":  "test-key",
		"base_url": "https://example.invalid/backend-api/codex",
	}}
	_, err := exec.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":[{"role":"user","content":"hi"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       false,
	})
	if err == nil {
		t.Fatalf("expected upstream error")
	}
	assertSessionTripleInCapturedRequest(t, rt)
}

func TestCodexExecuteCompactInjectsPromptCacheKeyAndSessionHeaders(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusBadRequest, respBody: `{"error":"mock"}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"api_key":  "test-key",
		"base_url": "https://example.invalid/backend-api/codex",
	}}
	_, err := exec.Execute(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":[{"role":"user","content":"hi"}]}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Alt:          "responses/compact",
		Stream:       false,
	})
	if err == nil {
		t.Fatalf("expected upstream error")
	}
	if !strings.HasSuffix(rt.requestURL, "/responses/compact") {
		t.Fatalf("unexpected request URL: %s", rt.requestURL)
	}
	assertSessionTripleInCapturedRequest(t, rt)
}

func TestCodexExecuteStreamInjectsPromptCacheKeyAndSessionHeaders(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusBadRequest, respBody: `{"error":"mock"}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{
		"api_key":  "test-key",
		"base_url": "https://example.invalid/backend-api/codex",
	}}
	_, err := exec.ExecuteStream(ctx, auth, cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":[{"role":"user","content":"hi"}],"stream":true}`),
	}, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai-response"),
		Stream:       true,
	})
	if err == nil {
		t.Fatalf("expected upstream error")
	}
	assertSessionTripleInCapturedRequest(t, rt)
}

func TestCodexHttpRequestBackfillsPromptCacheKeyFromSessionHeader(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
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

	promptCacheKey := gjson.GetBytes(rt.body, "prompt_cache_key").String()
	if promptCacheKey != "session-from-header" {
		t.Fatalf("prompt_cache_key = %q, want %q", promptCacheKey, "session-from-header")
	}
	if got := strings.TrimSpace(rt.headers.Get("Session_id")); got != "session-from-header" {
		t.Fatalf("Session_id = %q, want %q", got, "session-from-header")
	}
	if got := strings.TrimSpace(rt.headers.Get("Conversation_id")); got != "" {
		t.Fatalf("Conversation_id should be empty, got %q", got)
	}
}

func TestResolveCodexSessionIDReusesIncomingSessionHeader(t *testing.T) {
	exec := NewCodexExecutor(&config.Config{})
	req, _ := http.NewRequest(http.MethodPost, "http://example.invalid", strings.NewReader("{}"))
	req.Header.Set("Session_id", "session-only")
	req.Header.Set("Conversation_id", "session-only")

	ginCtx := &gin.Context{Request: req}
	ctx := context.WithValue(context.Background(), "gin", ginCtx)

	sessionID, _ := exec.resolveCodexSessionID(ctx, sdktranslator.FromString("openai-response"), cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"hi"}`),
	})
	if sessionID != "session-only" {
		t.Fatalf("sessionID = %q, want %q", sessionID, "session-only")
	}
}

func TestCodexHttpRequestNormalizesToolsToArray(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
	}
	bodyJSON := `{"model":"gpt-5","input":"hi","parallel_tool_calls":true,"tools":{"type":"function","name":"echo","description":"x","parameters":{"type":"object"}}}`
	httpReq := &http.Request{
		Method: http.MethodPost,
		URL:    reqURL,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body:          io.NopCloser(strings.NewReader(bodyJSON)),
		ContentLength: int64(len(bodyJSON)),
	}

	resp, errReq := exec.HttpRequest(ctx, auth, httpReq)
	if errReq != nil {
		t.Fatalf("HttpRequest error: %v", errReq)
	}
	if resp != nil && resp.Body != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	tools := gjson.GetBytes(rt.body, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	assertDefaultWebSearchOnly(t, rt.body)
	if got := gjson.GetBytes(rt.body, "parallel_tool_calls").Bool(); got {
		t.Fatalf("parallel_tool_calls should be false")
	}
}

func TestCodexHttpRequestNormalizesStringToolsToFunctionObjects(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
	}
	bodyJSON := `{"model":"gpt-5.3-codex","input":"hello","tools":["6666"],"parallel_tool_calls":true}`
	httpReq := &http.Request{
		Method: http.MethodPost,
		URL:    reqURL,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body:          io.NopCloser(strings.NewReader(bodyJSON)),
		ContentLength: int64(len(bodyJSON)),
	}

	resp, errReq := exec.HttpRequest(ctx, auth, httpReq)
	if errReq != nil {
		t.Fatalf("HttpRequest error: %v", errReq)
	}
	if resp != nil && resp.Body != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	if got := gjson.GetBytes(rt.body, "parallel_tool_calls").Bool(); got {
		t.Fatalf("parallel_tool_calls should be false")
	}
	tools := gjson.GetBytes(rt.body, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	assertDefaultWebSearchOnly(t, rt.body)
}

func TestNormalizeCodexToolsListForExecutePath(t *testing.T) {
	body := []byte(`{"model":"gpt-5.3-codex","tools":["6666"],"parallel_tool_calls":true}`)
	out := normalizeCodexToolsList(body)

	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); got {
		t.Fatalf("parallel_tool_calls should be false")
	}
	assertDefaultWebSearchOnly(t, out)
}

func TestNormalizeCodexToolsListBlocksOSLevelBuiltins(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"tools":[
			{"type":"code_interpreter"},
			{"type":"computer_use_preview"},
			{"type":"web_search"},
			"code_interpreter",
			"computer_use_preview",
			"6666"
		],
		"parallel_tool_calls":true
	}`)
	out := normalizeCodexToolsList(body)

	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); got {
		t.Fatalf("parallel_tool_calls should be false")
	}
	tools := gjson.GetBytes(out, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	assertDefaultWebSearchOnly(t, out)
}

func TestNormalizeCodexToolsListKeepsEmptyToolsArray(t *testing.T) {
	body := []byte(`{"model":"gpt-5.3-codex","tools":[],"parallel_tool_calls":true}`)
	out := normalizeCodexToolsList(body)

	tools := gjson.GetBytes(out, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	if len(tools.Array()) != 0 {
		t.Fatalf("tools length = %d, want 0", len(tools.Array()))
	}
	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); got {
		t.Fatalf("parallel_tool_calls should be false")
	}
}

func TestNormalizeCodexToolsListSetsDefaultWhenToolsMissing(t *testing.T) {
	body := []byte(`{"model":"gpt-5.3-codex","input":"hello"}`)
	out := normalizeCodexToolsList(body)

	assertDefaultWebSearchOnly(t, out)
}

func TestCodexHttpRequestSetsPromptCacheKeyInContextsArray(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
	}
	bodyJSON := `{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}],"input":"hi","prompt_cache_key":"pc-1"}`
	httpReq := &http.Request{
		Method: http.MethodPost,
		URL:    reqURL,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body:          io.NopCloser(strings.NewReader(bodyJSON)),
		ContentLength: int64(len(bodyJSON)),
	}

	resp, errReq := exec.HttpRequest(ctx, auth, httpReq)
	if errReq != nil {
		t.Fatalf("HttpRequest error: %v", errReq)
	}
	if resp != nil && resp.Body != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	if got := gjson.GetBytes(rt.body, "contexts.0.prompt_cache_key").String(); got != "pc-1" {
		t.Fatalf("contexts.0.prompt_cache_key = %q, want %q", got, "pc-1")
	}
	if got := gjson.GetBytes(rt.body, "contexts.1.prompt_cache_key").String(); got != "pc-1" {
		t.Fatalf("contexts.1.prompt_cache_key = %q, want %q", got, "pc-1")
	}
}

func TestCodexHttpRequestSetsPromptCacheKeyInContextsObject(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
	}
	bodyJSON := `{"model":"gpt-5","contexts":{"id":"solo"},"input":"hi","prompt_cache_key":"pc-2"}`
	httpReq := &http.Request{
		Method: http.MethodPost,
		URL:    reqURL,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body:          io.NopCloser(strings.NewReader(bodyJSON)),
		ContentLength: int64(len(bodyJSON)),
	}

	resp, errReq := exec.HttpRequest(ctx, auth, httpReq)
	if errReq != nil {
		t.Fatalf("HttpRequest error: %v", errReq)
	}
	if resp != nil && resp.Body != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	if got := gjson.GetBytes(rt.body, "contexts.prompt_cache_key").String(); got != "pc-2" {
		t.Fatalf("contexts.prompt_cache_key = %q, want %q", got, "pc-2")
	}
}

func assertSessionTripleInCapturedRequest(t *testing.T, rt *captureRoundTripper) {
	t.Helper()
	promptCacheKey := gjson.GetBytes(rt.body, "prompt_cache_key").String()
	if strings.TrimSpace(promptCacheKey) == "" {
		t.Fatalf("prompt_cache_key is missing in request body: %s", string(rt.body))
	}
	sessionID := strings.TrimSpace(rt.headers.Get("Session_id"))
	if sessionID == "" {
		t.Fatalf("Session_id header is missing")
	}
	if sessionID != promptCacheKey {
		t.Fatalf("Session_id(%s) != prompt_cache_key(%s)", sessionID, promptCacheKey)
	}
	if conversationID := strings.TrimSpace(rt.headers.Get("Conversation_id")); conversationID != "" {
		t.Fatalf("Conversation_id should be empty, got %q", conversationID)
	}
}

func assertDefaultWebSearchOnly(t *testing.T, raw []byte) {
	t.Helper()
	tools := gjson.GetBytes(raw, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	if got := len(tools.Array()); got != 1 {
		t.Fatalf("tools length = %d, want 1", got)
	}
	if got := gjson.GetBytes(raw, "tools.0.type").String(); got != "web_search" {
		t.Fatalf("tools.0.type = %q, want %q", got, "web_search")
	}
	if got := gjson.GetBytes(raw, "tools.0.external_web_access").Bool(); !got {
		t.Fatalf("tools.0.external_web_access should be true")
	}
}
