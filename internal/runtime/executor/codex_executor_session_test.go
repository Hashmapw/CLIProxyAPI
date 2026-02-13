package executor

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strconv"
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
	if got := len(tools.Array()); got != 2 {
		t.Fatalf("tools length = %d, want 2", got)
	}
	assertHasFunctionToolByName(t, rt.body, "echo")
	assertHasWebSearchTool(t, rt.body)
	if got := gjson.GetBytes(rt.body, "parallel_tool_calls").Bool(); !got {
		t.Fatalf("parallel_tool_calls should be true")
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

	if got := gjson.GetBytes(rt.body, "parallel_tool_calls").Bool(); !got {
		t.Fatalf("parallel_tool_calls should be true")
	}
	tools := gjson.GetBytes(rt.body, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	if got := len(tools.Array()); got != 2 {
		t.Fatalf("tools length = %d, want 2", got)
	}
	assertHasFunctionToolByName(t, rt.body, "6666")
	assertHasWebSearchTool(t, rt.body)
}

func TestNormalizeCodexToolsListForExecutePath(t *testing.T) {
	body := []byte(`{"model":"gpt-5.3-codex","tools":["6666"],"parallel_tool_calls":true}`)
	out := normalizeCodexToolsList(body)

	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); !got {
		t.Fatalf("parallel_tool_calls should be true")
	}
	assertHasFunctionToolByName(t, out, "6666")
	assertHasWebSearchTool(t, out)
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

	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); !got {
		t.Fatalf("parallel_tool_calls should be true")
	}
	tools := gjson.GetBytes(out, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	if got := len(tools.Array()); got == 0 {
		t.Fatalf("tools should not be empty")
	}
	assertHasFunctionToolByName(t, out, "code_interpreter")
	assertHasFunctionToolByName(t, out, "computer_use_preview")
	assertHasFunctionToolByName(t, out, "6666")
	assertHasWebSearchTool(t, out)
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
	if got := gjson.GetBytes(out, "parallel_tool_calls").Bool(); !got {
		t.Fatalf("parallel_tool_calls should be true")
	}
}

func TestNormalizeCodexToolsListSetsDefaultWhenToolsMissing(t *testing.T) {
	body := []byte(`{"model":"gpt-5.3-codex","input":"hello"}`)
	out := normalizeCodexToolsList(body)

	assertHasOnlyWebSearchTool(t, out)
}

func TestNormalizeCodexToolsListConvertsInputSchemaToFunctionParameters(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
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
			}
		]
	}`)
	out := normalizeCodexToolsList(body)

	assertHasFunctionToolByName(t, out, "Task")
	if got := gjson.GetBytes(out, "tools.0.type").String(); got != "function" {
		t.Fatalf("tools.0.type = %q, want %q", got, "function")
	}
	if got := gjson.GetBytes(out, "tools.0.parameters.type").String(); got != "object" {
		t.Fatalf("tools.0.parameters.type = %q, want %q", got, "object")
	}
	if got := gjson.GetBytes(out, "tools.0.parameters.properties.prompt.type").String(); got != "string" {
		t.Fatalf("tools.0.parameters.properties.prompt.type = %q, want %q", got, "string")
	}
	if gjson.GetBytes(out, "tools.0.input_schema").Exists() {
		t.Fatalf("tools.0.input_schema should be removed")
	}
	if gjson.GetBytes(out, "tools.0.parameters.$schema").Exists() {
		t.Fatalf("tools.0.parameters.$schema should be removed")
	}
	assertHasWebSearchTool(t, out)
}

func TestNormalizeCodexToolsListPreservesStrictField(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"tools":[
			{
				"type":"function",
				"name":"exec_command",
				"description":"run command",
				"strict":false,
				"parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}
			},
			{
				"type":"function",
				"name":"write_stdin",
				"description":"write stdin",
				"strict":true,
				"parameters":{"type":"object","properties":{"session_id":{"type":"number"}},"required":["session_id"]}
			}
		]
	}`)
	out := normalizeCodexToolsList(body)

	if !gjson.GetBytes(out, `tools.#(name=="exec_command").strict`).Exists() {
		t.Fatalf("exec_command.strict should exist")
	}
	if got := gjson.GetBytes(out, `tools.#(name=="exec_command").strict`).Bool(); got {
		t.Fatalf("exec_command.strict = %v, want false", got)
	}
	if !gjson.GetBytes(out, `tools.#(name=="write_stdin").strict`).Exists() {
		t.Fatalf("write_stdin.strict should exist")
	}
	if got := gjson.GetBytes(out, `tools.#(name=="write_stdin").strict`).Bool(); !got {
		t.Fatalf("write_stdin.strict = %v, want true", got)
	}
	assertHasWebSearchTool(t, out)
	assertToolChoiceAuto(t, out)
}

func TestNormalizeCodexToolsListForcesToolChoiceAuto(t *testing.T) {
	body := []byte(`{
		"model":"gpt-5.3-codex",
		"tool_choice":{"type":"function","name":"exec_command"},
		"tools":[{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}]
	}`)
	out := normalizeCodexToolsList(body)
	assertToolChoiceAuto(t, out)
}

func TestCodexHttpRequestForcesToolChoiceAuto(t *testing.T) {
	rt := &captureRoundTripper{status: http.StatusOK, respBody: `{"ok":true}`}
	ctx := context.WithValue(context.Background(), "cliproxy.roundtripper", http.RoundTripper(rt))

	exec := NewCodexExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"api_key": "test-key"}}
	reqURL, errURL := url.Parse("https://example.invalid/backend-api/codex/responses")
	if errURL != nil {
		t.Fatalf("parse url failed: %v", errURL)
	}
	bodyJSON := `{"model":"gpt-5","tool_choice":"none","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}}],"input":"hi"}`
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
	assertToolChoiceAuto(t, rt.body)
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

func assertHasOnlyWebSearchTool(t *testing.T, raw []byte) {
	t.Helper()
	tools := gjson.GetBytes(raw, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	if got := len(tools.Array()); got != 1 {
		t.Fatalf("tools length = %d, want 1", got)
	}
	assertHasWebSearchTool(t, raw)
}

func assertHasWebSearchTool(t *testing.T, raw []byte) {
	t.Helper()
	tools := gjson.GetBytes(raw, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	arr := tools.Array()
	for i := range arr {
		if gjson.GetBytes(raw, "tools."+strconv.Itoa(i)+".type").String() != "web_search" {
			continue
		}
		if got := gjson.GetBytes(raw, "tools."+strconv.Itoa(i)+".external_web_access").Bool(); !got {
			t.Fatalf("tools.%d.external_web_access should be true", i)
		}
		return
	}
	t.Fatalf("web_search tool not found")
}

func assertHasFunctionToolByName(t *testing.T, raw []byte, name string) {
	t.Helper()
	tools := gjson.GetBytes(raw, "tools")
	if !tools.IsArray() {
		t.Fatalf("tools should be array, got: %s", tools.Raw)
	}
	arr := tools.Array()
	for i := range arr {
		if gjson.GetBytes(raw, "tools."+strconv.Itoa(i)+".type").String() != "function" {
			continue
		}
		if gjson.GetBytes(raw, "tools."+strconv.Itoa(i)+".name").String() == name {
			return
		}
	}
	t.Fatalf("function tool %q not found", name)
}

func assertToolChoiceAuto(t *testing.T, raw []byte) {
	t.Helper()
	if got := gjson.GetBytes(raw, "tool_choice").String(); got != "auto" {
		t.Fatalf("tool_choice = %q, want %q", got, "auto")
	}
}
