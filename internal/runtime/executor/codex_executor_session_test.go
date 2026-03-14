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
	if got := rt.headers.Get("OpenAI-Beta"); got != "responses=experimental" {
		t.Fatalf("OpenAI-Beta = %q, want %q", got, "responses=experimental")
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
	request := cliproxyexecutor.Request{
		Model:   "gpt-5",
		Payload: []byte(`{"model":"gpt-5","input":"hi"}`),
	}

	sessionID1, cacheKey := exec.resolveCodexSessionID(context.Background(), sdktranslator.FromString("openai-response"), request)
	if sessionID1 == "" {
		t.Fatal("first sessionID should not be empty")
	}
	if cacheKey == "" {
		t.Fatal("cacheKey should not be empty")
	}

	sessionID2, cacheKey2 := exec.resolveCodexSessionID(context.Background(), sdktranslator.FromString("openai-response"), request)
	if cacheKey2 != cacheKey {
		t.Fatalf("cacheKey mismatch: %q != %q", cacheKey2, cacheKey)
	}
	if sessionID2 != sessionID1 {
		t.Fatalf("sessionID = %q, want %q", sessionID2, sessionID1)
	}

	deleteCodexCache(cacheKey)
	sessionID3, _ := exec.resolveCodexSessionID(context.Background(), sdktranslator.FromString("openai-response"), request)
	if sessionID3 == sessionID1 {
		t.Fatalf("sessionID should rotate after cache deletion, still %q", sessionID3)
	}
}

func TestApplyCodexPromptCacheHeadersSetsConversationSessionAndContexts(t *testing.T) {
	body, headers, cacheKey := applyCodexPromptCacheHeaders(
		context.Background(),
		cliproxyexecutor.Request{
			Model:   "gpt-5",
			Payload: []byte(`{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}]}`),
		},
		[]byte(`{"model":"gpt-5","contexts":[{"id":"a"},{"id":"b"}]}`),
	)

	if cacheKey == "" {
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
