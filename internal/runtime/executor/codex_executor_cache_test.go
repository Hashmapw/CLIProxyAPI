package executor

import (
	"context"
	"io"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_StablePromptCacheKeyFromAPIKey(t *testing.T) {
	recorder := httptest.NewRecorder()
	ginCtx, _ := gin.CreateTestContext(recorder)
	ginCtx.Set("apiKey", "test-api-key")

	ctx := context.WithValue(context.Background(), "gin", ginCtx)
	executor := &CodexExecutor{}
	rawJSON := []byte(`{"model":"gpt-5.3-codex","stream":true}`)
	req := cliproxyexecutor.Request{
		Model:   "gpt-5.3-codex",
		Payload: []byte(`{"model":"gpt-5.3-codex"}`),
	}
	url := "https://example.com/responses"

	httpReq, continuity, attemptBody, err := executor.cacheHelper(ctx, nil, sdktranslator.FromString("openai"), url, req, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
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

	httpReq2, _, _, err := executor.cacheHelper(ctx, nil, sdktranslator.FromString("openai"), url, req, cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FromString("openai"),
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
