package executor

import (
	"context"
	"io"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestCodexExecutorCacheHelper_OpenAIChatCompletions_ReusesSessionUntilCacheDeletion(t *testing.T) {
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

	httpReq, cacheKey, err := executor.cacheHelper(ctx, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error: %v", err)
	}

	body, errRead := io.ReadAll(httpReq.Body)
	if errRead != nil {
		t.Fatalf("read request body: %v", errRead)
	}

	firstKey := gjson.GetBytes(body, "prompt_cache_key").String()
	if firstKey == "" {
		t.Fatal("prompt_cache_key should not be empty")
	}
	if gotConversation := httpReq.Header.Get("Conversation_id"); gotConversation != "" {
		t.Fatalf("Conversation_id = %q, want empty", gotConversation)
	}
	if gotSession := httpReq.Header.Get("Session_id"); gotSession != firstKey {
		t.Fatalf("Session_id = %q, want %q", gotSession, firstKey)
	}

	httpReq2, cacheKey2, err := executor.cacheHelper(ctx, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error (second call): %v", err)
	}
	if cacheKey2 != cacheKey {
		t.Fatalf("cacheKey mismatch: %q != %q", cacheKey2, cacheKey)
	}
	body2, errRead2 := io.ReadAll(httpReq2.Body)
	if errRead2 != nil {
		t.Fatalf("read request body (second call): %v", errRead2)
	}
	gotKey2 := gjson.GetBytes(body2, "prompt_cache_key").String()
	if gotKey2 != firstKey {
		t.Fatalf("prompt_cache_key (second call) = %q, want %q", gotKey2, firstKey)
	}

	deleteCodexCache(cacheKey)
	httpReq3, _, err := executor.cacheHelper(ctx, sdktranslator.FromString("openai"), url, req, rawJSON)
	if err != nil {
		t.Fatalf("cacheHelper error (third call): %v", err)
	}
	body3, errRead3 := io.ReadAll(httpReq3.Body)
	if errRead3 != nil {
		t.Fatalf("read request body (third call): %v", errRead3)
	}
	gotKey3 := gjson.GetBytes(body3, "prompt_cache_key").String()
	if gotKey3 == firstKey {
		t.Fatalf("prompt_cache_key should rotate after cache deletion, still %q", gotKey3)
	}
}
