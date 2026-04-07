package executor

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type codexContinuity struct {
	Key      string
	Source   string
	CacheKey string
}

func ginContextFrom(ctx context.Context) *gin.Context {
	if ctx == nil {
		return nil
	}
	ginCtx, _ := ctx.Value("gin").(*gin.Context)
	return ginCtx
}

func logWithRequestID(ctx context.Context) *log.Entry {
	return helps.LogWithRequestID(ctx)
}

func apiKeyFromContext(ctx context.Context) string {
	return helps.APIKeyFromContext(ctx)
}

func (c codexContinuity) shouldPersistManagedCache() bool {
	return c.CacheKey != "" && (c.Source == "session_cache" || c.Source == "generated_uuid_v7")
}

func (c codexContinuity) allowsFixedSessionRetry() bool {
	return c.shouldPersistManagedCache()
}

func clearCodexContinuityCache(continuity codexContinuity) {
	if continuity.shouldPersistManagedCache() {
		deleteCodexCache(continuity.CacheKey)
	}
}

func shouldRetryCodexContinuity(ctx context.Context, attempt int, continuity codexContinuity, cause error) bool {
	if attempt > 0 || !continuity.allowsFixedSessionRetry() {
		return false
	}
	clearCodexContinuityCache(continuity)
	logWithRequestID(ctx).Warnf(
		"codex executor: clearing fixed session cache and retrying once (source=%s, cache_key=%s, err=%v)",
		continuity.Source,
		continuity.CacheKey,
		cause,
	)
	return true
}

func metadataString(meta map[string]any, key string) string {
	if len(meta) == 0 {
		return ""
	}
	raw, ok := meta[key]
	if !ok || raw == nil {
		return ""
	}
	switch v := raw.(type) {
	case string:
		return strings.TrimSpace(v)
	case []byte:
		return strings.TrimSpace(string(v))
	default:
		return ""
	}
}

func resolveCodexContinuity(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) codexContinuity {
	var incomingHeaders http.Header
	if ginCtx := ginContextFrom(ctx); ginCtx != nil && ginCtx.Request != nil {
		incomingHeaders = ginCtx.Request.Header
	}

	continuity := codexContinuity{
		CacheKey: buildCodexSessionCacheKey(auth, req.Payload, incomingHeaders, codexSessionFallbackAPIKey(ctx, auth)),
	}
	incomingSessionID := strings.TrimSpace(incomingHeaders.Get("Session_id"))
	incomingConversationID := strings.TrimSpace(incomingHeaders.Get("Conversation_id"))
	incomingPromptCacheKey := codexPromptCacheKey(req.Payload)

	switch {
	case incomingSessionID != "":
		continuity.Key = incomingSessionID
		continuity.Source = "session_id_header"
	case incomingPromptCacheKey != "":
		continuity.Key = incomingPromptCacheKey
		continuity.Source = "prompt_cache_key"
	case incomingConversationID != "":
		continuity.Key = incomingConversationID
		continuity.Source = "conversation_id"
	case metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey) != "":
		continuity.Key = metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey)
		continuity.Source = "execution_session"
	case strings.TrimSpace(incomingHeaders.Get("Idempotency-Key")) != "":
		continuity.Key = strings.TrimSpace(incomingHeaders.Get("Idempotency-Key"))
		continuity.Source = "idempotency_key"
	default:
		continuity = resolveManagedCodexContinuity(
			ctx,
			continuity.CacheKey,
			fmt.Sprintf("resolve codex continuity miss (from=%s)", strings.TrimSpace(opts.SourceFormat.String())),
		)
	}
	return continuity
}

func applyCodexContinuityBody(rawJSON []byte, continuity codexContinuity) []byte {
	if continuity.Key == "" {
		return rawJSON
	}
	rawJSON, _ = sjson.SetBytes(rawJSON, "prompt_cache_key", continuity.Key)
	return setPromptCacheKeyInContexts(rawJSON, continuity.Key)
}

func applyCodexContinuityHeaders(headers http.Header, continuity codexContinuity) {
	if headers == nil || continuity.Key == "" {
		return
	}
	headers.Set("Session_id", continuity.Key)
}

func logCodexRequestDiagnostics(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, headers http.Header, body []byte, continuity codexContinuity) {
	if !log.IsLevelEnabled(log.DebugLevel) {
		return
	}
	entry := logWithRequestID(ctx)
	authID := ""
	authFile := ""
	if auth != nil {
		authID = strings.TrimSpace(auth.ID)
		authFile = strings.TrimSpace(auth.FileName)
	}
	selectedAuthID := metadataString(opts.Metadata, cliproxyexecutor.SelectedAuthMetadataKey)
	executionSessionID := metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey)
	entry.Debugf(
		"codex request diagnostics auth_id=%s selected_auth_id=%s auth_file=%s exec_session=%s continuity_source=%s continuity_cache_key=%s session_id=%s prompt_cache_key=%s prompt_cache_retention=%s store=%t has_instructions=%t reasoning_effort=%s reasoning_summary=%s chatgpt_account_id=%t originator=%s model=%s source_format=%s",
		authID,
		selectedAuthID,
		authFile,
		executionSessionID,
		continuity.Source,
		continuity.CacheKey,
		strings.TrimSpace(headers.Get("Session_id")),
		gjson.GetBytes(body, "prompt_cache_key").String(),
		gjson.GetBytes(body, "prompt_cache_retention").String(),
		gjson.GetBytes(body, "store").Bool(),
		gjson.GetBytes(body, "instructions").Exists(),
		gjson.GetBytes(body, "reasoning.effort").String(),
		gjson.GetBytes(body, "reasoning.summary").String(),
		strings.TrimSpace(headers.Get("Chatgpt-Account-Id")) != "",
		strings.TrimSpace(headers.Get("Originator")),
		req.Model,
		opts.SourceFormat.String(),
	)
}
