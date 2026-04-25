package executor

import (
	"context"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const claudeUserIDTTL = 3 * time.Hour

type claudeUserIDContinuity struct {
	Key      string
	Source   string
	CacheKey string
}

func claudeUserIDCacheNamespaceKey(cacheKey string) string {
	if strings.TrimSpace(cacheKey) == "" {
		return ""
	}
	return "claude-user-id:" + strings.TrimSpace(cacheKey)
}

func buildClaudeUserIDCacheKey(auth *cliproxyauth.Auth, originalPayload []byte, fallbackAPIKey string) string {
	if auth != nil {
		if authID := strings.TrimSpace(auth.ID); authID != "" {
			return "auth:" + authID
		}
	}

	if strings.TrimSpace(fallbackAPIKey) != "" {
		fingerprint := uuid.NewSHA1(
			uuid.NameSpaceOID,
			[]byte("cli-proxy-api:claude:user-id-cache:"+strings.TrimSpace(fallbackAPIKey)),
		).String()
		return "api:" + fingerprint
	}

	userID := strings.TrimSpace(gjson.GetBytes(originalPayload, "metadata.user_id").String())
	if userID != "" {
		return "user:" + userID
	}

	return "default"
}

func resolveExplicitClaudeUserID(from sdktranslator.Format, originalPayload []byte) string {
	userID := strings.TrimSpace(gjson.GetBytes(originalPayload, "metadata.user_id").String())
	if userID == "" || !helps.IsValidUserID(userID) {
		return ""
	}

	source := strings.TrimSpace(strings.ToLower(from.String()))
	switch source {
	case "", "claude":
		return userID
	default:
		// Non-Claude source formats do not have a first-class metadata.user_id contract.
		// Treat translator-injected values as managed defaults rather than explicit continuity.
		return ""
	}
}

func (c claudeUserIDContinuity) shouldPersistManagedCache() bool {
	return c.CacheKey != "" && (c.Source == "user_id_cache" || c.Source == "generated_user_id")
}

func (c claudeUserIDContinuity) allowsFixedUserIDRetry() bool {
	return c.shouldPersistManagedCache()
}

func resolveManagedClaudeUserID(ctx context.Context, cacheKey string, missReason string) claudeUserIDContinuity {
	continuity := claudeUserIDContinuity{CacheKey: cacheKey}
	namespacedKey := claudeUserIDCacheNamespaceKey(cacheKey)
	if namespacedKey != "" {
		if cache, ok := helps.GetCodexCache(namespacedKey); ok {
			continuity.Key = strings.TrimSpace(cache.ID)
			continuity.Source = "user_id_cache"
		}
	}
	if continuity.Key == "" {
		continuity.Key = helps.GenerateFakeUserID()
		continuity.Source = "generated_user_id"
		logWithRequestID(ctx).Warnf(
			"claude executor: generated new managed metadata.user_id (reason=%s, cache_key=%s)",
			strings.TrimSpace(missReason),
			continuity.CacheKey,
		)
	}
	if continuity.shouldPersistManagedCache() {
		helps.SetCodexCache(namespacedKey, helps.CodexCache{
			ID:     continuity.Key,
			Expire: time.Now().Add(claudeUserIDTTL),
		})
	}
	return continuity
}

func clearClaudeUserIDCache(continuity claudeUserIDContinuity) {
	if !continuity.shouldPersistManagedCache() {
		return
	}
	helps.DeleteCodexCache(claudeUserIDCacheNamespaceKey(continuity.CacheKey))
}

func shouldRetryClaudeUserIDContinuity(ctx context.Context, attempt int, continuity claudeUserIDContinuity, cause error) bool {
	if !continuity.allowsFixedUserIDRetry() {
		return false
	}
	clearClaudeUserIDCache(continuity)
	if attempt > 0 {
		return false
	}
	logWithRequestID(ctx).Warnf(
		"claude executor: clearing managed metadata.user_id cache and retrying once (source=%s, cache_key=%s, err=%v)",
		continuity.Source,
		continuity.CacheKey,
		cause,
	)
	return true
}

func applyClaudeManagedUserID(ctx context.Context, auth *cliproxyauth.Auth, from sdktranslator.Format, originalPayload []byte, body []byte, apiKey string) ([]byte, claudeUserIDContinuity) {
	explicitUserID := resolveExplicitClaudeUserID(from, originalPayload)
	if explicitUserID != "" {
		current := strings.TrimSpace(gjson.GetBytes(body, "metadata.user_id").String())
		if current != explicitUserID {
			body, _ = sjson.SetBytes(body, "metadata.user_id", explicitUserID)
		}
		return body, claudeUserIDContinuity{
			Key:      explicitUserID,
			Source:   "metadata_user_id",
			CacheKey: buildClaudeUserIDCacheKey(auth, originalPayload, apiKey),
		}
	}

	continuity := resolveManagedClaudeUserID(
		ctx,
		buildClaudeUserIDCacheKey(auth, originalPayload, apiKey),
		"metadata.user_id missing or not explicit on Claude request",
	)
	if continuity.Key != "" {
		body, _ = sjson.SetBytes(body, "metadata.user_id", continuity.Key)
	}
	return body, continuity
}
