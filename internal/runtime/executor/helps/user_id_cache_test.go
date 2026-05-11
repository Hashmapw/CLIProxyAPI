package helps

import (
	"strings"
	"testing"
	"time"
)

func resetUserIDCache() {
	userIDCacheMu.Lock()
	userIDCache = make(map[string]userIDCacheEntry)
	userIDCacheMu.Unlock()
}

func TestCachedUserID_ReusesWithinTTL(t *testing.T) {
	resetUserIDCache()

	first := CachedUserID("api-key-1")
	second := CachedUserID("api-key-1")

	if first == "" {
		t.Fatal("expected generated user_id to be non-empty")
	}
	if first != second {
		t.Fatalf("expected cached user_id to be reused, got %q and %q", first, second)
	}
}

func TestCachedUserID_ExpiresAfterTTL(t *testing.T) {
	resetUserIDCache()

	expiredID := CachedUserID("api-key-expired")
	cacheKey := userIDCacheKey("api-key-expired")
	userIDCacheMu.Lock()
	userIDCache[cacheKey] = userIDCacheEntry{
		value:  expiredID,
		expire: time.Now().Add(-time.Minute),
	}
	userIDCacheMu.Unlock()

	newID := CachedUserID("api-key-expired")
	if newID == expiredID {
		t.Fatalf("expected expired user_id to be replaced, got %q", newID)
	}
	if newID == "" {
		t.Fatal("expected regenerated user_id to be non-empty")
	}
}

func TestCachedUserID_IsScopedByAPIKey(t *testing.T) {
	resetUserIDCache()

	first := CachedUserID("api-key-1")
	second := CachedUserID("api-key-2")

	if first == second {
		t.Fatalf("expected different API keys to have different user_ids, got %q", first)
	}
}

func TestCachedUserID_RenewsTTLOnHit(t *testing.T) {
	resetUserIDCache()

	key := "api-key-renew"
	id := CachedUserID(key)
	cacheKey := userIDCacheKey(key)

	soon := time.Now()
	userIDCacheMu.Lock()
	userIDCache[cacheKey] = userIDCacheEntry{
		value:  id,
		expire: soon.Add(2 * time.Second),
	}
	userIDCacheMu.Unlock()

	if refreshed := CachedUserID(key); refreshed != id {
		t.Fatalf("expected cached user_id to be reused before expiry, got %q", refreshed)
	}

	userIDCacheMu.RLock()
	entry := userIDCache[cacheKey]
	userIDCacheMu.RUnlock()

	if entry.expire.Sub(soon) < 30*time.Minute {
		t.Fatalf("expected TTL to renew, got %v remaining", entry.expire.Sub(soon))
	}
}

func TestUserIDFormatValidation_AcceptsCurrentAndLegacyFormats(t *testing.T) {
	generated := GenerateFakeUserID()
	if !IsValidUserID(generated) {
		t.Fatalf("generated user_id should be valid, got %q", generated)
	}
	if !strings.Contains(generated, "_account__session_") {
		t.Fatalf("generated user_id should use account__session format, got %q", generated)
	}

	hexPart := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	legacy := "user_" + hexPart + "_account_11111111-2222-3333-4444-555555555555_session_66666666-7777-8888-9999-aaaaaaaaaaaa"
	if !IsValidUserID(legacy) {
		t.Fatalf("legacy user_id should remain valid, got %q", legacy)
	}

	invalid := "user_" + hexPart + "_account_11111111-2222-3333_session_66666666-7777-8888-9999-aaaaaaaaaaaa"
	if IsValidUserID(invalid) {
		t.Fatalf("malformed user_id should be invalid, got %q", invalid)
	}
}
