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
	if !IsValidUserID(first) || !IsValidUserID(second) {
		t.Fatalf("expected cached user_ids to be valid, got %q and %q", first, second)
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
	if !IsValidUserID(newID) {
		t.Fatalf("expected regenerated user_id to be valid, got %q", newID)
	}
}

func TestCachedUserID_IsScopedByAPIKey(t *testing.T) {
	resetUserIDCache()

	first := CachedUserID("api-key-1")
	second := CachedUserID("api-key-2")

	if !IsValidUserID(first) || !IsValidUserID(second) {
		t.Fatalf("expected scoped user_ids to be valid, got %q and %q", first, second)
	}
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
	if !IsValidUserID(id) {
		t.Fatalf("expected cached user_id to be valid, got %q", id)
	}

	userIDCacheMu.RLock()
	entry := userIDCache[cacheKey]
	userIDCacheMu.RUnlock()

	if entry.expire.Sub(soon) < 30*time.Minute {
		t.Fatalf("expected TTL to renew, got %v remaining", entry.expire.Sub(soon))
	}
}

func TestGenerateFakeUserID_UsesEmptyAccountSegment(t *testing.T) {
	userID := GenerateFakeUserID()

	if !IsValidUserID(userID) {
		t.Fatalf("expected generated user_id to be valid, got %q", userID)
	}
	if len(userID) == 0 || !strings.Contains(userID, "_account__session_") {
		t.Fatalf("expected generated user_id to use empty account segment, got %q", userID)
	}
}
