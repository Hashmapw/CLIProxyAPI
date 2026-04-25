package helps

import (
	"sync"
	"time"
)

type CodexCache struct {
	ID     string
	Expire time.Time
}

// codexCacheMap stores lightweight continuity entries keyed by caller-defined scopes.
// Protected by codexCacheMu. Entries carry their own expiration times.
var (
	codexCacheMap = make(map[string]CodexCache)
	codexCacheMu  sync.RWMutex
)

// codexCacheCleanupInterval controls how often expired entries are purged.
const codexCacheCleanupInterval = 15 * time.Minute

// codexCacheCleanupOnce ensures the background cleanup goroutine starts only once.
var codexCacheCleanupOnce sync.Once

// startCodexCacheCleanup launches a background goroutine that periodically
// removes expired entries from codexCacheMap to prevent memory leaks.
func startCodexCacheCleanup() {
	go func() {
		ticker := time.NewTicker(codexCacheCleanupInterval)
		defer ticker.Stop()
		for range ticker.C {
			purgeExpiredCodexCache()
		}
	}()
}

// purgeExpiredCodexCache removes entries that have expired.
func purgeExpiredCodexCache() {
	now := time.Now()
	codexCacheMu.Lock()
	defer codexCacheMu.Unlock()
	for key, cache := range codexCacheMap {
		if cache.Expire.Before(now) {
			delete(codexCacheMap, key)
		}
	}
}

// GetCodexCache retrieves a cached entry, returning ok=false if not found or expired.
func GetCodexCache(key string) (CodexCache, bool) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.RLock()
	cache, ok := codexCacheMap[key]
	codexCacheMu.RUnlock()
	if !ok || cache.Expire.Before(time.Now()) {
		return CodexCache{}, false
	}
	return cache, true
}

// SetCodexCache stores a cache entry.
func SetCodexCache(key string, cache CodexCache) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.Lock()
	codexCacheMap[key] = cache
	codexCacheMu.Unlock()
}

// DeleteCodexCache removes a cache entry if present.
func DeleteCodexCache(key string) {
	if key == "" {
		return
	}
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.Lock()
	delete(codexCacheMap, key)
	codexCacheMu.Unlock()
}
