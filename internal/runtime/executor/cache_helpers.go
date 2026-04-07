package executor

import (
	"sync"
	"time"
)

type codexCache struct {
	ID     string
	Expire time.Time
}

// codexCacheMap stores auth-scoped Codex session continuity IDs.
// Protected by codexCacheMu. Entries expire after codexSessionTTL.
var (
	codexCacheMap = make(map[string]codexCache)
	codexCacheMu  sync.RWMutex
)

const codexSessionTTL = 3 * time.Hour

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

// getCodexCache retrieves a cached entry, returning ok=false if not found or expired.
func getCodexCache(key string) (codexCache, bool) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.RLock()
	cache, ok := codexCacheMap[key]
	codexCacheMu.RUnlock()
	if !ok || cache.Expire.Before(time.Now()) {
		return codexCache{}, false
	}
	return cache, true
}

// setCodexCache stores a cache entry.
func setCodexCache(key string, cache codexCache) {
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.Lock()
	codexCacheMap[key] = cache
	codexCacheMu.Unlock()
}

func deleteCodexCache(key string) {
	if key == "" {
		return
	}
	codexCacheCleanupOnce.Do(startCodexCacheCleanup)
	codexCacheMu.Lock()
	delete(codexCacheMap, key)
	codexCacheMu.Unlock()
}
