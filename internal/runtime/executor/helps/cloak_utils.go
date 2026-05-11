package helps

import (
	"crypto/sha256"
	"encoding/hex"
	"regexp"
	"strings"

	"github.com/google/uuid"
)

// userIDPattern matches Claude Code's current generated user ID format:
// user_[64-hex]_account__session_[uuid].
var userIDPattern = regexp.MustCompile(`^user_[a-fA-F0-9]{64}_account__session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// legacyUserIDPattern accepts older cached or caller-supplied IDs generated
// before the account segment was removed from the visible format.
var legacyUserIDPattern = regexp.MustCompile(`^user_[a-fA-F0-9]{64}_account_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_session_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)

// generateFakeUserID generates a fake user ID in Claude Code format.
// Format: user_[64-hex-chars]_account__session_[UUID-v4]
func generateFakeUserID() string {
	accountUUID := uuid.New().String()
	sessionUUID := uuid.New().String()
	sum := sha256.Sum256([]byte(accountUUID + sessionUUID))
	hexPart := hex.EncodeToString(sum[:])
	return "user_" + hexPart + "_account__session_" + sessionUUID
}

// isValidUserID checks if a user ID matches Claude Code format.
func isValidUserID(userID string) bool {
	return userIDPattern.MatchString(userID) || legacyUserIDPattern.MatchString(userID)
}

func GenerateFakeUserID() string {
	return generateFakeUserID()
}

func IsValidUserID(userID string) bool {
	return isValidUserID(userID)
}

// ShouldCloak determines if request should be cloaked based on config and client User-Agent.
// Returns true if cloaking should be applied.
func ShouldCloak(cloakMode string, userAgent string) bool {
	switch strings.ToLower(cloakMode) {
	case "always":
		return true
	case "never":
		return false
	default: // "auto" or empty
		// If client is Claude Code, don't cloak
		return !strings.HasPrefix(userAgent, "claude-cli")
	}
}

// isClaudeCodeClient checks if the User-Agent indicates a Claude Code client.
func isClaudeCodeClient(userAgent string) bool {
	return strings.HasPrefix(userAgent, "claude-cli")
}
