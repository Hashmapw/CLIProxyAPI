package gemini

import (
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	"github.com/tidwall/gjson"
)

func resetClaudeGeminiUserIDState() {
	user = ""
	account = ""
	session = ""
}

func TestConvertGeminiRequestToClaude_GeneratesEmptyAccountUserID(t *testing.T) {
	resetClaudeGeminiUserIDState()
	t.Cleanup(resetClaudeGeminiUserIDState)

	result := ConvertGeminiRequestToClaude(
		"claude-sonnet-4-5",
		[]byte(`{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`),
		false,
	)

	userID := gjson.GetBytes(result, "metadata.user_id").String()
	if !helps.IsValidUserID(userID) {
		t.Fatalf("expected valid metadata.user_id, got %q", userID)
	}
	if !strings.Contains(userID, "_account__session_") {
		t.Fatalf("expected metadata.user_id to use empty account segment, got %q", userID)
	}
}
