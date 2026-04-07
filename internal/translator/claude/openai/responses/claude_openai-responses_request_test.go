package responses

import (
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	"github.com/tidwall/gjson"
)

func resetClaudeResponsesUserIDState() {
	user = ""
	account = ""
	session = ""
}

func TestConvertOpenAIResponsesRequestToClaude_GeneratesEmptyAccountUserID(t *testing.T) {
	resetClaudeResponsesUserIDState()
	t.Cleanup(resetClaudeResponsesUserIDState)

	result := ConvertOpenAIResponsesRequestToClaude(
		"claude-sonnet-4-5",
		[]byte(`{"input":"hello"}`),
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
