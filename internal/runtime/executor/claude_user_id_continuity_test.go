package executor

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/runtime/executor/helps"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestClaudeExecutor_OpenAITranslationManagedUserIDScopedByAuth(t *testing.T) {
	var userIDs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		userIDs = append(userIDs, gjson.GetBytes(body, "metadata.user_id").String())
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","model":"claude-sonnet-4-5","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	executor := NewClaudeExecutor(&config.Config{})
	payload := []byte(`{"model":"gpt-4.1","messages":[{"role":"user","content":"hi"}]}`)

	authA := &cliproxyauth.Auth{ID: "claude-user-id-auth-A-main", Attributes: map[string]string{"api_key": "shared-claude-user-id-key", "base_url": server.URL}}
	authB := &cliproxyauth.Auth{ID: "claude-user-id-auth-B-main", Attributes: map[string]string{"api_key": "shared-claude-user-id-key", "base_url": server.URL}}

	for _, auth := range []*cliproxyauth.Auth{authA, authB} {
		_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "claude-sonnet-4-5", Payload: payload}, cliproxyexecutor.Options{OriginalRequest: payload, SourceFormat: sdktranslator.FromString("openai")})
		if err != nil {
			t.Fatalf("Execute error for auth %s: %v", auth.ID, err)
		}
	}

	if len(userIDs) != 2 {
		t.Fatalf("expected 2 requests, got %d", len(userIDs))
	}
	if !helps.IsValidUserID(userIDs[0]) || !helps.IsValidUserID(userIDs[1]) {
		t.Fatalf("expected valid managed user_ids, got %q and %q", userIDs[0], userIDs[1])
	}
	if userIDs[0] == userIDs[1] {
		t.Fatalf("expected managed user_id to be auth-scoped, got identical values %q", userIDs[0])
	}
}

func TestClaudeExecutor_ExplicitUserIDDoesNotPolluteManagedCache(t *testing.T) {
	var userIDs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		userIDs = append(userIDs, gjson.GetBytes(body, "metadata.user_id").String())
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","model":"claude-3-5-sonnet","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	executor := NewClaudeExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{ID: "claude-user-id-explicit-vs-managed-main", Attributes: map[string]string{"api_key": "claude-user-id-explicit-vs-managed-key", "base_url": server.URL}}
	explicitUserID := helps.GenerateFakeUserID()

	explicitPayload := []byte(`{"model":"claude-3-5-sonnet","metadata":{"user_id":"` + explicitUserID + `"},"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)
	missingPayload := []byte(`{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

	_, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "claude-3-5-sonnet", Payload: explicitPayload}, cliproxyexecutor.Options{OriginalRequest: explicitPayload, SourceFormat: sdktranslator.FromString("claude")})
	if err != nil {
		t.Fatalf("explicit Execute error: %v", err)
	}

	for i := 0; i < 2; i++ {
		_, err = executor.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "claude-3-5-sonnet", Payload: missingPayload}, cliproxyexecutor.Options{OriginalRequest: missingPayload, SourceFormat: sdktranslator.FromString("claude")})
		if err != nil {
			t.Fatalf("managed Execute call %d error: %v", i+1, err)
		}
	}

	if len(userIDs) != 3 {
		t.Fatalf("expected 3 requests, got %d", len(userIDs))
	}
	if userIDs[0] != explicitUserID {
		t.Fatalf("expected first request to preserve explicit user_id %q, got %q", explicitUserID, userIDs[0])
	}
	if userIDs[1] == explicitUserID {
		t.Fatalf("expected managed user_id to differ from explicit value %q", explicitUserID)
	}
	if userIDs[1] != userIDs[2] {
		t.Fatalf("expected managed user_id cache to be reused after explicit request, got %q and %q", userIDs[1], userIDs[2])
	}
}

func TestClaudeExecutor_Execute_RetriesWithNewManagedUserIDAfterError(t *testing.T) {
	var attempts int
	var userIDs []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		body, _ := io.ReadAll(r.Body)
		userIDs = append(userIDs, gjson.GetBytes(body, "metadata.user_id").String())
		w.Header().Set("Content-Type", "application/json")
		if attempts == 1 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"internal_error","message":"retry me"}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","model":"claude-3-5-sonnet","role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	executor := NewClaudeExecutor(&config.Config{})
	auth := &cliproxyauth.Auth{ID: "claude-user-id-retry-execute-main", Attributes: map[string]string{"api_key": "claude-user-id-retry-execute-key", "base_url": server.URL}}
	payload := []byte(`{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)

	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "claude-3-5-sonnet", Payload: payload}, cliproxyexecutor.Options{OriginalRequest: payload, SourceFormat: sdktranslator.FromString("claude")}); err != nil {
		t.Fatalf("Execute error after retry path: %v", err)
	}
	if _, err := executor.Execute(context.Background(), auth, cliproxyexecutor.Request{Model: "claude-3-5-sonnet", Payload: payload}, cliproxyexecutor.Options{OriginalRequest: payload, SourceFormat: sdktranslator.FromString("claude")}); err != nil {
		t.Fatalf("follow-up Execute error: %v", err)
	}

	if len(userIDs) != 3 {
		t.Fatalf("expected 3 upstream requests, got %d", len(userIDs))
	}
	if userIDs[0] == userIDs[1] {
		t.Fatalf("expected retry to regenerate managed user_id, got identical values %q", userIDs[0])
	}
	if userIDs[1] != userIDs[2] {
		t.Fatalf("expected regenerated managed user_id to be reused after success, got %q and %q", userIDs[1], userIDs[2])
	}
}
