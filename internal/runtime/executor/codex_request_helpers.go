package executor

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	defaultCodexWebSearchToolJSON = `{"type":"web_search","external_web_access":true}`
	defaultCodexToolsJSON         = "[" + defaultCodexWebSearchToolJSON + "]"
)

func codexPromptCacheKey(rawJSON []byte) string {
	promptCacheKey := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt_cache_key").String())
	if promptCacheKey != "" {
		return promptCacheKey
	}
	return strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt_key_cache").String())
}

func firstNonEmptyCodexPromptCacheKey(payload []byte, rawJSON []byte) string {
	if key := codexPromptCacheKey(payload); key != "" {
		return key
	}
	return codexPromptCacheKey(rawJSON)
}

func buildCodexSessionCacheKey(auth *cliproxyauth.Auth, rawJSON []byte, headers http.Header, fallbackAPIKey string) string {
	if auth != nil {
		if authID := strings.TrimSpace(auth.ID); authID != "" {
			return "auth:" + authID
		}
	}

	if strings.TrimSpace(fallbackAPIKey) != "" {
		fingerprint := uuid.NewSHA1(
			uuid.NameSpaceOID,
			[]byte("cli-proxy-api:codex:session-cache:"+strings.TrimSpace(fallbackAPIKey)),
		).String()
		return "api:" + fingerprint
	}

	userID := strings.TrimSpace(gjson.GetBytes(rawJSON, "metadata.user_id").String())
	if userID == "" && headers != nil {
		userID = strings.TrimSpace(headers.Get("Chatgpt-Account-Id"))
	}
	if userID != "" {
		return "user:" + userID
	}

	return "default"
}

func codexSessionFallbackAPIKey(ctx context.Context, auth *cliproxyauth.Auth) string {
	apiKey := strings.TrimSpace(apiKeyFromContext(ctx))
	if apiKey != "" {
		return apiKey
	}
	if auth != nil && auth.Attributes != nil {
		return strings.TrimSpace(auth.Attributes["api_key"])
	}
	return ""
}

func resolveManagedCodexContinuity(
	ctx context.Context,
	cacheKey string,
	missReason string,
) codexContinuity {
	continuity := codexContinuity{CacheKey: cacheKey}
	if cacheKey != "" {
		if cache, ok := getCodexCache(cacheKey); ok {
			continuity.Key = strings.TrimSpace(cache.ID)
			continuity.Source = "session_cache"
		}
	}
	if continuity.Key == "" {
		continuity.Key = generateCodexSessionID(ctx, missReason)
		continuity.Source = "generated_uuid_v7"
	}
	if continuity.Key == "" {
		continuity.Key = generateCodexSessionID(ctx, "resolved empty session id after cache lookup")
		continuity.Source = "generated_uuid_v7"
	}
	if continuity.shouldPersistManagedCache() {
		setCodexCache(cacheKey, codexCache{
			ID:     continuity.Key,
			Expire: time.Now().Add(codexSessionTTL),
		})
	}
	return continuity
}

func generateCodexSessionID(ctx context.Context, reason string) string {
	sessionUUID, err := uuid.NewV7()
	if err != nil {
		sessionID := uuid.New().String()
		logWithRequestID(ctx).Warnf(
			"codex executor: generated fallback uuid v4 session id because uuid v7 failed (reason=%s, err=%v)",
			reason,
			err,
		)
		return sessionID
	}
	sessionID := sessionUUID.String()
	logWithRequestID(ctx).Warnf("codex executor: generated new uuid v7 session id=%s (reason=%s)", sessionID, reason)
	return sessionID
}

func (e *CodexExecutor) resolveCodexSessionID(
	ctx context.Context,
	auth *cliproxyauth.Auth,
	from sdktranslator.Format,
	req cliproxyexecutor.Request,
) (sessionID, cacheKey string) {
	var incomingHeaders http.Header
	if ginCtx := ginContextFrom(ctx); ginCtx != nil && ginCtx.Request != nil {
		incomingHeaders = ginCtx.Request.Header
	}

	incomingSessionID := strings.TrimSpace(incomingHeaders.Get("Session_id"))
	incomingConversationID := strings.TrimSpace(incomingHeaders.Get("Conversation_id"))
	incomingPromptCacheKey := codexPromptCacheKey(req.Payload)

	cacheKey = buildCodexSessionCacheKey(auth, req.Payload, incomingHeaders, codexSessionFallbackAPIKey(ctx, auth))
	switch {
	case incomingSessionID != "":
		sessionID = incomingSessionID
	case incomingPromptCacheKey != "":
		sessionID = incomingPromptCacheKey
	case incomingConversationID != "":
		sessionID = incomingConversationID
	default:
		continuity := resolveManagedCodexContinuity(
			ctx,
			cacheKey,
			fmt.Sprintf("session_id and prompt_cache_key missing (from=%s)", strings.TrimSpace(from.String())),
		)
		sessionID = continuity.Key
		cacheKey = continuity.CacheKey
	}
	return
}

func (e *CodexExecutor) ensureCodexSessionTripleOnRawRequest(req *http.Request, auth *cliproxyauth.Auth) {
	if req == nil || req.URL == nil || req.Body == nil {
		return
	}
	if req.Method != "" && !strings.EqualFold(req.Method, http.MethodPost) {
		return
	}

	path := strings.TrimSpace(req.URL.Path)
	if path != "" &&
		!strings.HasSuffix(path, "/responses") &&
		!strings.HasSuffix(path, "/responses/compact") {
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		return
	}
	if len(bytes.TrimSpace(body)) == 0 {
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		return
	}

	continuity := codexContinuity{
		CacheKey: buildCodexSessionCacheKey(auth, body, req.Header, codexSessionFallbackAPIKey(req.Context(), auth)),
	}
	switch {
	case strings.TrimSpace(req.Header.Get("Session_id")) != "":
		continuity.Key = strings.TrimSpace(req.Header.Get("Session_id"))
		continuity.Source = "session_id_header"
	case codexPromptCacheKey(body) != "":
		continuity.Key = codexPromptCacheKey(body)
		continuity.Source = "prompt_cache_key"
	case strings.TrimSpace(req.Header.Get("Conversation_id")) != "":
		continuity.Key = strings.TrimSpace(req.Header.Get("Conversation_id"))
		continuity.Source = "conversation_id"
	default:
		continuity = resolveManagedCodexContinuity(
			req.Context(),
			continuity.CacheKey,
			"session_id and prompt_cache_key missing on raw request",
		)
	}
	if continuity.Key != "" {
		if updated, errSet := sjson.SetBytes(body, "prompt_cache_key", continuity.Key); errSet == nil {
			body = updated
		}
		body = setPromptCacheKeyInContexts(body, continuity.Key)
		req.Header.Set("Session_id", continuity.Key)
	}

	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
}

func setPromptCacheKeyInContexts(rawJSON []byte, promptCacheKey string) []byte {
	if strings.TrimSpace(promptCacheKey) == "" {
		return rawJSON
	}
	contexts := gjson.GetBytes(rawJSON, "contexts")
	if !contexts.Exists() {
		return rawJSON
	}
	if contexts.IsArray() {
		result := rawJSON
		for i := range contexts.Array() {
			path := fmt.Sprintf("contexts.%d.prompt_cache_key", i)
			if updated, err := sjson.SetBytes(result, path, promptCacheKey); err == nil {
				result = updated
			}
		}
		return result
	}
	if contexts.IsObject() {
		if updated, err := sjson.SetBytes(rawJSON, "contexts.prompt_cache_key", promptCacheKey); err == nil {
			return updated
		}
	}
	return rawJSON
}

func (e *CodexExecutor) ensureCodexToolsListOnRawRequest(req *http.Request) {
	if req == nil || req.URL == nil || req.Body == nil {
		return
	}
	if req.Method != "" && !strings.EqualFold(req.Method, http.MethodPost) {
		return
	}

	path := strings.TrimSpace(req.URL.Path)
	if path != "" &&
		!strings.HasSuffix(path, "/responses") &&
		!strings.HasSuffix(path, "/responses/compact") {
		return
	}

	body, err := io.ReadAll(req.Body)
	if err != nil {
		return
	}
	body = normalizeCodexToolsList(body)
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
}

func normalizeCodexToolsList(rawJSON []byte) []byte {
	result := rawJSON
	result, _ = sjson.SetBytes(result, "parallel_tool_calls", true)
	result, _ = sjson.SetBytes(result, "tool_choice", "auto")

	tools := gjson.GetBytes(result, "tools")
	if !tools.Exists() {
		return setDefaultCodexTools(result)
	}
	if tools.IsArray() && len(tools.Array()) == 0 {
		return result
	}

	normalizedTools := normalizeCodexToolsArray(tools)
	if len(normalizedTools) == 0 {
		return setDefaultCodexTools(result)
	}
	if updated, err := sjson.SetRawBytes(result, "tools", normalizedTools); err == nil {
		return updated
	}
	return result
}

func normalizeCodexToolsArray(tools gjson.Result) []byte {
	toolResults := []gjson.Result{tools}
	if tools.IsArray() {
		toolResults = tools.Array()
	}

	normalized := make([]string, 0, len(toolResults)+1)
	hasWebSearch := false
	for i := range toolResults {
		item, isWebSearch := normalizeCodexTool(toolResults[i])
		if strings.TrimSpace(item) == "" {
			continue
		}
		normalized = append(normalized, item)
		if isWebSearch {
			hasWebSearch = true
		}
	}

	if !hasWebSearch {
		normalized = append(normalized, defaultCodexWebSearchToolJSON)
	}
	if len(normalized) == 0 {
		return nil
	}
	return []byte("[" + strings.Join(normalized, ",") + "]")
}

func normalizeCodexTool(tool gjson.Result) (string, bool) {
	if !tool.Exists() {
		return "", false
	}
	if tool.Type == gjson.String {
		name := strings.TrimSpace(tool.String())
		if name == "" {
			return "", false
		}
		if isWebSearchToolName(name) {
			return defaultCodexWebSearchToolJSON, true
		}
		return buildCodexFunctionTool(name, "", "", false, false), false
	}
	if !tool.IsObject() {
		return "", false
	}

	toolType := strings.TrimSpace(tool.Get("type").String())
	if isWebSearchToolName(toolType) {
		return defaultCodexWebSearchToolJSON, true
	}
	if functionTool := normalizeCodexFunctionTool(tool); functionTool != "" {
		return functionTool, false
	}
	if toolType != "" {
		return buildCodexFunctionTool(toolType, tool.Get("description").String(), "", false, false), false
	}
	return "", false
}

func normalizeCodexFunctionTool(tool gjson.Result) string {
	toolType := strings.TrimSpace(tool.Get("type").String())
	name := strings.TrimSpace(tool.Get("name").String())
	description := tool.Get("description").String()
	parameters := tool.Get("parameters")
	inputSchema := tool.Get("input_schema")
	strictValue, strictExists := extractCodexToolStrict(tool)

	if function := tool.Get("function"); function.IsObject() {
		if name == "" {
			name = strings.TrimSpace(function.Get("name").String())
		}
		if strings.TrimSpace(description) == "" {
			description = function.Get("description").String()
		}
		if !parameters.Exists() {
			parameters = function.Get("parameters")
		}
		if !inputSchema.Exists() {
			inputSchema = function.Get("input_schema")
		}
	}

	if name == "" {
		if toolType == "function" {
			return ""
		}
		if !inputSchema.Exists() && !parameters.Exists() {
			return ""
		}
	}

	return buildCodexFunctionTool(
		name,
		description,
		pickCodexToolSchema(parameters, inputSchema),
		strictValue,
		strictExists,
	)
}

func buildCodexFunctionTool(name, description, schemaRaw string, strict bool, strictExists bool) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}

	tool := `{"type":"function","name":"","parameters":{"type":"object","properties":{}}}`
	tool, _ = sjson.Set(tool, "name", name)
	if strings.TrimSpace(description) != "" {
		tool, _ = sjson.Set(tool, "description", description)
	}
	tool, _ = sjson.SetRaw(tool, "parameters", normalizeCodexToolSchema(schemaRaw))
	if strictExists {
		tool, _ = sjson.Set(tool, "strict", strict)
	}
	return tool
}

func extractCodexToolStrict(tool gjson.Result) (bool, bool) {
	if strict := tool.Get("strict"); strict.Exists() {
		return strict.Bool(), true
	}
	if function := tool.Get("function"); function.IsObject() {
		if strict := function.Get("strict"); strict.Exists() {
			return strict.Bool(), true
		}
	}
	return false, false
}

func pickCodexToolSchema(parameters, inputSchema gjson.Result) string {
	if parameters.Exists() && strings.TrimSpace(parameters.Raw) != "" && strings.TrimSpace(parameters.Raw) != "null" {
		return parameters.Raw
	}
	if inputSchema.Exists() && strings.TrimSpace(inputSchema.Raw) != "" && strings.TrimSpace(inputSchema.Raw) != "null" {
		return inputSchema.Raw
	}
	return ""
}

func normalizeCodexToolSchema(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "null" || !gjson.Valid(raw) {
		return `{"type":"object","properties":{}}`
	}

	schema := raw
	parsed := gjson.Parse(raw)
	schemaType := strings.TrimSpace(parsed.Get("type").String())
	if schemaType == "" {
		schema, _ = sjson.Set(schema, "type", "object")
		schemaType = "object"
	}
	if schemaType == "object" && !parsed.Get("properties").Exists() {
		schema, _ = sjson.SetRaw(schema, "properties", `{}`)
	}
	schema, _ = sjson.Delete(schema, "$schema")
	return schema
}

func isWebSearchToolName(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "web_search", "web_search_preview":
		return true
	default:
		return false
	}
}

func setDefaultCodexTools(rawJSON []byte) []byte {
	if updated, err := sjson.SetRawBytes(rawJSON, "tools", []byte(defaultCodexToolsJSON)); err == nil {
		return updated
	}
	return rawJSON
}
