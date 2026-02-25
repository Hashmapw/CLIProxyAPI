package executor

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	codexauth "github.com/router-for-me/CLIProxyAPI/v6/internal/auth/codex"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v6/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/tiktoken-go/tokenizer"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	codexClientVersion = "0.101.0"
	codexUserAgent     = "codex_cli_rs/0.101.0 (Mac OS 26.0.1; arm64) Apple_Terminal/464"
	codexSessionTTL    = 3 * time.Hour
)

var dataTag = []byte("data:")

// CodexExecutor is a stateless executor for Codex (OpenAI Responses API entrypoint).
// If api_key is unavailable on auth, it falls back to legacy via ClientAdapter.
type CodexExecutor struct {
	cfg *config.Config
}

func NewCodexExecutor(cfg *config.Config) *CodexExecutor { return &CodexExecutor{cfg: cfg} }

func (e *CodexExecutor) Identifier() string { return "codex" }

// PrepareRequest injects Codex credentials into the outgoing HTTP request.
func (e *CodexExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	apiKey, _ := codexCreds(auth)
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Codex credentials into the request and executes it.
func (e *CodexExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("codex executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	e.ensureCodexToolsListOnRawRequest(httpReq)
	e.ensureCodexSessionTripleOnRawRequest(httpReq)
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

func (e *CodexExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	if opts.Alt == "responses/compact" {
		return e.executeCompact(ctx, auth, req, opts)
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("codex")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, false)
	body := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.SetBytes(body, "stream", true)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	if !gjson.GetBytes(body, "instructions").Exists() {
		body, _ = sjson.SetBytes(body, "instructions", "")
	}
	body = normalizeCodexToolsList(body)

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	sessionID, cacheKey := e.resolveCodexSessionID(ctx, from, req)
	body, _ = sjson.SetBytes(body, "prompt_cache_key", sessionID)
	body = setPromptCacheKeyInContexts(body, sessionID)
	log.Debugf("codex executor: session triple set to: %s (cacheKey=%s, from=%s)", sessionID, cacheKey, from)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Session_id", sessionID)
	applyCodexHeaders(httpReq, auth, apiKey, true)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		deleteCodexCache(cacheKey)
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	data, err := io.ReadAll(httpResp.Body)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	appendAPIResponseChunk(ctx, e.cfg, data)

	lines := bytes.Split(data, []byte("\n"))
	for _, line := range lines {
		if !bytes.HasPrefix(line, dataTag) {
			continue
		}

		line = bytes.TrimSpace(line[5:])
		lineType := gjson.GetBytes(line, "type").String()

		if lineType == "error" {
			errorCode := gjson.GetBytes(line, "code").String()
			errorMsg := gjson.GetBytes(line, "message").String()
			logWithRequestID(ctx).Debugf("codex executor: received error in response data - code: %s, message: %s", errorCode, errorMsg)
			deleteCodexCache(cacheKey)
			err = statusErr{code: 400, msg: fmt.Sprintf("error type: %s, code: %s, message: %s", lineType, errorCode, errorMsg)}
			return resp, err
		}

		if lineType != "response.completed" {
			continue
		}

		if detail, ok := parseCodexUsage(line); ok {
			reporter.publish(ctx, detail)
		}

		var param any
		out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, originalPayload, body, line, &param)
		resp = cliproxyexecutor.Response{Payload: []byte(out), Headers: httpResp.Header.Clone()}
		return resp, nil
	}
	deleteCodexCache(cacheKey)
	err = statusErr{code: 408, msg: "stream error: stream disconnected before completion: stream closed before response.completed"}
	return resp, err
}

func (e *CodexExecutor) executeCompact(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai-response")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, false)
	body := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.DeleteBytes(body, "stream")
	body = normalizeCodexToolsList(body)

	url := strings.TrimSuffix(baseURL, "/") + "/responses/compact"
	sessionID, cacheKey := e.resolveCodexSessionID(ctx, from, req)
	body, _ = sjson.SetBytes(body, "prompt_cache_key", sessionID)
	body = setPromptCacheKeyInContexts(body, sessionID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Session_id", sessionID)
	applyCodexHeaders(httpReq, auth, apiKey, false)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		deleteCodexCache(cacheKey)
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	data, err := io.ReadAll(httpResp.Body)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	appendAPIResponseChunk(ctx, e.cfg, data)
	reporter.publish(ctx, parseOpenAIUsage(data))
	reporter.ensurePublished(ctx)
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, originalPayload, body, data, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out), Headers: httpResp.Header.Clone()}
	return resp, nil
}

func (e *CodexExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	if opts.Alt == "responses/compact" {
		return nil, statusErr{code: http.StatusBadRequest, msg: "streaming not supported for /responses/compact"}
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("codex")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	body := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, true)

	body, err = thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	requestedModel := payloadRequestedModel(opts, req.Model)
	body = applyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	body, _ = sjson.SetBytes(body, "model", baseModel)
	if !gjson.GetBytes(body, "instructions").Exists() {
		body, _ = sjson.SetBytes(body, "instructions", "")
	}
	body = normalizeCodexToolsList(body)

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	sessionID, cacheKey := e.resolveCodexSessionID(ctx, from, req)
	body, _ = sjson.SetBytes(body, "prompt_cache_key", sessionID)
	body = setPromptCacheKeyInContexts(body, sessionID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Session_id", sessionID)
	applyCodexHeaders(httpReq, auth, apiKey, true)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, readErr := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
		if readErr != nil {
			recordAPIResponseError(ctx, e.cfg, readErr)
			return nil, readErr
		}
		appendAPIResponseChunk(ctx, e.cfg, data)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		deleteCodexCache(cacheKey)
		err = statusErr{code: httpResp.StatusCode, msg: string(data)}
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("codex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB
		var param any
		for scanner.Scan() {
			line := scanner.Bytes()
			appendAPIResponseChunk(ctx, e.cfg, line)

			if bytes.HasPrefix(line, dataTag) {
				data := bytes.TrimSpace(line[5:])
				dataType := gjson.GetBytes(data, "type").String()

				if dataType == "error" {
					errorCode := gjson.GetBytes(data, "code").String()
					errorMsg := gjson.GetBytes(data, "message").String()
					logWithRequestID(ctx).Debugf("codex executor: received error in stream data - code: %s, message: %s", errorCode, errorMsg)
					deleteCodexCache(cacheKey)
					reporter.publishFailure(ctx)
					out <- cliproxyexecutor.StreamChunk{Err: fmt.Errorf("error type: %s, code: %s, message: %s", dataType, errorCode, errorMsg)}
					return
				}

				if dataType == "response.completed" {
					if detail, ok := parseCodexUsage(data); ok {
						reporter.publish(ctx, detail)
					}
				}
			}

			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, originalPayload, body, bytes.Clone(line), &param)
			for i := range chunks {
				out <- cliproxyexecutor.StreamChunk{Payload: []byte(chunks[i])}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			recordAPIResponseError(ctx, e.cfg, errScan)
			deleteCodexCache(cacheKey)
			reporter.publishFailure(ctx)
			out <- cliproxyexecutor.StreamChunk{Err: errScan}
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

func (e *CodexExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("codex")
	body := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	body, err := thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	body, _ = sjson.SetBytes(body, "stream", false)
	if !gjson.GetBytes(body, "instructions").Exists() {
		body, _ = sjson.SetBytes(body, "instructions", "")
	}

	enc, err := tokenizerForCodexModel(baseModel)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("codex executor: tokenizer init failed: %w", err)
	}

	count, err := countCodexInputTokens(enc, body)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("codex executor: token counting failed: %w", err)
	}

	usageJSON := fmt.Sprintf(`{"response":{"usage":{"input_tokens":%d,"output_tokens":0,"total_tokens":%d}}}`, count, count)
	translated := sdktranslator.TranslateTokenCount(ctx, to, from, count, []byte(usageJSON))
	return cliproxyexecutor.Response{Payload: []byte(translated)}, nil
}

func tokenizerForCodexModel(model string) (tokenizer.Codec, error) {
	sanitized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case sanitized == "":
		return tokenizer.Get(tokenizer.Cl100kBase)
	case strings.HasPrefix(sanitized, "gpt-5"):
		return tokenizer.ForModel(tokenizer.GPT5)
	case strings.HasPrefix(sanitized, "gpt-4.1"):
		return tokenizer.ForModel(tokenizer.GPT41)
	case strings.HasPrefix(sanitized, "gpt-4o"):
		return tokenizer.ForModel(tokenizer.GPT4o)
	case strings.HasPrefix(sanitized, "gpt-4"):
		return tokenizer.ForModel(tokenizer.GPT4)
	case strings.HasPrefix(sanitized, "gpt-3.5"), strings.HasPrefix(sanitized, "gpt-3"):
		return tokenizer.ForModel(tokenizer.GPT35Turbo)
	default:
		return tokenizer.Get(tokenizer.Cl100kBase)
	}
}

func countCodexInputTokens(enc tokenizer.Codec, body []byte) (int64, error) {
	if enc == nil {
		return 0, fmt.Errorf("encoder is nil")
	}
	if len(body) == 0 {
		return 0, nil
	}

	root := gjson.ParseBytes(body)
	var segments []string

	if inst := strings.TrimSpace(root.Get("instructions").String()); inst != "" {
		segments = append(segments, inst)
	}

	inputItems := root.Get("input")
	if inputItems.IsArray() {
		arr := inputItems.Array()
		for i := range arr {
			item := arr[i]
			switch item.Get("type").String() {
			case "message":
				content := item.Get("content")
				if content.IsArray() {
					parts := content.Array()
					for j := range parts {
						part := parts[j]
						if text := strings.TrimSpace(part.Get("text").String()); text != "" {
							segments = append(segments, text)
						}
					}
				}
			case "function_call":
				if name := strings.TrimSpace(item.Get("name").String()); name != "" {
					segments = append(segments, name)
				}
				if args := strings.TrimSpace(item.Get("arguments").String()); args != "" {
					segments = append(segments, args)
				}
			case "function_call_output":
				if out := strings.TrimSpace(item.Get("output").String()); out != "" {
					segments = append(segments, out)
				}
			default:
				if text := strings.TrimSpace(item.Get("text").String()); text != "" {
					segments = append(segments, text)
				}
			}
		}
	}

	tools := root.Get("tools")
	if tools.IsArray() {
		tarr := tools.Array()
		for i := range tarr {
			tool := tarr[i]
			if name := strings.TrimSpace(tool.Get("name").String()); name != "" {
				segments = append(segments, name)
			}
			if desc := strings.TrimSpace(tool.Get("description").String()); desc != "" {
				segments = append(segments, desc)
			}
			if params := tool.Get("parameters"); params.Exists() {
				val := params.Raw
				if params.Type == gjson.String {
					val = params.String()
				}
				if trimmed := strings.TrimSpace(val); trimmed != "" {
					segments = append(segments, trimmed)
				}
			}
		}
	}

	textFormat := root.Get("text.format")
	if textFormat.Exists() {
		if name := strings.TrimSpace(textFormat.Get("name").String()); name != "" {
			segments = append(segments, name)
		}
		if schema := textFormat.Get("schema"); schema.Exists() {
			val := schema.Raw
			if schema.Type == gjson.String {
				val = schema.String()
			}
			if trimmed := strings.TrimSpace(val); trimmed != "" {
				segments = append(segments, trimmed)
			}
		}
	}

	text := strings.Join(segments, "\n")
	if text == "" {
		return 0, nil
	}

	count, err := enc.Count(text)
	if err != nil {
		return 0, err
	}
	return int64(count), nil
}

func (e *CodexExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("codex executor: refresh called")
	if auth == nil {
		return nil, statusErr{code: 500, msg: "codex executor: auth is nil"}
	}
	var refreshToken string
	if auth.Metadata != nil {
		if v, ok := auth.Metadata["refresh_token"].(string); ok && v != "" {
			refreshToken = v
		}
	}
	if refreshToken == "" {
		return auth, nil
	}
	svc := codexauth.NewCodexAuth(e.cfg)
	td, err := svc.RefreshTokensWithRetry(ctx, refreshToken, 3)
	if err != nil {
		return nil, err
	}
	if auth.Metadata == nil {
		auth.Metadata = make(map[string]any)
	}
	auth.Metadata["id_token"] = td.IDToken
	auth.Metadata["access_token"] = td.AccessToken
	if td.RefreshToken != "" {
		auth.Metadata["refresh_token"] = td.RefreshToken
	}
	if td.AccountID != "" {
		auth.Metadata["account_id"] = td.AccountID
	}
	auth.Metadata["email"] = td.Email
	// Use unified key in files
	auth.Metadata["expired"] = td.Expire
	auth.Metadata["type"] = "codex"
	now := time.Now().Format(time.RFC3339)
	auth.Metadata["last_refresh"] = now
	return auth, nil
}

func codexPromptCacheKey(rawJSON []byte) string {
	promptCacheKey := strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt_cache_key").String())
	if promptCacheKey != "" {
		return promptCacheKey
	}
	// Backward-compatible alias used by some callers.
	return strings.TrimSpace(gjson.GetBytes(rawJSON, "prompt_key_cache").String())
}

func buildCodexSessionCacheKey(model string, rawJSON []byte, headers http.Header) string {
	resolvedModel := strings.TrimSpace(model)
	if resolvedModel == "" {
		resolvedModel = strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	}
	if resolvedModel == "" {
		resolvedModel = "codex"
	}
	userID := strings.TrimSpace(gjson.GetBytes(rawJSON, "metadata.user_id").String())
	if userID == "" && headers != nil {
		userID = strings.TrimSpace(headers.Get("Chatgpt-Account-Id"))
	}
	if userID == "" {
		return resolvedModel + "-default"
	}
	return fmt.Sprintf("%s-%s", resolvedModel, userID)
}

func resolveCodexSessionWithCache(ctx context.Context, incomingSessionID, incomingPromptCacheKey, incomingConversationID, cacheKey, missReason string) string {
	sessionID := strings.TrimSpace(incomingSessionID)
	if sessionID == "" {
		sessionID = strings.TrimSpace(incomingPromptCacheKey)
	}
	if sessionID == "" {
		sessionID = strings.TrimSpace(incomingConversationID)
	}
	if sessionID == "" {
		if cache, ok := getCodexCache(cacheKey); ok {
			sessionID = strings.TrimSpace(cache.ID)
		}
	}
	if sessionID == "" {
		sessionID = generateCodexSessionID(ctx, missReason)
	}
	if sessionID == "" {
		sessionID = generateCodexSessionID(ctx, "resolved empty session id after cache lookup")
	}
	setCodexCache(cacheKey, codexCache{
		ID:     sessionID,
		Expire: time.Now().Add(codexSessionTTL),
	})
	return sessionID
}

// resolveCodexSessionID determines the session ID to use for prompt/session sync.
// It returns the session ID and the cache key (for invalidation on error).
func (e *CodexExecutor) resolveCodexSessionID(ctx context.Context, from sdktranslator.Format, req cliproxyexecutor.Request) (sessionID, cacheKey string) {
	incomingPromptCacheKey := codexPromptCacheKey(req.Payload)
	var incomingSessionID, incomingConversationID string
	var incomingHeaders http.Header
	if ginCtx, ok := ctx.Value("gin").(*gin.Context); ok && ginCtx != nil && ginCtx.Request != nil {
		incomingHeaders = ginCtx.Request.Header
		incomingSessionID = strings.TrimSpace(incomingHeaders.Get("Session_id"))
		incomingConversationID = strings.TrimSpace(incomingHeaders.Get("Conversation_id"))
	}

	cacheKey = buildCodexSessionCacheKey(req.Model, req.Payload, incomingHeaders)
	sessionID = resolveCodexSessionWithCache(
		ctx,
		incomingSessionID,
		incomingPromptCacheKey,
		incomingConversationID,
		cacheKey,
		fmt.Sprintf("session_id and prompt_cache_key missing (from=%s)", strings.TrimSpace(from.String())),
	)
	return
}

func (e *CodexExecutor) ensureCodexSessionTripleOnRawRequest(req *http.Request) {
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

	promptCacheKey := codexPromptCacheKey(body)
	incomingSessionID := strings.TrimSpace(req.Header.Get("Session_id"))
	incomingConversationID := strings.TrimSpace(req.Header.Get("Conversation_id"))
	model := strings.TrimSpace(gjson.GetBytes(body, "model").String())
	cacheKey := buildCodexSessionCacheKey(model, body, req.Header)
	sessionID := resolveCodexSessionWithCache(
		req.Context(),
		incomingSessionID,
		promptCacheKey,
		incomingConversationID,
		cacheKey,
		"session_id and prompt_cache_key missing on raw request",
	)

	if updated, errSet := sjson.SetBytes(body, "prompt_cache_key", sessionID); errSet == nil {
		body = updated
	}
	body = setPromptCacheKeyInContexts(body, sessionID)
	req.Body = io.NopCloser(bytes.NewReader(body))
	req.ContentLength = int64(len(body))
	req.Header.Set("Session_id", sessionID)
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
		arr := contexts.Array()
		for i := range arr {
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

func generateCodexSessionID(ctx context.Context, reason string) string {
	sessionUUID, err := uuid.NewV7()
	if err != nil {
		sessionID := uuid.New().String()
		logWithRequestID(ctx).Warnf("codex executor: generated fallback uuid v4 session id because uuid v7 failed (reason=%s, err=%v)", reason, err)
		return sessionID
	}
	sessionID := sessionUUID.String()
	logWithRequestID(ctx).Warnf("codex executor: generated new uuid v7 session id=%s (reason=%s)", sessionID, reason)
	return sessionID
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

	// Unsupported built-ins are converted into generic function tools so Codex can accept them.
	if toolType != "" {
		return buildCodexFunctionTool(toolType, tool.Get("description").String(), "", false, false), false
	}
	return "", false
}

func normalizeCodexFunctionTool(tool gjson.Result) string {
	toolType := strings.TrimSpace(tool.Get("type").String())
	name := strings.TrimSpace(tool.Get("name").String())
	desc := tool.Get("description").String()
	parameters := tool.Get("parameters")
	inputSchema := tool.Get("input_schema")
	strictValue, strictExists := extractCodexToolStrict(tool)

	if function := tool.Get("function"); function.IsObject() {
		if name == "" {
			name = strings.TrimSpace(function.Get("name").String())
		}
		if strings.TrimSpace(desc) == "" {
			desc = function.Get("description").String()
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
		// Tools in {"name","input_schema"} style are always converted to function.
		if !inputSchema.Exists() && !parameters.Exists() {
			return ""
		}
	}

	return buildCodexFunctionTool(name, desc, pickCodexToolSchema(parameters, inputSchema), strictValue, strictExists)
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

const (
	defaultCodexWebSearchToolJSON = `{"type":"web_search","external_web_access":true}`
	defaultCodexToolsJSON         = "[" + defaultCodexWebSearchToolJSON + "]"
)

func applyCodexHeaders(r *http.Request, auth *cliproxyauth.Auth, token string, stream bool) {
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+token)

	var ginHeaders http.Header
	if ginCtx, ok := r.Context().Value("gin").(*gin.Context); ok && ginCtx != nil && ginCtx.Request != nil {
		ginHeaders = ginCtx.Request.Header
	}

	misc.EnsureHeader(r.Header, ginHeaders, "Version", codexClientVersion)
	misc.EnsureHeader(r.Header, ginHeaders, "Openai-Beta", "responses=experimental")
	misc.EnsureHeader(r.Header, ginHeaders, "User-Agent", codexUserAgent)

	if stream {
		r.Header.Set("Accept", "text/event-stream")
	} else {
		r.Header.Set("Accept", "application/json")
	}
	r.Header.Set("Connection", "Keep-Alive")

	isAPIKey := false
	if auth != nil && auth.Attributes != nil {
		if v := strings.TrimSpace(auth.Attributes["api_key"]); v != "" {
			isAPIKey = true
		}
	}
	if !isAPIKey {
		r.Header.Set("Originator", "codex_cli_rs")
		if auth != nil && auth.Metadata != nil {
			if accountID, ok := auth.Metadata["account_id"].(string); ok {
				r.Header.Set("Chatgpt-Account-Id", accountID)
			}
		}
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(r, attrs)
}

func codexCreds(a *cliproxyauth.Auth) (apiKey, baseURL string) {
	if a == nil {
		return "", ""
	}
	if a.Attributes != nil {
		apiKey = a.Attributes["api_key"]
		baseURL = a.Attributes["base_url"]
	}
	if apiKey == "" && a.Metadata != nil {
		if v, ok := a.Metadata["access_token"].(string); ok {
			apiKey = v
		}
	}
	return
}

func (e *CodexExecutor) resolveCodexConfig(auth *cliproxyauth.Auth) *config.CodexKey {
	if auth == nil || e.cfg == nil {
		return nil
	}
	var attrKey, attrBase string
	if auth.Attributes != nil {
		attrKey = strings.TrimSpace(auth.Attributes["api_key"])
		attrBase = strings.TrimSpace(auth.Attributes["base_url"])
	}
	for i := range e.cfg.CodexKey {
		entry := &e.cfg.CodexKey[i]
		cfgKey := strings.TrimSpace(entry.APIKey)
		cfgBase := strings.TrimSpace(entry.BaseURL)
		if attrKey != "" && attrBase != "" {
			if strings.EqualFold(cfgKey, attrKey) && strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
			continue
		}
		if attrKey != "" && strings.EqualFold(cfgKey, attrKey) {
			if cfgBase == "" || strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
		}
		if attrKey == "" && attrBase != "" && strings.EqualFold(cfgBase, attrBase) {
			return entry
		}
	}
	if attrKey != "" {
		for i := range e.cfg.CodexKey {
			entry := &e.cfg.CodexKey[i]
			if strings.EqualFold(strings.TrimSpace(entry.APIKey), attrKey) {
				return entry
			}
		}
	}
	return nil
}
