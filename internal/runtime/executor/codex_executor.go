package executor

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	codexauth "github.com/router-for-me/CLIProxyAPI/v7/internal/auth/codex"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/tiktoken-go/tokenizer"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	codexUserAgent             = "codex_cli_rs/0.118.0 (Mac OS 26.3.1; arm64) iTerm.app/3.6.9"
	codexOriginator            = "codex_cli_rs"
	codexDefaultImageToolModel = "gpt-image-2"
)

var dataTag = []byte("data:")

// Streamed Codex responses may emit response.output_item.done events while leaving
// response.completed.response.output empty. Keep the stream path aligned with the
// already-patched non-stream path by reconstructing response.output from those items.
func collectCodexOutputItemDone(eventData []byte, outputItemsByIndex map[int64][]byte, outputItemsFallback *[][]byte) {
	itemResult := gjson.GetBytes(eventData, "item")
	if !itemResult.Exists() || itemResult.Type != gjson.JSON {
		return
	}
	outputIndexResult := gjson.GetBytes(eventData, "output_index")
	if outputIndexResult.Exists() {
		outputItemsByIndex[outputIndexResult.Int()] = []byte(itemResult.Raw)
		return
	}
	*outputItemsFallback = append(*outputItemsFallback, []byte(itemResult.Raw))
}

func patchCodexCompletedOutput(eventData []byte, outputItemsByIndex map[int64][]byte, outputItemsFallback [][]byte) []byte {
	outputResult := gjson.GetBytes(eventData, "response.output")
	shouldPatchOutput := (!outputResult.Exists() || !outputResult.IsArray() || len(outputResult.Array()) == 0) && (len(outputItemsByIndex) > 0 || len(outputItemsFallback) > 0)
	if !shouldPatchOutput {
		return eventData
	}

	indexes := make([]int64, 0, len(outputItemsByIndex))
	for idx := range outputItemsByIndex {
		indexes = append(indexes, idx)
	}
	sort.Slice(indexes, func(i, j int) bool {
		return indexes[i] < indexes[j]
	})

	items := make([][]byte, 0, len(outputItemsByIndex)+len(outputItemsFallback))
	for _, idx := range indexes {
		items = append(items, outputItemsByIndex[idx])
	}
	items = append(items, outputItemsFallback...)

	outputArray := []byte("[]")
	if len(items) > 0 {
		var buf bytes.Buffer
		totalLen := 2
		for _, item := range items {
			totalLen += len(item)
		}
		if len(items) > 1 {
			totalLen += len(items) - 1
		}
		buf.Grow(totalLen)
		buf.WriteByte('[')
		for i, item := range items {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.Write(item)
		}
		buf.WriteByte(']')
		outputArray = buf.Bytes()
	}

	completedDataPatched, _ := sjson.SetRawBytes(eventData, "response.output", outputArray)
	return completedDataPatched
}

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
	e.ensureCodexToolsListOnRawRequest(httpReq)
	e.ensureCodexSessionTripleOnRawRequest(httpReq, auth)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
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

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

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

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel, requestPath)
	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.SetBytes(body, "stream", true)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	body, _ = sjson.DeleteBytes(body, "stream_options")
	body = normalizeCodexInstructions(body)
	if e.cfg == nil || e.cfg.DisableImageGeneration == config.DisableImageGenerationOff {
		body = ensureImageGenerationTool(body, baseModel, auth)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	for attempt := 0; attempt < 2; attempt++ {
		httpReq, continuity, attemptBody, errReq := e.cacheHelper(ctx, auth, from, url, req, opts, body)
		if errReq != nil {
			return resp, errReq
		}
		applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
		logCodexRequestDiagnostics(ctx, auth, req, opts, httpReq.Header, attemptBody, continuity)
		helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
			URL:       url,
			Method:    http.MethodPost,
			Headers:   httpReq.Header.Clone(),
			Body:      attemptBody,
			Provider:  e.Identifier(),
			AuthID:    authID,
			AuthLabel: authLabel,
			AuthType:  authType,
			AuthValue: authValue,
		})

		httpResp, errReq := httpClient.Do(httpReq)
		if errReq != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errReq)
			if shouldRetryCodexContinuity(ctx, attempt, continuity, errReq) {
				continue
			}
			return resp, errReq
		}

		func() {
			defer func() {
				if errClose := httpResp.Body.Close(); errClose != nil {
					log.Errorf("codex executor: close response body error: %v", errClose)
				}
			}()

			helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
			if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
				b, _ := io.ReadAll(httpResp.Body)
				helps.AppendAPIResponseChunk(ctx, e.cfg, b)
				helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
				err = newCodexStatusErr(httpResp.StatusCode, b)
				return
			}
			data, errRead := io.ReadAll(httpResp.Body)
			if errRead != nil {
				helps.RecordAPIResponseError(ctx, e.cfg, errRead)
				err = errRead
				return
			}
			helps.AppendAPIResponseChunk(ctx, e.cfg, data)

			lines := bytes.Split(data, []byte("\n"))
			outputItemsByIndex := make(map[int64][]byte)
			var outputItemsFallback [][]byte
			for _, line := range lines {
				if !bytes.HasPrefix(line, dataTag) {
					continue
				}

				eventData := bytes.TrimSpace(line[5:])
				eventType := gjson.GetBytes(eventData, "type").String()

				if eventType == "error" {
					errorCode := gjson.GetBytes(eventData, "code").String()
					errorMsg := gjson.GetBytes(eventData, "message").String()
					err = statusErr{code: http.StatusBadRequest, msg: fmt.Sprintf("error type: %s, code: %s, message: %s", eventType, errorCode, errorMsg)}
					return
				}

				if eventType == "response.output_item.done" {
					collectCodexOutputItemDone(eventData, outputItemsByIndex, &outputItemsFallback)
					continue
				}

				if eventType != "response.completed" {
					continue
				}

				if detail, ok := helps.ParseCodexUsage(eventData); ok {
					reporter.Publish(ctx, detail)
				}
				publishCodexImageToolUsage(ctx, reporter, attemptBody, eventData)

				completedData := patchCodexCompletedOutput(eventData, outputItemsByIndex, outputItemsFallback)
				var param any
				out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, originalPayload, attemptBody, completedData, &param)
				resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
				err = nil
				return
			}
			err = statusErr{code: http.StatusRequestTimeout, msg: "stream error: stream disconnected before completion: stream closed before response.completed"}
		}()

		if err == nil {
			return resp, nil
		}
		if shouldRetryCodexContinuity(ctx, attempt, continuity, err) {
			continue
		}
		return resp, err
	}
	return resp, err
}

func (e *CodexExecutor) executeCompact(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

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

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel, requestPath)
	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.DeleteBytes(body, "stream")
	body = normalizeCodexInstructions(body)
	if e.cfg == nil || e.cfg.DisableImageGeneration == config.DisableImageGenerationOff {
		body = ensureImageGenerationTool(body, baseModel, auth)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses/compact"
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	for attempt := 0; attempt < 2; attempt++ {
		httpReq, continuity, attemptBody, errReq := e.cacheHelper(ctx, auth, from, url, req, opts, body)
		if errReq != nil {
			return resp, errReq
		}
		applyCodexHeaders(httpReq, auth, apiKey, false, e.cfg)
		logCodexRequestDiagnostics(ctx, auth, req, opts, httpReq.Header, attemptBody, continuity)
		helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
			URL:       url,
			Method:    http.MethodPost,
			Headers:   httpReq.Header.Clone(),
			Body:      attemptBody,
			Provider:  e.Identifier(),
			AuthID:    authID,
			AuthLabel: authLabel,
			AuthType:  authType,
			AuthValue: authValue,
		})
		httpResp, errReq := httpClient.Do(httpReq)
		if errReq != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errReq)
			if shouldRetryCodexContinuity(ctx, attempt, continuity, errReq) {
				continue
			}
			return resp, errReq
		}

		func() {
			defer func() {
				if errClose := httpResp.Body.Close(); errClose != nil {
					log.Errorf("codex executor: close response body error: %v", errClose)
				}
			}()
			helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
			if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
				b, _ := io.ReadAll(httpResp.Body)
				helps.AppendAPIResponseChunk(ctx, e.cfg, b)
				helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
				err = newCodexStatusErr(httpResp.StatusCode, b)
				return
			}
			data, errRead := io.ReadAll(httpResp.Body)
			if errRead != nil {
				helps.RecordAPIResponseError(ctx, e.cfg, errRead)
				err = errRead
				return
			}
			helps.AppendAPIResponseChunk(ctx, e.cfg, data)
			reporter.Publish(ctx, helps.ParseOpenAIUsage(data))
			reporter.EnsurePublished(ctx)
			var param any
			out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, originalPayload, attemptBody, data, &param)
			resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
			err = nil
		}()

		if err == nil {
			return resp, nil
		}
		if shouldRetryCodexContinuity(ctx, attempt, continuity, err) {
			continue
		}
		return resp, err
	}
	return resp, err
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

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

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

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	body = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", body, originalTranslated, requestedModel, requestPath)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	body, _ = sjson.DeleteBytes(body, "stream_options")
	body, _ = sjson.SetBytes(body, "model", baseModel)
	body = normalizeCodexInstructions(body)
	if e.cfg == nil || e.cfg.DisableImageGeneration == config.DisableImageGenerationOff {
		body = ensureImageGenerationTool(body, baseModel, auth)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	var httpResp *http.Response
	var continuity codexContinuity
	var attemptBody []byte
	for attempt := 0; attempt < 2; attempt++ {
		httpReq, continuityAttempt, currentBody, errReq := e.cacheHelper(ctx, auth, from, url, req, opts, body)
		if errReq != nil {
			return nil, errReq
		}
		continuity = continuityAttempt
		attemptBody = currentBody
		applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
		logCodexRequestDiagnostics(ctx, auth, req, opts, httpReq.Header, attemptBody, continuity)
		helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
			URL:       url,
			Method:    http.MethodPost,
			Headers:   httpReq.Header.Clone(),
			Body:      attemptBody,
			Provider:  e.Identifier(),
			AuthID:    authID,
			AuthLabel: authLabel,
			AuthType:  authType,
			AuthValue: authValue,
		})

		httpResp, err = httpClient.Do(httpReq)
		if err != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, err)
			if shouldRetryCodexContinuity(ctx, attempt, continuity, err) {
				continue
			}
			return nil, err
		}
		helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
		if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
			data, readErr := io.ReadAll(httpResp.Body)
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("codex executor: close response body error: %v", errClose)
			}
			if readErr != nil {
				helps.RecordAPIResponseError(ctx, e.cfg, readErr)
				if shouldRetryCodexContinuity(ctx, attempt, continuity, readErr) {
					continue
				}
				return nil, readErr
			}
			helps.AppendAPIResponseChunk(ctx, e.cfg, data)
			helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
			errReq = newCodexStatusErr(httpResp.StatusCode, data)
			if shouldRetryCodexContinuity(ctx, attempt, continuity, errReq) {
				continue
			}
			return nil, errReq
		}
		break
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
		outputItemsByIndex := make(map[int64][]byte)
		var outputItemsFallback [][]byte
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			translatedLine := bytes.Clone(line)

			if bytes.HasPrefix(line, dataTag) {
				data := bytes.TrimSpace(line[5:])
				switch gjson.GetBytes(data, "type").String() {
				case "error":
					errorCode := gjson.GetBytes(data, "code").String()
					errorMsg := gjson.GetBytes(data, "message").String()
					clearCodexContinuityCache(continuity)
					reporter.PublishFailure(ctx)
					select {
					case out <- cliproxyexecutor.StreamChunk{Err: fmt.Errorf("error type: %s, code: %s, message: %s", "error", errorCode, errorMsg)}:
					case <-ctx.Done():
					}
					return
				case "response.output_item.done":
					collectCodexOutputItemDone(data, outputItemsByIndex, &outputItemsFallback)
				case "response.completed":
					if detail, ok := helps.ParseCodexUsage(data); ok {
						reporter.Publish(ctx, detail)
					}
					publishCodexImageToolUsage(ctx, reporter, attemptBody, data)
					data = patchCodexCompletedOutput(data, outputItemsByIndex, outputItemsFallback)
					translatedLine = append([]byte("data: "), data...)
				}
			}

			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, originalPayload, attemptBody, translatedLine, &param)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					return
				}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			clearCodexContinuityCache(continuity)
			reporter.PublishFailure(ctx, errScan)
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errScan}:
			case <-ctx.Done():
			}
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
	body, _ = sjson.DeleteBytes(body, "stream_options")
	body, _ = sjson.SetBytes(body, "stream", false)
	body = normalizeCodexInstructions(body)

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
	return cliproxyexecutor.Response{Payload: translated}, nil
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
	if refreshed, handled, err := helps.RefreshAuthViaHome(ctx, e.cfg, auth); handled {
		return refreshed, err
	}
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
	svc := codexauth.NewCodexAuthWithProxyURL(e.cfg, auth.ProxyURL)
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

const (
	codexSessionTTL               = 3 * time.Hour
	defaultCodexWebSearchToolJSON = `{"type":"web_search","external_web_access":true}`
	defaultCodexToolsJSON         = "[" + defaultCodexWebSearchToolJSON + "]"
)

type codexContinuity struct {
	Key      string
	Source   string
	CacheKey string
}

func ginContextFrom(ctx context.Context) *gin.Context {
	if ctx == nil {
		return nil
	}
	ginCtx, _ := ctx.Value("gin").(*gin.Context)
	return ginCtx
}

func logWithRequestID(ctx context.Context) *log.Entry {
	return helps.LogWithRequestID(ctx)
}

func apiKeyFromContext(ctx context.Context) string {
	return helps.APIKeyFromContext(ctx)
}

func (c codexContinuity) shouldPersistManagedCache() bool {
	return c.CacheKey != "" && (c.Source == "session_cache" || c.Source == "generated_uuid_v7")
}

func (c codexContinuity) allowsFixedSessionRetry() bool {
	return c.shouldPersistManagedCache()
}

func clearCodexContinuityCache(continuity codexContinuity) {
	if continuity.shouldPersistManagedCache() {
		helps.DeleteCodexCache(continuity.CacheKey)
	}
}

func shouldRetryCodexContinuity(ctx context.Context, attempt int, continuity codexContinuity, cause error) bool {
	if !continuity.allowsFixedSessionRetry() {
		return false
	}
	clearCodexContinuityCache(continuity)
	if attempt > 0 {
		return false
	}
	logWithRequestID(ctx).Warnf(
		"codex executor: clearing fixed session cache and retrying once (source=%s, cache_key=%s, err=%v)",
		continuity.Source,
		continuity.CacheKey,
		cause,
	)
	return true
}

func metadataString(meta map[string]any, key string) string {
	if len(meta) == 0 {
		return ""
	}
	raw, ok := meta[key]
	if !ok || raw == nil {
		return ""
	}
	switch v := raw.(type) {
	case string:
		return strings.TrimSpace(v)
	case []byte:
		return strings.TrimSpace(string(v))
	default:
		return ""
	}
}

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

func resolveManagedCodexContinuity(ctx context.Context, cacheKey string, missReason string) codexContinuity {
	continuity := codexContinuity{CacheKey: cacheKey}
	if cacheKey != "" {
		if cache, ok := helps.GetCodexCache(cacheKey); ok {
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
		helps.SetCodexCache(cacheKey, helps.CodexCache{
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

func resolveCodexContinuity(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) codexContinuity {
	var incomingHeaders http.Header
	if ginCtx := ginContextFrom(ctx); ginCtx != nil && ginCtx.Request != nil {
		incomingHeaders = ginCtx.Request.Header
	}

	continuity := codexContinuity{
		CacheKey: buildCodexSessionCacheKey(auth, req.Payload, incomingHeaders, codexSessionFallbackAPIKey(ctx, auth)),
	}
	incomingSessionID := strings.TrimSpace(incomingHeaders.Get("Session_id"))
	incomingConversationID := strings.TrimSpace(incomingHeaders.Get("Conversation_id"))
	incomingPromptCacheKey := codexPromptCacheKey(req.Payload)

	switch {
	case incomingSessionID != "":
		continuity.Key = incomingSessionID
		continuity.Source = "session_id_header"
	case incomingPromptCacheKey != "":
		continuity.Key = incomingPromptCacheKey
		continuity.Source = "prompt_cache_key"
	case incomingConversationID != "":
		continuity.Key = incomingConversationID
		continuity.Source = "conversation_id"
	case metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey) != "":
		continuity.Key = metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey)
		continuity.Source = "execution_session"
	case strings.TrimSpace(incomingHeaders.Get("Idempotency-Key")) != "":
		continuity.Key = strings.TrimSpace(incomingHeaders.Get("Idempotency-Key"))
		continuity.Source = "idempotency_key"
	default:
		continuity = resolveManagedCodexContinuity(
			ctx,
			continuity.CacheKey,
			fmt.Sprintf("resolve codex continuity miss (from=%s)", strings.TrimSpace(opts.SourceFormat.String())),
		)
	}
	return continuity
}

func applyCodexContinuityBody(rawJSON []byte, continuity codexContinuity) []byte {
	if continuity.Key == "" {
		return rawJSON
	}
	rawJSON, _ = sjson.SetBytes(rawJSON, "prompt_cache_key", continuity.Key)
	return setPromptCacheKeyInContexts(rawJSON, continuity.Key)
}

func applyCodexContinuityHeaders(headers http.Header, continuity codexContinuity) {
	if headers == nil || continuity.Key == "" {
		return
	}
	headers.Set("Session_id", continuity.Key)
}

func logCodexRequestDiagnostics(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, headers http.Header, body []byte, continuity codexContinuity) {
	if !log.IsLevelEnabled(log.DebugLevel) {
		return
	}
	entry := logWithRequestID(ctx)
	authID := ""
	authFile := ""
	if auth != nil {
		authID = strings.TrimSpace(auth.ID)
		authFile = strings.TrimSpace(auth.FileName)
	}
	selectedAuthID := metadataString(opts.Metadata, cliproxyexecutor.SelectedAuthMetadataKey)
	executionSessionID := metadataString(opts.Metadata, cliproxyexecutor.ExecutionSessionMetadataKey)
	entry.Debugf(
		"codex request diagnostics auth_id=%s selected_auth_id=%s auth_file=%s exec_session=%s continuity_source=%s continuity_cache_key=%s session_id=%s prompt_cache_key=%s prompt_cache_retention=%s store=%t has_instructions=%t reasoning_effort=%s reasoning_summary=%s chatgpt_account_id=%t originator=%s model=%s source_format=%s",
		authID,
		selectedAuthID,
		authFile,
		executionSessionID,
		continuity.Source,
		continuity.CacheKey,
		strings.TrimSpace(headers.Get("Session_id")),
		gjson.GetBytes(body, "prompt_cache_key").String(),
		gjson.GetBytes(body, "prompt_cache_retention").String(),
		gjson.GetBytes(body, "store").Bool(),
		gjson.GetBytes(body, "instructions").Exists(),
		gjson.GetBytes(body, "reasoning.effort").String(),
		gjson.GetBytes(body, "reasoning.summary").String(),
		strings.TrimSpace(headers.Get("Chatgpt-Account-Id")) != "",
		strings.TrimSpace(headers.Get("Originator")),
		req.Model,
		opts.SourceFormat.String(),
	)
}

func (e *CodexExecutor) ensureCodexSessionTripleOnRawRequest(req *http.Request, auth *cliproxyauth.Auth) {
	if req == nil || req.URL == nil || req.Body == nil {
		return
	}
	if req.Method != "" && !strings.EqualFold(req.Method, http.MethodPost) {
		return
	}

	path := strings.TrimSpace(req.URL.Path)
	if path != "" && !strings.HasSuffix(path, "/responses") && !strings.HasSuffix(path, "/responses/compact") {
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
		continuity = resolveManagedCodexContinuity(req.Context(), continuity.CacheKey, "session_id and prompt_cache_key missing on raw request")
	}
	if continuity.Key != "" {
		body = applyCodexContinuityBody(body, continuity)
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
	if path != "" && !strings.HasSuffix(path, "/responses") && !strings.HasSuffix(path, "/responses/compact") {
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

func (e *CodexExecutor) cacheHelper(ctx context.Context, auth *cliproxyauth.Auth, _ sdktranslator.Format, url string, req cliproxyexecutor.Request, opts cliproxyexecutor.Options, rawJSON []byte) (*http.Request, codexContinuity, []byte, error) {
	continuity := resolveCodexContinuity(ctx, auth, req, opts)
	rawJSON = applyCodexContinuityBody(rawJSON, continuity)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawJSON))
	if err != nil {
		return nil, continuity, rawJSON, err
	}
	applyCodexContinuityHeaders(httpReq.Header, continuity)
	return httpReq, continuity, rawJSON, nil
}
func applyCodexHeaders(r *http.Request, auth *cliproxyauth.Auth, token string, stream bool, cfg *config.Config) {
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+token)

	var ginHeaders http.Header
	if ginCtx, ok := r.Context().Value("gin").(*gin.Context); ok && ginCtx != nil && ginCtx.Request != nil {
		ginHeaders = ginCtx.Request.Header
	}

	if ginHeaders.Get("X-Codex-Beta-Features") != "" {
		r.Header.Set("X-Codex-Beta-Features", ginHeaders.Get("X-Codex-Beta-Features"))
	}
	misc.EnsureHeader(r.Header, ginHeaders, "Version", "")
	misc.EnsureHeader(r.Header, ginHeaders, "X-Codex-Turn-Metadata", "")
	misc.EnsureHeader(r.Header, ginHeaders, "X-Client-Request-Id", "")
	cfgUserAgent, _ := codexHeaderDefaults(cfg, auth)
	ensureHeaderWithConfigPrecedence(r.Header, ginHeaders, "User-Agent", cfgUserAgent, codexUserAgent)

	if strings.Contains(r.Header.Get("User-Agent"), "Mac OS") {
		misc.EnsureHeader(r.Header, ginHeaders, "Session_id", uuid.NewString())
	}

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
	if originator := strings.TrimSpace(ginHeaders.Get("Originator")); originator != "" {
		r.Header.Set("Originator", originator)
	} else if !isAPIKey {
		r.Header.Set("Originator", codexOriginator)
	}
	if !isAPIKey {
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

func newCodexStatusErr(statusCode int, body []byte) statusErr {
	errCode := statusCode
	if isCodexModelCapacityError(body) {
		errCode = http.StatusTooManyRequests
	}
	body = classifyCodexStatusError(errCode, body)
	err := statusErr{code: errCode, msg: string(body)}
	if retryAfter := parseCodexRetryAfter(errCode, body, time.Now()); retryAfter != nil {
		err.retryAfter = retryAfter
	}
	return err
}

func classifyCodexStatusError(statusCode int, body []byte) []byte {
	code, errType, ok := codexStatusErrorClassification(statusCode, body)
	if !ok {
		return body
	}
	message := gjson.GetBytes(body, "error.message").String()
	if message == "" {
		message = gjson.GetBytes(body, "message").String()
	}
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" {
		message = http.StatusText(statusCode)
	}
	out := []byte(`{"error":{}}`)
	out, _ = sjson.SetBytes(out, "error.message", message)
	out, _ = sjson.SetBytes(out, "error.type", errType)
	out, _ = sjson.SetBytes(out, "error.code", code)
	return out
}

func codexStatusErrorClassification(statusCode int, body []byte) (code string, errType string, ok bool) {
	errorMessage := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.message").String()))
	if errorMessage == "" {
		errorMessage = strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "message").String()))
	}
	lower := strings.ToLower(strings.TrimSpace(string(body)))
	upstreamCode := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.code").String()))
	upstreamType := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.type").String()))
	isInvalidRequest := upstreamType == "" || upstreamType == "invalid_request_error"

	switch {
	case statusCode == http.StatusRequestEntityTooLarge || upstreamCode == "context_length_exceeded" || upstreamCode == "context_too_large" || isInvalidRequest && (strings.Contains(errorMessage, "context length") || strings.Contains(errorMessage, "context_length") || strings.Contains(errorMessage, "maximum context") || strings.Contains(errorMessage, "too many tokens")):
		return "context_too_large", "invalid_request_error", true
	case strings.Contains(lower, "invalid signature in thinking block") || strings.Contains(lower, "invalid_encrypted_content"):
		return "thinking_signature_invalid", "invalid_request_error", true
	case upstreamCode == "previous_response_not_found" || strings.Contains(lower, "previous_response_not_found") || strings.Contains(lower, "previous_response_id") && strings.Contains(lower, "not found"):
		return "previous_response_not_found", "invalid_request_error", true
	case statusCode == http.StatusUnauthorized || upstreamType == "authentication_error" || upstreamCode == "invalid_api_key" || strings.Contains(lower, "invalid or expired token") || strings.Contains(lower, "refresh_token_reused"):
		return "auth_unavailable", "authentication_error", true
	default:
		return "", "", false
	}
}

func normalizeCodexInstructions(body []byte) []byte {
	instructions := gjson.GetBytes(body, "instructions")
	if !instructions.Exists() || instructions.Type == gjson.Null {
		body, _ = sjson.SetBytes(body, "instructions", "")
	}
	return body
}

var imageGenToolJSON = []byte(`{"type":"image_generation","output_format":"png"}`)
var imageGenToolArrayJSON = []byte(`[{"type":"image_generation","output_format":"png"}]`)

func isCodexFreePlanAuth(auth *cliproxyauth.Auth) bool {
	if auth == nil || auth.Attributes == nil {
		return false
	}
	if !strings.EqualFold(strings.TrimSpace(auth.Provider), "codex") {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(auth.Attributes["plan_type"]), "free")
}

func ensureImageGenerationTool(body []byte, baseModel string, auth *cliproxyauth.Auth) []byte {
	if strings.HasSuffix(baseModel, "spark") {
		return body
	}
	if isCodexFreePlanAuth(auth) {
		return body
	}

	tools := gjson.GetBytes(body, "tools")
	if !tools.Exists() || !tools.IsArray() {
		body, _ = sjson.SetRawBytes(body, "tools", imageGenToolArrayJSON)
		return body
	}
	for _, t := range tools.Array() {
		if t.Get("type").String() == "image_generation" {
			return body
		}
	}
	body, _ = sjson.SetRawBytes(body, "tools.-1", imageGenToolJSON)
	return body
}

func publishCodexImageToolUsage(ctx context.Context, reporter *helps.UsageReporter, body []byte, completedData []byte) {
	detail, ok := helps.ParseCodexImageToolUsage(completedData)
	if !ok {
		return
	}
	reporter.EnsurePublished(ctx)
	reporter.PublishAdditionalModel(ctx, codexImageGenerationToolModel(body), detail)
}

func codexImageGenerationToolModel(body []byte) string {
	tools := gjson.GetBytes(body, "tools")
	if tools.IsArray() {
		for _, tool := range tools.Array() {
			if tool.Get("type").String() != "image_generation" {
				continue
			}
			if model := strings.TrimSpace(tool.Get("model").String()); model != "" {
				return model
			}
			break
		}
	}
	return codexDefaultImageToolModel
}

func isCodexModelCapacityError(errorBody []byte) bool {
	if len(errorBody) == 0 {
		return false
	}
	candidates := []string{
		gjson.GetBytes(errorBody, "error.message").String(),
		gjson.GetBytes(errorBody, "message").String(),
		string(errorBody),
	}
	for _, candidate := range candidates {
		lower := strings.ToLower(strings.TrimSpace(candidate))
		if lower == "" {
			continue
		}
		if strings.Contains(lower, "selected model is at capacity") ||
			strings.Contains(lower, "model is at capacity. please try a different model") {
			return true
		}
	}
	return false
}

func parseCodexRetryAfter(statusCode int, errorBody []byte, now time.Time) *time.Duration {
	if statusCode != http.StatusTooManyRequests || len(errorBody) == 0 {
		return nil
	}
	if strings.TrimSpace(gjson.GetBytes(errorBody, "error.type").String()) != "usage_limit_reached" {
		return nil
	}
	if resetsAt := gjson.GetBytes(errorBody, "error.resets_at").Int(); resetsAt > 0 {
		resetAtTime := time.Unix(resetsAt, 0)
		if resetAtTime.After(now) {
			retryAfter := resetAtTime.Sub(now)
			return &retryAfter
		}
	}
	if resetsInSeconds := gjson.GetBytes(errorBody, "error.resets_in_seconds").Int(); resetsInSeconds > 0 {
		retryAfter := time.Duration(resetsInSeconds) * time.Second
		return &retryAfter
	}
	return nil
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
