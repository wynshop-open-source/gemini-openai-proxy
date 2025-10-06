package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"google.golang.org/genai"
	"iter"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/pkg/errors"
	"github.com/sashabaranov/go-openai"
	"google.golang.org/api/googleapi"

	"github.com/zhu327/gemini-openai-proxy/pkg/util"
)

const (
	genaiRoleUser  = "user"
	genaiRoleModel = "model"
)

type GeminiAdapter struct {
	client *genai.Client
	model  string
}

func NewGeminiAdapter(client *genai.Client, model string) *GeminiAdapter {
	return &GeminiAdapter{
		client: client,
		model:  model,
	}
}

func (g *GeminiAdapter) GenerateContent(
	ctx context.Context,
	req *ChatCompletionRequest,
	messages []*genai.Content,
) (*openai.ChatCompletionResponse, error) {
	// Remove 'models/' prefix if present
	modelName := strings.TrimPrefix(g.model, "models/")

	conf := getGenaiContentConfigByOpenaiRequest(req)
	chat, err := g.client.Chats.Create(ctx, modelName, &conf, messages)
	if err != nil {
		return nil, errors.Wrap(err, "genai create chat error")
	}

	lastMessageParts := messages[len(messages)-1].Parts
	parts := make([]genai.Part, len(lastMessageParts))
	for i, p := range lastMessageParts {
		parts[i] = *p
	}
	genaiResp, err := chat.SendMessage(ctx, parts...)
	if err != nil {
		var apiErr *googleapi.Error
		if errors.As(err, &apiErr) {
			if apiErr.Code == http.StatusTooManyRequests {
				return nil, errors.Wrap(&openai.APIError{
					Code:    http.StatusTooManyRequests,
					Message: err.Error(),
				}, "genai send message error")
			}
		} else {
			log.Printf("Error is not of type *googleapi.Error: %v\n", err)
		}
		return nil, errors.Wrap(err, "genai send message error")
	}
	openaiResp := genaiResponseToOpenaiResponse(g.model, genaiResp)
	return &openaiResp, nil
}

func (g *GeminiAdapter) GenerateStreamContent(
	ctx context.Context,
	req *ChatCompletionRequest,
	messages []*genai.Content,
) (<-chan string, error) {
	// Remove 'models/' prefix if present
	modelName := strings.TrimPrefix(g.model, "models/")

	conf := getGenaiContentConfigByOpenaiRequest(req)
	chat, err := g.client.Chats.Create(ctx, modelName, &conf, messages)
	if err != nil {
		return nil, errors.Wrap(err, "genai create chat error")
	}

	lastMessageParts := messages[len(messages)-1].Parts
	parts := make([]genai.Part, len(lastMessageParts))
	for i, p := range lastMessageParts {
		parts[i] = *p
	}
	it := chat.SendMessageStream(ctx, parts...)

	dataChan := make(chan string)
	go handleStreamIter(g.model, it, dataChan, req.StreamOptions.IncludeUsage)

	return dataChan, nil
}

func handleStreamIter(model string, it iter.Seq2[*genai.GenerateContentResponse, error], dataChan chan string, sendUsage bool) {
	defer close(dataChan)

	respID := util.GetUUID()
	created := time.Now().Unix()

	// For character-by-character streaming
	var textBuffer string

	// Counter for character-by-character streaming - increased for better performance
	sentenceLength := 1000
	charCount := 0

	// Function to send a single character with proper formatting
	sendCharacter := func(char string) {
		openaiResp := &CompletionResponse{
			ID:      fmt.Sprintf("chatcmpl-%s", respID),
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   GetMappedModel(model),
			Choices: []CompletionChoice{
				{
					Index: 0,
					Delta: struct {
						Content   string            `json:"content,omitempty"`
						Role      string            `json:"role,omitempty"`
						ToolCalls []openai.ToolCall `json:"tool_calls,omitempty"`
					}{
						Content: char,
					},
				},
			},
		}
		resp, _ := json.Marshal(openaiResp)
		dataChan <- string(resp)
	}

	// Function to send entire text at once (for finish conditions)
	sendFullText := func(text string) {
		if text == "" {
			return
		}
		openaiResp := &CompletionResponse{
			ID:      fmt.Sprintf("chatcmpl-%s", respID),
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   GetMappedModel(model),
			Choices: []CompletionChoice{
				{
					Index: 0,
					Delta: struct {
						Content   string            `json:"content,omitempty"`
						Role      string            `json:"role,omitempty"`
						ToolCalls []openai.ToolCall `json:"tool_calls,omitempty"`
					}{
						Content: text,
					},
				},
			},
		}
		resp, _ := json.Marshal(openaiResp)
		dataChan <- string(resp)
	}

	sendUsageMetadata := func(usage *genai.GenerateContentResponseUsageMetadata) {
		if usage == nil || !sendUsage {
			return
		}
		openaiResp := &CompletionResponse{
			ID:      fmt.Sprintf("chatcmpl-%s", respID),
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   GetMappedModel(model),
			Choices: []CompletionChoice{},
			Usage: openai.Usage{
				PromptTokens:     int(usage.PromptTokenCount),
				CompletionTokens: int(usage.CandidatesTokenCount),
				TotalTokens:      int(usage.TotalTokenCount),
			},
		}
		resp, _ := json.Marshal(openaiResp)
		dataChan <- string(resp)
	}

	var usageMetadata *genai.GenerateContentResponseUsageMetadata

	for genaiResp, err := range it {
		if err != nil {
			log.Printf("genai get stream message error %v\n", err)

			// Check for context cancellation
			if errors.Is(err, context.Canceled) {
				log.Printf("Context was canceled by client")
				apiErr := openai.APIError{
					Code:    http.StatusRequestTimeout,
					Message: "Request was canceled",
					Type:    "canceled_error",
				}
				resp, _ := json.Marshal(apiErr)
				dataChan <- string(resp)
				break
			}

			// Check for rate limit errors
			var apiErr *googleapi.Error
			if errors.As(err, &apiErr) && apiErr.Code == http.StatusTooManyRequests {
				log.Printf("Rate limit exceeded: %v\n", err)
				rateLimitErr := openai.APIError{
					Code:    http.StatusTooManyRequests,
					Message: "Rate limit exceeded",
					Type:    "rate_limit_error",
				}
				resp, _ := json.Marshal(rateLimitErr)
				dataChan <- string(resp)
				break
			}

			// Handle other errors
			generalErr := openai.APIError{
				Code:    http.StatusInternalServerError,
				Message: err.Error(),
				Type:    "internal_server_error",
			}
			resp, _ := json.Marshal(generalErr)
			dataChan <- string(resp)
			break
		}
		// gemini returns the usage data on each response, adding to it each time, so always get the latest
		usageMetadata = genaiResp.UsageMetadata

		// Process each candidate's text content
		for _, candidate := range genaiResp.Candidates {
			if candidate.Content == nil {
				continue
			}

			// Check if this is the last message with a finish reason
			isLastMessage := candidate.FinishReason > genai.FinishReasonStop

			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					text := part.Text
					if isLastMessage {
						// If this is the last message, collect the text in buffer
						textBuffer += text
					} else if charCount < sentenceLength {
						// Stream character by character until we reach sentenceLength
						for i, char := range text {
							if charCount < sentenceLength {
								sendCharacter(string(char))
								// No delay between characters for faster streaming
								charCount++
							} else {
								// Once we've reached sentenceLength, send the rest of this text at once
								remaining := text[i:]
								if remaining != "" {
									sendFullText(remaining)
								}
								break
							}
						}
					} else {
						// For subsequent chunks after sentenceLength, send the entire text at once
						sendFullText(text)
					}
				} else if part.FunctionCall != nil {
					// Handle function calls as before
					openaiResp := genaiResponseToStreamCompletionResponse(model, genaiResp, respID, created)
					resp, _ := json.Marshal(openaiResp)
					dataChan <- string(resp)
				}
			}
		}

		// Send finish reason if present
		if len(genaiResp.Candidates) > 0 && genaiResp.Candidates[0].FinishReason > genai.FinishReasonStop {
			// Send any accumulated text all at once
			if len(textBuffer) > 0 {
				sendFullText(textBuffer)
			}

			// Send the finish reason
			for _, candidate := range genaiResp.Candidates {
				if candidate.FinishReason > genai.FinishReasonStop {
					openaiFinishReason := string(convertFinishReason(candidate.FinishReason))
					openaiResp := &CompletionResponse{
						ID:      fmt.Sprintf("chatcmpl-%s", respID),
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   GetMappedModel(model),
						Choices: []CompletionChoice{
							{
								Index: 0,
								Delta: struct {
									Content   string            `json:"content,omitempty"`
									Role      string            `json:"role,omitempty"`
									ToolCalls []openai.ToolCall `json:"tool_calls,omitempty"`
								}{
									// Empty content for finish reason message
								},
								FinishReason: &openaiFinishReason,
							},
						},
					}
					resp, _ := json.Marshal(openaiResp)
					dataChan <- string(resp)
					break
				}
			}
			break
		}
	}

	// Send any remaining text when done - all at once
	if len(textBuffer) > 0 {
		// Send all remaining text at once when done
		sendFullText(textBuffer)
	}
	// per https://community.openai.com/t/usage-stats-now-available-when-using-streaming-with-the-chat-completions-api-or-completions-api/738156
	// the usage is sent after everything else
	sendUsageMetadata(usageMetadata)
}

func genaiResponseToStreamCompletionResponse(
	model string,
	genaiResp *genai.GenerateContentResponse,
	respID string,
	created int64,
) *CompletionResponse {
	resp := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", respID),
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   GetMappedModel(model),
		Choices: make([]CompletionChoice, 0, len(genaiResp.Candidates)),
	}

	count := 0
	toolCalls := make([]openai.ToolCall, 0)

	for _, candidate := range genaiResp.Candidates {
		parts := candidate.Content.Parts
		for _, part := range parts {
			index := count
			if part.Text != "" {
				choice := CompletionChoice{
					Index: index,
				}
				choice.Delta.Content = part.Text

				if candidate.FinishReason > genai.FinishReasonStop {
					log.Printf("genai message finish reason %s\n", candidate.FinishReason)
					openaiFinishReason := string(convertFinishReason(candidate.FinishReason))
					choice.FinishReason = &openaiFinishReason
				}

				resp.Choices = append(resp.Choices, choice)
			} else if part.FunctionCall != nil {
				args, _ := json.Marshal(part.FunctionCall.Args)
				toolCalls = append(toolCalls, openai.ToolCall{
					Index:    genai.Ptr(int(index)),
					ID:       fmt.Sprintf("%s-%d", part.FunctionCall.Name, index),
					Type:     openai.ToolTypeFunction,
					Function: openai.FunctionCall{Name: part.FunctionCall.Name, Arguments: string(args)},
				})
			}
			count++
		}
	}

	if len(toolCalls) > 0 {
		choice := CompletionChoice{
			Index: 0,
		}
		// For tool calls, we need to set a special finish reason
		openaiFinishReason := string(openai.FinishReasonToolCalls)
		choice.FinishReason = &openaiFinishReason

		// Add the tool calls to the response
		toolCallsJSON, _ := json.Marshal(toolCalls)
		choice.Delta.Content = string(toolCallsJSON)

		resp.Choices = append(resp.Choices, choice)
	}

	return &resp
}

func genaiResponseToOpenaiResponse(
	model string,
	genaiResp *genai.GenerateContentResponse,
) openai.ChatCompletionResponse {
	resp := openai.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", util.GetUUID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   GetMappedModel(model),
		Choices: make([]openai.ChatCompletionChoice, 0, len(genaiResp.Candidates)),
	}

	if genaiResp.UsageMetadata != nil {
		resp.Usage = openai.Usage{
			PromptTokens:     int(genaiResp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(genaiResp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(genaiResp.UsageMetadata.TotalTokenCount),
		}
	}

	for i, candidate := range genaiResp.Candidates {
		toolCalls := make([]openai.ToolCall, 0)
		var content string

		if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			for j, part := range candidate.Content.Parts {
				if part.Text != "" {
					content = part.Text
				} else if part.FunctionCall != nil {
					args, _ := json.Marshal(part.FunctionCall.Args)
					toolCalls = append(toolCalls, openai.ToolCall{
						Index:    genai.Ptr(j),
						ID:       fmt.Sprintf("%s-%d", part.FunctionCall.Name, j),
						Type:     openai.ToolTypeFunction,
						Function: openai.FunctionCall{Name: part.FunctionCall.Name, Arguments: string(args)},
					})
				}
			}
		}

		choice := openai.ChatCompletionChoice{
			Index:        i,
			FinishReason: convertFinishReason(candidate.FinishReason),
		}

		if len(toolCalls) > 0 {
			choice.Message = openai.ChatCompletionMessage{
				Role:      openai.ChatMessageRoleAssistant,
				ToolCalls: toolCalls,
			}
			choice.FinishReason = openai.FinishReasonToolCalls
		} else {
			choice.Message = openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: content,
			}
		}

		resp.Choices = append(resp.Choices, choice)
	}
	return resp
}

func convertFinishReason(reason genai.FinishReason) openai.FinishReason {
	openaiFinishReason := openai.FinishReasonStop
	switch reason {
	case genai.FinishReasonMaxTokens:
		openaiFinishReason = openai.FinishReasonLength
	case genai.FinishReasonSafety, genai.FinishReasonRecitation:
		openaiFinishReason = openai.FinishReasonContentFilter
	}
	return openaiFinishReason
}

func getGenaiContentConfigByOpenaiRequest(req *ChatCompletionRequest) genai.GenerateContentConfig {
	config := genai.GenerateContentConfig{}

	if req.MaxTokens != 0 {
		config.MaxOutputTokens = req.MaxTokens
	}
	if req.Temperature != 0 {
		config.Temperature = &req.Temperature
	}
	if req.TopP != 0 {
		config.TopP = &req.TopP
	}
	if len(req.Stop) != 0 {
		config.StopSequences = req.Stop
	}

	// Set response format if specified
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json" {
		config.ResponseMIMEType = "application/json"
	}

	// Configure tools if provided
	if len(req.Tools) > 0 {
		tools := convertOpenAIToolsToGenAI(req.Tools)
		config.Tools = tools

		// Configure tool choice/function calling mode
		config.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{},
		}

		switch v := req.ToolChoice.(type) {
		case string:
			switch v {
			case "none":
				config.ToolConfig.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeNone
			case "auto":
				config.ToolConfig.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeAuto
			}
		case map[string]interface{}:
			if funcObj, ok := v["function"]; ok {
				if funcMap, ok := funcObj.(map[string]interface{}); ok {
					if name, ok := funcMap["name"].(string); ok {
						config.ToolConfig.FunctionCallingConfig.Mode = genai.FunctionCallingConfigModeAny
						config.ToolConfig.FunctionCallingConfig.AllowedFunctionNames = []string{name}
					}
				}
			}
		}
	}

	config.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
	}
	return config
}

func (g *GeminiAdapter) GenerateEmbedding(
	ctx context.Context,
	messages []*genai.Content,
) (*openai.EmbeddingResponse, error) {
	// Remove 'models/' prefix if present
	modelName := strings.TrimPrefix(g.model, "models/")

	genaiResp, err := g.client.Models.EmbedContent(ctx, modelName, messages, nil)
	if err != nil {
		return nil, errors.Wrap(err, "genai generate embeddings error")
	}

	openaiResp := openai.EmbeddingResponse{
		Object: "list",
		Data:   make([]openai.Embedding, 0, len(genaiResp.Embeddings)),
		Model:  openai.EmbeddingModel(GetMappedModel(g.model)),
	}

	for i, genaiEmbedding := range genaiResp.Embeddings {
		embedding := openai.Embedding{
			Object:    "embedding",
			Embedding: genaiEmbedding.Values,
			Index:     i,
		}
		openaiResp.Data = append(openaiResp.Data, embedding)
	}

	return &openaiResp, nil
}
