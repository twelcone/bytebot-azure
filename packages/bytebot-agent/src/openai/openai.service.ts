import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI, { APIUserAbortError } from 'openai';
import {
  MessageContentBlock,
  MessageContentType,
  TextContentBlock,
  ToolUseContentBlock,
  ToolResultContentBlock,
  ThinkingContentBlock,
  ImageContentBlock,
  isUserActionContentBlock,
  isComputerToolUseContentBlock,
  isImageContentBlock,
} from '@bytebot/shared';
import { DEFAULT_MODEL } from './openai.constants';
import { Message, Role } from '@prisma/client';
import { openaiTools, chatCompletionTools } from './openai.tools';
import {
  BytebotAgentService,
  BytebotAgentInterrupt,
  BytebotAgentResponse,
} from '../agent/agent.types';

@Injectable()
export class OpenAIService implements BytebotAgentService {
  private readonly openai: OpenAI;
  private readonly logger = new Logger(OpenAIService.name);

  constructor(private readonly configService: ConfigService) {
    // Check for vLLM configuration first
    const vllmBaseUrl = this.configService.get<string>('VLLM_BASE_URL');

    if (vllmBaseUrl) {
      // Configure for vLLM
      const vllmApiKey = this.configService.get<string>('VLLM_API_KEY');

      this.openai = new OpenAI({
        baseURL: `${vllmBaseUrl}/v1`,
        apiKey: vllmApiKey || 'dummy-key',
      });
      this.logger.log(
        `OpenAIService configured for vLLM (endpoint: ${vllmBaseUrl})`,
      );
    } else {
      // Original Azure/OpenAI configuration
      const azureEndpoint = this.configService.get<string>(
        'AZURE_OPENAI_ENDPOINT',
      );

      if (azureEndpoint) {
        const azureApiKey = this.configService.get<string>(
          'AZURE_OPENAI_API_KEY',
        );

        if (!azureApiKey) {
          this.logger.warn(
            'AZURE_OPENAI_API_KEY is not set. Azure OpenAI will not work properly.',
          );
        }

        this.openai = new OpenAI({
          baseURL: azureEndpoint,
          apiKey: azureApiKey || 'dummy-key-for-initialization',
        });
        this.logger.log(
          `OpenAIService configured for Azure OpenAI (endpoint: ${azureEndpoint})`,
        );
      } else {
        const apiKey = this.configService.get<string>('OPENAI_API_KEY');

        if (!apiKey) {
          this.logger.warn(
            'OPENAI_API_KEY is not set. OpenAIService will not work properly.',
          );
        }

        this.openai = new OpenAI({
          apiKey: apiKey || 'dummy-key-for-initialization',
        });
      }
    }
  }

  async generateMessage(
    systemPrompt: string,
    messages: Message[],
    model: string = DEFAULT_MODEL.name,
    useTools: boolean = true,
    signal?: AbortSignal,
  ): Promise<BytebotAgentResponse> {
    const isVLLM = !!this.configService.get<string>('VLLM_BASE_URL');
    const isReasoning = model.startsWith('o');

    try {
      if (isVLLM) {
        // vLLM uses chat.completions API for better tool support
        const chatMessages = this.formatMessagesForChatCompletion(systemPrompt, messages);

        const response = await this.openai.chat.completions.create(
          {
            model,
            messages: chatMessages,
            max_tokens: 8192,
            ...(useTools && { tools: chatCompletionTools }),
          },
          { signal },
        );

        return {
          contentBlocks: this.formatChatCompletionResponse(response),
          tokenUsage: {
            inputTokens: response.usage?.prompt_tokens || 0,
            outputTokens: response.usage?.completion_tokens || 0,
            totalTokens: response.usage?.total_tokens || 0,
          },
        };
      } else {
        // OpenAI uses responses.create API
        const openaiMessages = this.formatMessagesForOpenAI(messages);

        const maxTokens = 8192;
        const response = await this.openai.responses.create(
          {
            model,
            max_output_tokens: maxTokens,
            input: openaiMessages,
            instructions: systemPrompt,
            tools: useTools ? openaiTools : [],
            reasoning: isReasoning ? { effort: 'medium' } : null,
            store: false,
            include: isReasoning ? ['reasoning.encrypted_content'] : [],
          },
          { signal },
        );

        return {
          contentBlocks: this.formatOpenAIResponse(response.output),
          tokenUsage: {
            inputTokens: response.usage?.input_tokens || 0,
            outputTokens: response.usage?.output_tokens || 0,
            totalTokens: response.usage?.total_tokens || 0,
          },
        };
      }
    } catch (error: any) {
      console.log('error', error);
      console.log('error name', error.name);

      if (error instanceof APIUserAbortError) {
        this.logger.log('OpenAI API call aborted');
        throw new BytebotAgentInterrupt();
      }
      this.logger.error(
        `Error sending message to OpenAI: ${error.message}`,
        error.stack,
      );
      throw error;
    }
  }

  private formatMessagesForOpenAI(
    messages: Message[],
  ): OpenAI.Responses.ResponseInputItem[] {
    const openaiMessages: OpenAI.Responses.ResponseInputItem[] = [];

    for (const message of messages) {
      const messageContentBlocks = message.content as MessageContentBlock[];

      if (
        messageContentBlocks.every((block) => isUserActionContentBlock(block))
      ) {
        const userActionContentBlocks = messageContentBlocks.flatMap(
          (block) => block.content,
        );
        for (const block of userActionContentBlocks) {
          if (isComputerToolUseContentBlock(block)) {
            openaiMessages.push({
              type: 'message',
              role: 'user',
              content: [
                {
                  type: 'input_text',
                  text: `User performed action: ${block.name}\n${JSON.stringify(block.input, null, 2)}`,
                },
              ],
            });
          } else if (isImageContentBlock(block)) {
            openaiMessages.push({
              role: 'user',
              type: 'message',
              content: [
                {
                  type: 'input_image',
                  detail: 'high',
                  image_url: `data:${block.source.media_type};base64,${block.source.data}`,
                },
              ],
            } as OpenAI.Responses.ResponseInputItem.Message);
          }
        }
      } else {
        // Convert content blocks to OpenAI format
        for (const block of messageContentBlocks) {
          switch (block.type) {
            case MessageContentType.Text: {
              if (message.role === Role.USER) {
                openaiMessages.push({
                  type: 'message',
                  role: 'user',
                  content: [
                    {
                      type: 'input_text',
                      text: block.text,
                    },
                  ],
                } as OpenAI.Responses.ResponseInputItem.Message);
              } else {
                openaiMessages.push({
                  type: 'message',
                  role: 'assistant',
                  content: [
                    {
                      type: 'output_text',
                      text: block.text,
                    },
                  ],
                } as OpenAI.Responses.ResponseOutputMessage);
              }
              break;
            }
            case MessageContentType.ToolUse:
              // For assistant messages with tool use, convert to function call
              if (message.role === Role.ASSISTANT) {
                const toolBlock = block as ToolUseContentBlock;
                openaiMessages.push({
                  type: 'function_call',
                  call_id: toolBlock.id,
                  name: toolBlock.name,
                  arguments: JSON.stringify(toolBlock.input),
                } as OpenAI.Responses.ResponseFunctionToolCall);
              }
              break;

            case MessageContentType.Thinking: {
              const thinkingBlock = block;
              openaiMessages.push({
                type: 'reasoning',
                id: thinkingBlock.signature,
                encrypted_content: thinkingBlock.thinking,
                summary: [],
              } as OpenAI.Responses.ResponseReasoningItem);
              break;
            }
            case MessageContentType.ToolResult: {
              // Handle tool results as function call outputs
              const toolResult = block;
              // Tool results should be added as separate items in the response

              toolResult.content.forEach((content) => {
                if (content.type === MessageContentType.Text) {
                  openaiMessages.push({
                    type: 'function_call_output',
                    call_id: toolResult.tool_use_id,
                    output: content.text,
                  } as OpenAI.Responses.ResponseInputItem.FunctionCallOutput);
                }

                if (content.type === MessageContentType.Image) {
                  openaiMessages.push({
                    type: 'function_call_output',
                    call_id: toolResult.tool_use_id,
                    output: 'screenshot',
                  } as OpenAI.Responses.ResponseInputItem.FunctionCallOutput);
                  openaiMessages.push({
                    role: 'user',
                    type: 'message',
                    content: [
                      {
                        type: 'input_image',
                        detail: 'high',
                        image_url: `data:${content.source.media_type};base64,${content.source.data}`,
                      },
                    ],
                  } as OpenAI.Responses.ResponseInputItem.Message);
                }
              });
              break;
            }

            default:
              // Handle unknown content types as text
              openaiMessages.push({
                role: 'user',
                type: 'message',
                content: [
                  {
                    type: 'input_text',
                    text: JSON.stringify(block),
                  },
                ],
              } as OpenAI.Responses.ResponseInputItem.Message);
          }
        }
      }
    }

    return openaiMessages;
  }

  private formatOpenAIResponse(
    response: OpenAI.Responses.ResponseOutputItem[],
  ): MessageContentBlock[] {
    const contentBlocks: MessageContentBlock[] = [];

    for (const item of response) {
      // Check the type of the output item
      switch (item.type) {
        case 'message':
          // Handle ResponseOutputMessage
          const message = item;
          for (const content of message.content) {
            if ('text' in content) {
              // ResponseOutputText
              contentBlocks.push({
                type: MessageContentType.Text,
                text: content.text,
              } as TextContentBlock);
            } else if ('refusal' in content) {
              // ResponseOutputRefusal
              contentBlocks.push({
                type: MessageContentType.Text,
                text: `Refusal: ${content.refusal}`,
              } as TextContentBlock);
            }
          }
          break;

        case 'function_call':
          // Handle ResponseFunctionToolCall
          const toolCall = item;
          contentBlocks.push({
            type: MessageContentType.ToolUse,
            id: toolCall.call_id,
            name: toolCall.name,
            input: JSON.parse(toolCall.arguments),
          } as ToolUseContentBlock);
          break;

        case 'file_search_call':
        case 'web_search_call':
        case 'computer_call':
        case 'reasoning':
          const reasoning = item as OpenAI.Responses.ResponseReasoningItem;
          if (reasoning.encrypted_content) {
            contentBlocks.push({
              type: MessageContentType.Thinking,
              thinking: reasoning.encrypted_content,
              signature: reasoning.id,
            } as ThinkingContentBlock);
          }
          break;
        case 'image_generation_call':
        case 'code_interpreter_call':
        case 'local_shell_call':
        case 'mcp_call':
        case 'mcp_list_tools':
        case 'mcp_approval_request':
          // Handle other tool types as text for now
          this.logger.warn(
            `Unsupported response output item type: ${item.type}`,
          );
          contentBlocks.push({
            type: MessageContentType.Text,
            text: JSON.stringify(item),
          } as TextContentBlock);
          break;

        default:
          // Handle unknown types
          this.logger.warn(
            `Unknown response output item type: ${JSON.stringify(item)}`,
          );
          contentBlocks.push({
            type: MessageContentType.Text,
            text: JSON.stringify(item),
          } as TextContentBlock);
      }
    }

    return contentBlocks;
  }

  /**
   * Format messages for vLLM Chat Completions API
   */
  private formatMessagesForChatCompletion(
    systemPrompt: string,
    messages: Message[],
  ): any[] {
    const chatMessages: any[] = [];

    // Add system message
    chatMessages.push({
      role: 'system',
      content: systemPrompt,
    });

    // Process each message
    for (const message of messages) {
      const messageContentBlocks = message.content as MessageContentBlock[];

      for (const block of messageContentBlocks) {
        switch (block.type) {
          case MessageContentType.Text: {
            chatMessages.push({
              role: message.role === Role.USER ? 'user' : 'assistant',
              content: block.text,
            });
            break;
          }
          case MessageContentType.Image: {
            const imageBlock = block as ImageContentBlock;
            chatMessages.push({
              role: 'user',
              content: [
                {
                  type: 'image_url',
                  image_url: {
                    url: `data:${imageBlock.source.media_type};base64,${imageBlock.source.data}`,
                    detail: 'high',
                  },
                },
              ],
            });
            break;
          }
          case MessageContentType.ToolUse: {
            const toolBlock = block as ToolUseContentBlock;
            chatMessages.push({
              role: 'assistant',
              tool_calls: [
                {
                  id: toolBlock.id,
                  type: 'function',
                  function: {
                    name: toolBlock.name,
                    arguments: JSON.stringify(toolBlock.input),
                  },
                },
              ],
            });
            break;
          }
          case MessageContentType.ToolResult: {
            const toolResultBlock = block as ToolResultContentBlock;
            toolResultBlock.content.forEach((content) => {
              if (content.type === MessageContentType.Text) {
                chatMessages.push({
                  role: 'tool',
                  tool_call_id: toolResultBlock.tool_use_id,
                  content: content.text,
                });
              }
            });
            break;
          }
        }
      }
    }

    return chatMessages;
  }

  /**
   * Format Chat Completion response to MessageContentBlocks
   */
  private formatChatCompletionResponse(
    response: any,
  ): MessageContentBlock[] {
    const contentBlocks: MessageContentBlock[] = [];
    const message = response.choices?.[0]?.message;

    if (!message) {
      return contentBlocks;
    }

    // Handle text content
    if (message.content) {
      contentBlocks.push({
        type: MessageContentType.Text,
        text: message.content,
      } as TextContentBlock);
    }

    // Handle tool calls
    if (message.tool_calls && message.tool_calls.length > 0) {
      for (const toolCall of message.tool_calls) {
        if (toolCall.type === 'function') {
          let parsedInput = {};
          try {
            parsedInput = JSON.parse(toolCall.function.arguments || '{}');
          } catch (e) {
            this.logger.warn(
              `Failed to parse tool call arguments: ${toolCall.function.arguments}`,
            );
            parsedInput = {};
          }

          contentBlocks.push({
            type: MessageContentType.ToolUse,
            id: toolCall.id,
            name: toolCall.function.name,
            input: parsedInput,
          } as ToolUseContentBlock);
        }
      }
    }

    return contentBlocks;
  }
}
