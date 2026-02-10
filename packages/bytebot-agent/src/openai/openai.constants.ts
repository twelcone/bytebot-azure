import { BytebotAgentModel } from 'src/agent/agent.types';

export const OPENAI_MODELS: BytebotAgentModel[] = [
  {
    provider: 'openai',
    name: 'o3-2025-04-16',
    title: 'o3',
    contextWindow: 200000,
  },
  {
    provider: 'openai',
    name: 'gpt-4.1-2025-04-14',
    title: 'GPT-4.1',
    contextWindow: 1047576,
  },
];

export const DEFAULT_MODEL = OPENAI_MODELS[0];

const azureDeployment = process.env.AZURE_OPENAI_DEPLOYMENT;

export const AZURE_OPENAI_MODELS: BytebotAgentModel[] = azureDeployment
  ? [
      {
        provider: 'openai',
        name: azureDeployment,
        title: `Azure: ${azureDeployment}`,
        contextWindow: 128000,
      },
    ]
  : [];

// vLLM models configuration
const vllmBaseUrl = process.env.VLLM_BASE_URL;
const vllmModelNames = process.env.VLLM_MODEL_NAMES?.split(',') || [];

export const VLLM_MODELS: BytebotAgentModel[] =
  vllmBaseUrl && vllmModelNames.length > 0
    ? vllmModelNames.map((modelName) => ({
        provider: 'openai' as const,  // Use openai provider for vLLM (OpenAI-compatible API)
        name: modelName.trim(),
        title: `vLLM: ${modelName.trim()}`,
        contextWindow: parseInt(process.env.VLLM_CONTEXT_WINDOW || '32768'),
      }))
    : [];
