# vLLM Integration Guide for Bytebot

This guide explains how to configure Bytebot to use vLLM (Versatile Large Language Model) endpoints for running models with high-performance inference.

## Overview

vLLM provides an OpenAI-compatible API server that can serve various open-source models with optimized performance. Bytebot integrates with vLLM through its existing proxy service, which handles OpenAI-compatible endpoints.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   bytebot-ui    │────▶│ bytebot-agent   │────▶│  ProxyService   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │   vLLM Server   │
                                                 │  (OpenAI API)   │
                                                 └─────────────────┘
```

## Prerequisites

1. **vLLM Server**: You need a running vLLM server. Install and start it with:

```bash
# Install vLLM
pip install vllm

# Start vLLM server with a model
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --api-key your-api-key \
  --port 8000
```

2. **Bytebot Stack**: Ensure all Bytebot services are running.

## Configuration

### Environment Variables

Add these variables to your `.env` file in the project root:

```bash
# Required: vLLM server base URL (without /v1 suffix)
VLLM_BASE_URL=http://localhost:8000

# Optional: API key if your vLLM server requires authentication
VLLM_API_KEY=your-api-key

# Required: Comma-separated list of model names available on your vLLM server
# These should match the model names used when starting vLLM
VLLM_MODEL_NAMES=meta-llama/Meta-Llama-3.1-70B-Instruct,mistralai/Mistral-7B-Instruct-v0.3

# Optional: Context window size for vLLM models (default: 32768)
# Set this based on your model's capabilities
VLLM_CONTEXT_WINDOW=131072
```

### Multiple Models

To serve multiple models with vLLM, you can either:

1. **Run multiple vLLM instances** on different ports:
```bash
# Terminal 1
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --port 8000

# Terminal 2
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --port 8001
```

2. **Use a vLLM server with model switching** (if supported by your deployment).

## Usage

### Web Interface

Once configured, vLLM models will appear in the Bytebot UI's model selector with "vLLM:" prefix:
- vLLM: Llama 3.1 70B
- vLLM: Mistral 7B

### API Usage

When using the API directly, specify the model in your request:

```bash
curl -X POST http://localhost:9991/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Open a new browser tab",
    "model": {
      "provider": "proxy",
      "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
      "title": "vLLM: Llama 3.1 70B"
    }
  }'
```

### OSWorld Benchmark

Run the OSWorld benchmark with vLLM models:

```bash
# Navigate to benchmark directory
cd osworld-bench

# Run with vLLM model
python3 run_benchmark.py --model '{
  "provider": "proxy",
  "name": "meta-llama/Meta-Llama-3.1-70B-Instruct",
  "title": "vLLM: Llama 3.1 70B"
}'

# Run specific domain tasks
python3 run_benchmark.py --domain chrome --max-tasks 10 --model '{
  "provider": "proxy",
  "name": "mistralai/Mistral-7B-Instruct-v0.3",
  "title": "vLLM: Mistral 7B"
}'
```

## Supported Models

vLLM supports a wide range of open-source models. Popular choices include:

### Large Models (70B+)
- `meta-llama/Meta-Llama-3.1-70B-Instruct` - Llama 3.1 70B
- `meta-llama/Meta-Llama-3.1-405B-Instruct` - Llama 3.1 405B (requires significant resources)
- `Qwen/Qwen2.5-72B-Instruct` - Qwen 2.5 72B

### Medium Models (7B-34B)
- `meta-llama/Meta-Llama-3.1-8B-Instruct` - Llama 3.1 8B
- `mistralai/Mistral-7B-Instruct-v0.3` - Mistral 7B
- `mistralai/Mixtral-8x7B-Instruct-v0.1` - Mixtral 8x7B
- `codellama/CodeLlama-34b-Instruct-hf` - CodeLlama 34B

### Vision Models (Multimodal)
- `llava-hf/llava-1.5-7b-hf` - LLaVA 1.5 7B
- `llava-hf/llava-1.5-13b-hf` - LLaVA 1.5 13B

## Performance Optimization

### vLLM Server Options

Optimize your vLLM server for better performance:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \  # Use 4 GPUs for tensor parallelism
  --max-model-len 32768 \     # Set maximum sequence length
  --gpu-memory-utilization 0.95 \  # Use 95% of GPU memory
  --max-num-seqs 256          # Maximum number of sequences to process
```

### Context Window Considerations

- Set `VLLM_CONTEXT_WINDOW` based on your model's actual capabilities
- Larger context windows use more memory but handle longer conversations
- For OSWorld tasks, 32K-64K tokens is usually sufficient

## Troubleshooting

### Common Issues

1. **Models not appearing in UI**
   - Check that `VLLM_BASE_URL` and `VLLM_MODEL_NAMES` are set correctly
   - Restart the bytebot-agent service after changing environment variables

2. **Connection errors**
   - Verify vLLM server is running: `curl http://localhost:8000/v1/models`
   - Check firewall settings if vLLM is on a remote server

3. **Authentication errors**
   - Ensure `VLLM_API_KEY` matches the key used when starting vLLM server
   - Try without API key if authentication isn't required

4. **Out of memory errors**
   - Reduce `--gpu-memory-utilization` when starting vLLM
   - Use a smaller model or enable tensor parallelism
   - Reduce `VLLM_CONTEXT_WINDOW` setting

### Debugging

Enable detailed logging:

```bash
# For vLLM server
export VLLM_LOGGING_LEVEL=DEBUG

# For Bytebot agent
export LOG_LEVEL=debug
```

Check logs:
```bash
# Bytebot agent logs
docker logs bytebot-agent

# or if running locally
npm run start:dev
```

## Docker Deployment

For production deployments using Docker:

```yaml
# docker-compose.yml addition
services:
  bytebot-agent:
    environment:
      - VLLM_BASE_URL=http://vllm-server:8000
      - VLLM_API_KEY=${VLLM_API_KEY}
      - VLLM_MODEL_NAMES=${VLLM_MODEL_NAMES}
      - VLLM_CONTEXT_WINDOW=${VLLM_CONTEXT_WINDOW}

  vllm-server:
    image: vllm/vllm-openai:latest
    command: --model meta-llama/Meta-Llama-3.1-70B-Instruct
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Security Considerations

1. **API Keys**: Always use API keys in production to prevent unauthorized access
2. **Network**: Use HTTPS for remote vLLM servers
3. **Rate Limiting**: Configure rate limits on your vLLM server
4. **Resource Limits**: Set appropriate memory and compute limits

## Monitoring

Monitor your vLLM deployment:

```bash
# Check model availability
curl http://localhost:8000/v1/models

# Health check
curl http://localhost:8000/health

# Get current stats (if enabled)
curl http://localhost:8000/metrics
```

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Supported Models List](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [OpenAI API Compatibility](https://platform.openai.com/docs/api-reference)