#!/usr/bin/env python3
"""Check available models in the Bytebot agent."""

import requests
import json
import os

def check_models():
    base_url = os.getenv("BYTEBOT_AGENT_BASE_URL", "http://localhost:9991")

    try:
        response = requests.get(f"{base_url}/tasks/models", timeout=5)
        response.raise_for_status()
        models = response.json()

        print("=" * 60)
        print("AVAILABLE MODELS IN BYTEBOT")
        print("=" * 60)

        # Group models by provider
        by_provider = {}
        for model in models:
            provider = model.get("provider", "unknown")
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(model)

        # Display models by provider
        for provider, provider_models in sorted(by_provider.items()):
            print(f"\n{provider.upper()} Models:")
            print("-" * 40)
            for model in provider_models:
                name = model.get("name", "unknown")
                title = model.get("title", name)
                context = model.get("contextWindow", "?")
                print(f"  • {title}")
                print(f"    Name: {name}")
                print(f"    Context: {context} tokens")

                # Check if this is an Azure model
                if "Azure:" in title:
                    print(f"    ⭐ Azure Deployment Ready!")

        print("\n" + "=" * 60)
        print(f"Total models available: {len(models)}")

        # Check for Azure models specifically
        azure_models = [m for m in models if "Azure:" in m.get("title", "")]
        if azure_models:
            print(f"\n✅ Azure Claude is configured and available!")
            print(f"   Deployment: {azure_models[0].get('name')}")
        else:
            print("\n⚠️  No Azure models found. To enable Azure Claude:")
            print("   1. Set AZURE_ANTHROPIC_* environment variables")
            print("   2. Restart Docker services")

    except requests.ConnectionError:
        print(f"❌ Cannot connect to Bytebot agent at {base_url}")
        print("   Make sure services are running:")
        print("   docker-compose -f docker/docker-compose.yml up -d")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_models()