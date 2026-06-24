#!/usr/bin/env python3
"""
DiCAI Python API Client
Simple client for testing the distributed inference API
"""

import argparse
import requests
import json
import sys


def chat_completion(api_url: str, model: str, prompt: str, max_tokens: int = 50):
    """Send a chat completion request to the DiCAI API"""
    
    endpoint = f"{api_url}/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def list_providers(api_url: str):
    """List all registered providers"""
    
    endpoint = f"{api_url}/api/v1/providers"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def list_models(api_url: str):
    """List available models"""
    
    endpoint = f"{api_url}/v1/models"
    
    try:
        response = requests.get(endpoint)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='DiCAI API Client')
    parser.add_argument('--api-url', default='http://localhost:8080', help='Admin API URL')
    parser.add_argument('--model', default='llama-3-70b', help='Model ID')
    parser.add_argument('--prompt', default='Hello!', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--list-providers', action='store_true', help='List providers')
    parser.add_argument('--list-models', action='store_true', help='List models')
    
    args = parser.parse_args()
    
    if args.list_providers:
        providers = list_providers(args.api_url)
        if providers:
            print(json.dumps(providers, indent=2))
        return
    
    if args.list_models:
        models = list_models(args.api_url)
        if models:
            print(json.dumps(models, indent=2))
        return
    
    # Send chat completion
    print(f"Sending request to {args.api_url}...")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    result = chat_completion(args.api_url, args.model, args.prompt, args.max_tokens)
    
    if result:
        print("Response:")
        print(json.dumps(result, indent=2))
        
        # Extract and print just the text
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0].get('message', {}).get('content', '')
            print("\nGenerated text:")
            print(content)
    else:
        print("Failed to get response")
        sys.exit(1)


if __name__ == '__main__':
    main()
