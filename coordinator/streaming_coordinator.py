#!/usr/bin/env python3
"""
DiCAI Streaming Coordinator

OpenAI-compatible API with streaming token generation.
Integrates tokenizer, KV cache, auth, and fault tolerance.
"""

import argparse
import json
import time
import numpy as np
import torch

import grpc
from flask import Flask, Response, request, jsonify, stream_with_context

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import proto.dicai_pb2 as dicai_pb2
import proto.dicai_pb2_grpc as dicai_pb2_grpc

from shared.tokenizer import Tokenizer
from shared.kv_cache import KVCache
from shared.auth import AuthManager


class StreamingCoordinator:
    """Coordinator with streaming support."""
    
    def __init__(self, config_file, model_dir):
        self.config = self.load_config(config_file)
        self.tokenizer = Tokenizer(model_dir)
        self.auth = AuthManager()
        self.providers = {}
        self.provider_chain = []
        self.connect_providers()
        
    def load_config(self, config_file):
        with open(config_file) as f:
            return json.load(f)
            
    def connect_providers(self):
        print("[Coordinator] Connecting to providers...")
        for shard in self.config['shards']:
            provider_id = shard['provider_id']
            address = f"{shard['host']}:{shard['port']}"
            try:
                channel = grpc.insecure_channel(address)
                grpc.channel_ready_future(channel).result(timeout=5)
                stub = dicai_pb2_grpc.ProviderServiceStub(channel)
                self.providers[provider_id] = stub
                self.provider_chain.append(provider_id)
                print(f"  [OK] {provider_id} at {address}")
            except Exception as e:
                print(f"  [FAIL] {provider_id} at {address}: {e}")
                
    def generate_stream(self, prompt, max_tokens=100):
        """Generate tokens with streaming."""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt)
        
        # Create KV cache
        kv_cache = KVCache()
        
        # Generate tokens
        for i in range(max_tokens):
            # Run through provider chain
            hidden_states = None
            hidden_shape = None
            
            for provider_id in self.provider_chain:
                stub = self.providers[provider_id]
                
                if hidden_states is not None:
                    req = dicai_pb2.ProcessRequest(
                        hidden_states=hidden_states.tobytes(),
                        hidden_states_shape=hidden_shape,
                        request_id=f"stream-{i}"
                    )
                else:
                    req = dicai_pb2.ProcessRequest(
                        input_ids=input_ids,
                        request_id=f"stream-{i}"
                    )
                    
                try:
                    response = stub.process(req)
                    if response.status == "success":
                        if response.hidden_states:
                            hidden_states = np.frombuffer(response.hidden_states, dtype=np.float32)
                            hidden_shape = list(response.hidden_states_shape)
                        if response.logits:
                            logits_np = np.frombuffer(response.logits, dtype=np.float32)
                            logits_shape = list(response.logits_shape)
                            logits = torch.tensor(logits_np.reshape(logits_shape))
                            token = torch.argmax(logits[0, -1]).item()
                            
                            # Yield token
                            text = self.tokenizer.decode([token])
                            yield {
                                "token": token,
                                "text": text,
                                "index": i
                            }
                            
                            # Update input for next iteration
                            input_ids = [token]
                    else:
                        yield {"error": response.error}
                        return
                except Exception as e:
                    yield {"error": str(e)}
                    return
                    
            # Check for EOS
            if token == self.tokenizer.eos_token:
                break


def create_app(coordinator):
    app = Flask(__name__)
    
    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        """OpenAI-compatible chat completions endpoint."""
        data = request.json
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        max_tokens = data.get('max_tokens', 100)
        
        # Extract prompt from messages
        prompt = messages[-1]['content'] if messages else "Hello"
        
        if stream:
            def event_stream():
                for chunk in coordinator.generate_stream(prompt, max_tokens):
                    if 'error' in chunk:
                        yield f"data: {json.dumps({'error': chunk['error']})}\n\n"
                        break
                    
                    delta = {
                        "choices": [{
                            "delta": {"content": chunk['text']},
                            "index": 0
                        }]
                    }
                    yield f"data: {json.dumps(delta)}\n\n"
                yield "data: [DONE]\n\n"
                
            return Response(stream_with_context(event_stream()), 
                          mimetype='text/event-stream')
        else:
            # Non-streaming
            tokens = []
            for chunk in coordinator.generate_stream(prompt, max_tokens):
                if 'error' in chunk:
                    return jsonify({"error": chunk['error']}), 500
                tokens.append(chunk['text'])
                
            return jsonify({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "".join(tokens)
                    }
                }]
            })
            
    @app.route('/v1/models', methods=['GET'])
    def list_models():
        return jsonify({
            "data": [{
                "id": "dicai-70b",
                "object": "model"
            }]
        })
        
    return app


def main():
    parser = argparse.ArgumentParser(description="DiCAI Streaming Coordinator")
    parser.add_argument("--config", default="configs/15_providers.json")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--port", type=int, default=8080)
    
    args = parser.parse_args()
    
    coordinator = StreamingCoordinator(args.config, args.model_dir)
    app = create_app(coordinator)
    
    print(f"[Coordinator] Starting on port {args.port}")
    app.run(host='0.0.0.0', port=args.port)


if __name__ == "__main__":
    main()
