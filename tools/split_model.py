import os
import json
import hashlib
import torch
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import llama_cpp
    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False
    print("[ERROR] llama-cpp-python not installed.")


def split_gguf_model(model_path: str, output_dir: str, n_layers: int = 22):
    """Split a GGUF model into per-layer binary chunks."""
    print(f"[Splitter] Loading model: {model_path}")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return False

    if not HAS_LLAMA_CPP:
        print("[ERROR] llama-cpp-python required to read GGUF metadata")
        return False

    # Load model with llama.cpp to get metadata
    model = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=512,
        verbose=False
    )

    n_vocab = model.n_vocab()
    n_ctx = model.n_ctx()

    n_embd = 2048
    n_head = 32
    n_kv_head = 4

    print(f"[Splitter] Model info: layers={n_layers}, embd={n_embd}, heads={n_head}")

    os.makedirs(output_dir, exist_ok=True)

    total_size = os.path.getsize(model_path)
    chunk_size = total_size // n_layers

    manifest = {
        "model_name": Path(model_path).stem,
        "original_path": os.path.abspath(model_path),
        "n_layers": n_layers,
        "n_embd": n_embd,
        "n_vocab": n_vocab,
        "n_ctx": n_ctx,
        "n_head": n_head,
        "n_kv_head": n_kv_head,
        "total_size": total_size,
        "format": "gguf_chunks",
        "layers": [],
        "created_at": ""
    }

    with open(model_path, 'rb') as f:
        for layer_idx in range(n_layers):
            layer_file = os.path.join(output_dir, f"layer_{layer_idx:03d}.pt")

            start_byte = layer_idx * chunk_size
            end_byte = start_byte + chunk_size
            if layer_idx == n_layers - 1:
                end_byte = total_size

            f.seek(start_byte)
            chunk_data = f.read(end_byte - start_byte)

            layer_data = {
                "layer_idx": layer_idx,
                "start_byte": start_byte,
                "end_byte": end_byte,
                "total_size": total_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_kv_head": n_kv_head,
                "layer_type": "gguf_chunk",
                "chunk": chunk_data
            }

            torch.save(layer_data, layer_file)

            with open(layer_file, 'rb') as f_check:
                checksum = hashlib.sha256(f_check.read()).hexdigest()[:16]

            manifest["layers"].append({
                "index": layer_idx,
                "file": f"layer_{layer_idx:03d}.pt",
                "checksum": checksum,
                "size_mb": os.path.getsize(layer_file) / (1024 * 1024),
                "start_byte": start_byte,
                "end_byte": end_byte
            })

            print(f"  Layer {layer_idx}: bytes {start_byte}-{end_byte}, {os.path.getsize(layer_file)/(1024*1024):.1f} MB")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"[Splitter] Done. Output: {output_dir}")
    print(f"[Splitter] Manifest: {manifest_path}")
    print(f"[Splitter] Total layers: {n_layers}")
    print(f"[Splitter] Chunk size: ~{chunk_size/(1024*1024):.1f} MB each")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Split a GGUF model into per-layer binary chunks")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--output", default="layers", help="Output directory for layer files")
    parser.add_argument("--n-layers", type=int, default=22, help="Number of layers")
    args = parser.parse_args()

    success = split_gguf_model(args.model, args.output, args.n_layers)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
