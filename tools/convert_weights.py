"""Convert pytorch_model.bin → fp16 safetensors (written by hand, no safetensors
lib) + meta.json. Output goes to ../weights/ (the files the package ships)."""
import json
import os
import struct
import torch

OUT_DIR = "../weights"
os.makedirs(OUT_DIR, exist_ok=True)

sd = torch.load("pytorch_model.bin", map_location="cpu")
cfg = json.load(open("config.json"))

SKIP = ("attn.attention.bias", "attn.attention.masked_bias")  # causal-mask buffers, recomputed in TS

tensors = {}
for name, t in sd.items():
    if any(name.endswith(s) for s in SKIP):
        continue
    tensors[name] = t.detach().half().contiguous()

header = {}
offset = 0
for name, t in tensors.items():
    n = t.numel() * 2  # fp16
    header[name] = {"dtype": "F16", "shape": list(t.shape), "data_offsets": [offset, offset + n]}
    offset += n

header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
path = os.path.join(OUT_DIR, "tinystories-1m.safetensors")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    for t in tensors.values():
        f.write(t.numpy().tobytes())

meta = {
    "model": "roneneldan/TinyStories-1M",
    "arch": "gpt_neo",
    "hidden": cfg["hidden_size"],
    "layers": cfg["num_layers"],
    "heads": cfg["num_heads"],
    "vocab": cfg["vocab_size"],
    "maxPos": cfg["max_position_embeddings"],
    "lnEps": cfg["layer_norm_epsilon"],
    "activation": cfg["activation_function"],
    "license": "roneneldan/TinyStories (research release)",
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"tensors: {len(tensors)}, file: {os.path.getsize(path) / 1e6:.1f} MB")
