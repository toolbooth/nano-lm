"""Dump ground-truth reference outputs from the official GPT-Neo implementation.
The TS hand-rolled forward pass must reproduce these (within fp16 tolerance)."""
import json
import torch
from transformers import AutoTokenizer, GPTNeoForCausalLM

tok = AutoTokenizer.from_pretrained(".")
model = GPTNeoForCausalLM.from_pretrained(".")
model.eval()

PROMPT = "Once upon a time"
enc = tok(PROMPT, return_tensors="pt")
ids = enc.input_ids

with torch.no_grad():
    out = model(ids, output_attentions=True)
    logits = out.logits[0, -1]  # last position
    top = torch.topk(logits, 10)
    greedy = model.generate(ids, max_new_tokens=12, do_sample=False)

fixture = {
    "prompt": PROMPT,
    "input_ids": ids[0].tolist(),
    "top10": [{"id": int(i), "logit": float(v)} for v, i in zip(top.values, top.indices)],
    "logits_first8": [float(x) for x in logits[:8]],
    "logits_mean": float(logits.mean()),
    "logits_std": float(logits.std()),
    "attn_l0_h0": [[float(x) for x in row] for row in out.attentions[0][0, 0]],
    "attn_l7_h3": [[float(x) for x in row] for row in out.attentions[7][0, 3]],
    "greedy_ids": greedy[0].tolist(),
    "greedy_text": tok.decode(greedy[0]),
}
with open("reference.json", "w") as f:
    json.dump(fixture, f)
print("greedy:", fixture["greedy_text"])
print("top3:", [(t["id"], tok.decode([t["id"]]), round(t["logit"], 3)) for t in fixture["top10"][:3]])
print("wrote reference.json")
