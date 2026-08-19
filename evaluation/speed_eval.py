import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).parent
ROOT_DIR = EVAL_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(EVAL_DIR))

from plot_results import plot_benchmark

MODELS = [
    ("Dense", ROOT_DIR / "runs" / "dense_checkpoint" / "final", EVAL_DIR / "dense_speed_results.json"),
    ("MoE", ROOT_DIR / "runs" / "moe_checkpoint" / "final", EVAL_DIR / "moe_speed_results.json"),
]

PROMPT_LEN = 32
GEN_LEN = 64
WARMUP_STEPS = 2
TRIALS = 5


def load_model(model_path, device):
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.modules import LLMConfig, LLM
    AutoConfig.register("custom_llm", LLMConfig)
    AutoModelForCausalLM.register(LLMConfig, LLM)

    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_generation(model, input_ids):
    ids = input_ids.clone()
    for _ in range(GEN_LEN):
        logits = model(ids).logits
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def measure_generation_speed(model, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (1, PROMPT_LEN), device=device)

    for _ in range(WARMUP_STEPS):
        run_generation(model, input_ids)

    times = []
    for _ in range(TRIALS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_generation(model, input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    times = np.array(times)
    tokens_per_sec = GEN_LEN / times
    mean = float(tokens_per_sec.mean())
    stderr = float(tokens_per_sec.std(ddof=1) / np.sqrt(len(tokens_per_sec)))
    return mean, stderr


def measure_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    vocab_size = model.config.vocab_size

    print("Measuring generation speed...")
    mean, stderr = measure_generation_speed(model, vocab_size, device)
    print(f"tokens/sec: {mean:.2f} +/- {stderr:.2f}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "name": "speed",
        "alias": "speed",
        "tokens_per_sec,none": mean,
        "tokens_per_sec_stderr,none": stderr,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    for name, model_path, out_json in MODELS:
        data = measure_model(model_path, device)
        results[name] = data
        with open(out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"saved {out_json}")

    plot_benchmark("speed", results["Dense"], results["MoE"])


if __name__ == "__main__":
    main()