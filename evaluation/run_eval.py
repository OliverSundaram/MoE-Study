import sys, json, argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from transformers import AutoConfig, AutoModelForCausalLM
    from models.modules import LLMConfig, LLM
    AutoConfig.register("custom_llm", LLMConfig)
    AutoModelForCausalLM.register(LLMConfig, LLM)
    import lm_eval

    device = "cuda" if torch.cuda.is_available() else "cpu"
    groups = {
        0: ["arc_easy", "piqa", "wikitext", "lambada_openai"],
        5: ["winogrande"],
        10: ["hellaswag"],
        25: ["arc_challenge"],
    }

    combined = {}
    for num_fewshot, tasks in groups.items():
        r = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.model_path},max_length=1024,dtype=float32",
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=1,
            device=device,
        )
        combined.update(r["results"])

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2)