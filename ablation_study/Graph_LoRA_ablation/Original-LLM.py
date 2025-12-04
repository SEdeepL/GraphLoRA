import json
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEvaluator:
    def __init__(self, model_name: str, model_path: str):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model '{self.model_name}' from local path: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if DEVICE == "cpu":
            self.model.to(DEVICE)

    def _build_prompt(self, patch: str) -> str:

        prompt = (
            "You are a code review assistant.\n\n"
            "Task: Assess whether the patch is correct.\n"
            "You are given a code patch.\n\n"
            "Please decide whether the patch is a truly correct fix or an overfitting patch that only\n\n"
            "Output format:\n"
            "- Return 0 if the patch is correct and does not overfit the tests.\n"
            "- Return 1 if the patch overfits the tests.\n"
            "Answer with a single digit only: 0 or 1.\n"
        )
        return prompt

    def _extract_label(self, text: str) -> int:
        for ch in text:
            if ch == "0":
                return 0
            if ch == "1":
                return 1
        raise ValueError(f"Cannot extract label from model output: {text!r}")

    def predict(
        self,
        patch: str,
        max_new_tokens: int = 8,
        temperature: float = 0.0,
    ) -> int:
        prompt = self._build_prompt(patch)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        label = self._extract_label(text)
        print(f"[{self.model_name}] raw output: {text!r} -> label: {label}")
        return label


def evaluate_file(
    input_path: str,
    output_path: str,
    model_name: str,
    model_path: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
):
    """
    读取 input.jsonl，逐行用单个模型预测并写入 output.jsonl，同时计算指标。
    input.jsonl 每行: {"patch": "...", "label": 0 或 1}
    output.jsonl 每行: {"patch": "...", "label": 0, "pred": 1, "model_name": "..."}
    """
    evaluator = PatchEvaluator(model_name=model_name, model_path=model_path)

    tp = fp = tn = fn = 0
    total = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Line {line_idx}] JSON decode error: {e}")
                continue

            if "patch" not in record or "label" not in record:
                print(f"[Line {line_idx}] Missing 'patch' or 'label' keys, skip.")
                continue

            patch = record["patch"]
            try:
                gold = int(record["label"])
            except Exception:
                print(f"[Line {line_idx}] Invalid label value: {record['label']}, skip.")
                continue

            try:
                pred = evaluator.predict(
                    patch,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"[Line {line_idx}] Prediction error: {e}, skip for metrics.")
                pred = None

            if pred is None:
                skipped += 1
            else:
                total += 1
                # label: 1 = overfitting 作为正类
                if gold == 1 and pred == 1:
                    tp += 1
                elif gold == 0 and pred == 1:
                    fp += 1
                elif gold == 0 and pred == 0:
                    tn += 1
                elif gold == 1 and pred == 0:
                    fn += 1

            out_rec = {
                "patch": patch,
                "label": gold,
                "pred": pred,
                "model_name": model_name,
            }

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")

    if total == 0:
        print("No valid predictions to evaluate.")
        return

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    print("\n===== Evaluation results =====")
    print(f"Model name: {model_name}")
    print(f"Total evaluated samples: {total}")
    print(f"Skipped (prediction error): {skipped}")
    print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (positive class = label 1 = overfitting)")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 score : {f1:.4f}")
    print("================================")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate patch correctness with a single local LLM.")
    parser.add_argument("--input", type=str, required=True, help="Path to input jsonl file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output jsonl file.")

    parser.add_argument("--model_name", type=str, required=True, help="Model name tag (for logging/output).")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to model.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_file(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
