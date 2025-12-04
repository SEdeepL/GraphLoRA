#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from apsg import build_apsg, LlamaEntropy, StubEntropyProvider

def maybe_wrap_method(code: str) -> str:
    s = code.strip()
    if "{" in s and "}" in s and "(" in s and ")" in s:
        return s
    return "void f(){ " + s + " }"

def main():
    ap = argparse.ArgumentParser(description="Build APSG JSONL from a JSONL input (each line has {code,label}).")
    ap.add_argument("--input", required=True, help="Input JSONL file, lines like {\"code\":..., \"label\":...}")
    ap.add_argument("--output", required=True, help="Output JSONL; each line has {code, graph, label}")
    ap.add_argument("--model-path", default=None, help="Local path to Llama3 (e.g., /home/sdu/yangzhenyu/LLM/llama3_1/)")
    ap.add_argument("--device", default="auto", help="Transformers device map (auto/cpu/cuda)")
    ap.add_argument("--dtype", default=None, choices=["float16","bfloat16","float32"], help="Torch dtype")
    ap.add_argument("--skip-errors", action="store_true", help="Skip lines that fail to parse/build")
    args = ap.parse_args()

    ent = None
    if args.model_path:
        try:
            ent = LlamaEntropy(args.model_path, device=args.device, dtype=args.dtype)
        except Exception as e:
            print(f"[WARN] Failed to load Llama3 at {args.model_path}: {e}. Using StubEntropyProvider.", file=sys.stderr)
            ent = StubEntropyProvider()
    else:
        ent = StubEntropyProvider()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    ok = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n += 1
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                code = obj.get("code", "")
                label = obj.get("label")
            except Exception as e:
                if args.skip_errors:
                    print(f"[ERROR] line {n}: invalid JSON - {e}", file=sys.stderr)
                    continue
                else:
                    raise
            if not code.strip():
                if args.skip_errors:
                    print(f"[ERROR] line {n}: empty code", file=sys.stderr)
                    continue
                else:
                    raise ValueError(f"Empty code at line {n}")

            code2 = maybe_wrap_method(code)
            try:
                g = build_apsg(code2, entropy_provider=ent)
                rec = {"code": code, "graph": g.to_dict(), "label": label}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                msg = f"[ERROR] line {n}: {e}"
                if args.skip_errors:
                    print(msg, file=sys.stderr)
                    continue
                else:
                    raise
    print(f"Done. Wrote {ok} records to {out_path} (processed {n} lines).")

if __name__ == "__main__":
    main()
