import argparse, json, sys
from pathlib import Path
from apsg.ablation_structure_only import ablate_structure_only_graph_dict

def main():
    ap = argparse.ArgumentParser(description="structure_only")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--skip-errors", action="store_true")
    args = ap.parse_args()
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
                g = obj.get("graph")
                if not isinstance(g, dict):
                    raise ValueError("graph not dict")
                obj["graph"] = ablate_structure_only_graph_dict(g)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as e:
                if args.skip_errors:
                    print(f"[ERROR] line {n}: {e}", file=sys.stderr)
                    continue
                else:
                    raise
    print(f"Done. Wrote {ok} records to {out_path} (processed {n} lines).")

if __name__ == "__main__":
    main()
