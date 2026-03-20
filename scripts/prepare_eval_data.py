from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare public eval slices for AIMO3 experiments")
    parser.add_argument("--dataset", default="nvidia/OpenMathReasoning")
    parser.add_argument("--split", default="train[:100]")
    parser.add_argument("--output", default="eval/eval_small.jsonl")
    parser.add_argument("--hard-output", default="eval/eval_hard.jsonl")
    parser.add_argument("--hard-count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dataset support with `pip install -e .[runtime]`.") from exc

    args = parse_args()
    output_path = Path(args.output)
    hard_output_path = Path(args.hard_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    rows = []
    for item in dataset:
        problem = item.get("problem") or item.get("question") or item.get("prompt") or item.get("input")
        answer = item.get("answer") or item.get("solution") or item.get("target")
        rows.append({"problem": problem, "answer": answer})

    output_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    hard_rows = rows[-args.hard_count :]
    hard_output_path.write_text("\n".join(json.dumps(row) for row in hard_rows), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {output_path} and {len(hard_rows)} rows to {hard_output_path}")


if __name__ == "__main__":
    main()
