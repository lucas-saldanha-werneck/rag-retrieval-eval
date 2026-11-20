import argparse
import json

from .core import evaluate, load, to_markdown


def main() -> None:
    p = argparse.ArgumentParser(prog="retrieval_eval", description=__doc__)
    p.add_argument("dataset", help="JSON file with docs and labeled questions")
    p.add_argument("-k", type=int, default=3, help="top-k for hit@k (default 3)")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = p.parse_args()
    result = evaluate(load(a.dataset), a.k)
    print(json.dumps(result, indent=2, ensure_ascii=False) if a.json else to_markdown(result))


main()
