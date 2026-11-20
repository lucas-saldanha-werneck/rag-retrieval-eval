"""Tokenizer, two retrievers (BM25 and TF-IDF cosine), and metrics.

Pure standard library. No network, no model downloads.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

STOPWORDS = frozenset(
    "a o as os um uma de do da dos das em no na nos nas por para com sem e ou que "
    "se ao aos é são foi ser como qual quais quando onde mais já não meu minha".split()
)


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def tokenize(text: str) -> list[str]:
    text = strip_accents(text.lower())
    stop = {strip_accents(w) for w in STOPWORDS}
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in stop]


def stem(token: str) -> str:
    """Very light suffix stripping (Portuguese plurals/common endings)."""
    for suffix in ("coes", "oes", "mente", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def terms(text: str, use_stem: bool = False) -> list[str]:
    toks = tokenize(text)
    return [stem(t) for t in toks] if use_stem else toks


@dataclass
class Corpus:
    ids: list[str]
    texts: list[str]


class BM25:
    def __init__(self, corpus: Corpus, k1: float = 1.5, b: float = 0.75):
        self.corpus, self.k1, self.b = corpus, k1, b
        self.docs = [Counter(terms(t)) for t in corpus.texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = terms(query)
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1))
                    )
            out.append(s)
        return out


class TfidfCosine:
    """Bag-of-words TF-IDF vectors with light stemming, ranked by cosine.

    This is NOT a neural embedding. It is the cheapest stand-in that lets the
    same harness compare "another way to retrieve" against BM25.
    """

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        docs = [Counter(terms(t, use_stem=True)) for t in corpus.texts]
        df: Counter[str] = Counter()
        for d in docs:
            df.update(d.keys())
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + f)) + 1 for t, f in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, counts: Counter) -> dict[str, float]:
        v = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in counts.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    def scores(self, query: str) -> list[float]:
        q = self._vec(Counter(terms(query, use_stem=True)))
        return [sum(w * v.get(t, 0.0) for t, w in q.items()) for v in self.vecs]


def rank(scores: list[float]) -> list[int]:
    """Indices by score, best first. Ties keep corpus order (stable).

    Documents with a score of zero are dropped: "no match" must not count as
    a retrieved result just because it sits first in the corpus.
    """
    return sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion of several rankings."""
    total: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r):
            total[i] = total.get(i, 0.0) + 1.0 / (k + pos + 1)
    return sorted(total, key=lambda i: -total[i])


def est_tokens(text: str) -> int:
    """Rough token estimate (about 1.3 tokens per word). An estimate, not a tokenizer."""
    return round(len(text.split()) * 1.3)


def evaluate(data: dict, k: int = 3) -> dict:
    corpus = Corpus([d["id"] for d in data["docs"]], [d["text"] for d in data["docs"]])
    bm25, tfidf = BM25(corpus), TfidfCosine(corpus)
    methods = {
        "bm25": lambda q: rank(bm25.scores(q)),
        "tfidf": lambda q: rank(tfidf.scores(q)),
        "rrf": lambda q: rrf([rank(bm25.scores(q)), rank(tfidf.scores(q))]),
    }
    res = {}
    for name, fn in methods.items():
        hits, rr, ctx, misses = 0, 0.0, 0, []
        for item in data["questions"]:
            order = fn(item["q"])
            ids = [corpus.ids[i] for i in order]
            top = ids[:k]
            ctx += sum(est_tokens(corpus.texts[i]) for i in order[:k])
            if item["answer"] in top:
                hits += 1
            else:
                misses.append(item["q"])
            if item["answer"] in ids:
                rr += 1.0 / (ids.index(item["answer"]) + 1)
        n = len(data["questions"])
        res[name] = {
            "hit_at_k": hits / n,
            "mrr": rr / n,
            "avg_context_tokens": ctx / n,
            "misses": misses,
        }
    return {"k": k, "n_questions": len(data["questions"]), "methods": res}


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    ids = {d["id"] for d in data["docs"]}
    if len(ids) != len(data["docs"]):
        raise ValueError("duplicate doc ids")
    for q in data["questions"]:
        if q["answer"] not in ids:
            raise ValueError(f"answer {q['answer']!r} is not a doc id")
    return data


def to_markdown(result: dict) -> str:
    k = result["k"]
    lines = [
        f"| method | hit@{k} | MRR | avg context tokens (est.) |",
        "|---|---|---|---|",
    ]
    for name, m in result["methods"].items():
        lines.append(
            f"| {name} | {m['hit_at_k']:.2f} | {m['mrr']:.2f} | {m['avg_context_tokens']:.0f} |"
        )
    lines.append("")
    lines.append(f"{result['n_questions']} questions.")
    for name, m in result["methods"].items():
        if m["misses"]:
            lines.append(f"\nMissed by {name} (not in top {k}):")
            lines += [f"- {q}" for q in m["misses"]]
    return "\n".join(lines) + "\n"
