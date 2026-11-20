import json
import subprocess
import sys
from pathlib import Path

import pytest

from retrieval_eval.core import (
    BM25, Corpus, TfidfCosine, est_tokens, evaluate, load, rank, rrf, stem,
    to_markdown, tokenize,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "example.json"


def small():
    return Corpus(["a", "b", "c"], ["gato preto no telhado", "cachorro corre no parque", "carro vermelho na rua"])


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("A Política de Férias é nova") == ["politica", "ferias", "nova"]


def test_stem_light():
    assert stem("notas") == "nota"
    assert stem("em") == "em"  # too short, untouched


def test_bm25_ranks_matching_doc_first():
    s = BM25(small()).scores("cachorro parque")
    assert rank(s)[0] == 1


def test_bm25_no_overlap_scores_zero():
    assert BM25(small()).scores("foguete") == [0.0, 0.0, 0.0]


def test_tfidf_uses_stemming():
    c = Corpus(["a", "b"], ["as notas fiscais", "o carro novo"])
    assert rank(TfidfCosine(c).scores("nota fiscal"))[0] == 0


def test_rank_ties_keep_corpus_order():
    assert rank([1.0, 1.0, 1.0]) == [0, 1, 2]


def test_rank_drops_zero_scores():
    assert rank([0.0, 2.0, 0.0]) == [1]


def test_rrf_prefers_doc_ranked_well_by_both():
    assert rrf([[2, 0, 1], [2, 1, 0]])[0] == 2


def test_est_tokens():
    assert est_tokens("um dois três quatro") == 5


def test_evaluate_perfect_on_trivial_set():
    data = {"docs": [{"id": "x", "text": "banana prata"}, {"id": "y", "text": "carro azul"}],
            "questions": [{"q": "banana", "answer": "x"}, {"q": "carro", "answer": "y"}]}
    r = evaluate(data, k=1)
    for m in r["methods"].values():
        assert m["hit_at_k"] == 1.0 and m["mrr"] == 1.0 and m["misses"] == []


def test_evaluate_reports_a_miss_with_no_shared_words():
    data = {"docs": [{"id": "x", "text": "banana prata"}, {"id": "y", "text": "carro azul"}],
            "questions": [{"q": "fruta amarela", "answer": "x"}]}
    r = evaluate(data, k=1)
    assert r["methods"]["bm25"]["misses"] == ["fruta amarela"]


def test_load_rejects_unknown_answer(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"docs": [{"id": "a", "text": "x"}], "questions": [{"q": "q", "answer": "zzz"}]}))
    with pytest.raises(ValueError):
        load(str(p))


def test_load_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"docs": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "questions": []}))
    with pytest.raises(ValueError):
        load(str(p))


def test_example_dataset_is_marked_as_example():
    assert "EXAMPLE" in json.loads(DATA.read_text())["note"]


def test_example_dataset_numbers_are_stable():
    r = evaluate(load(str(DATA)), k=3)
    assert r["n_questions"] == 16
    assert r["methods"]["bm25"]["hit_at_k"] == pytest.approx(13 / 16)
    assert len(r["methods"]["bm25"]["misses"]) == 3


def test_markdown_has_all_methods():
    md = to_markdown(evaluate(load(str(DATA)), k=3))
    for name in ("bm25", "tfidf", "rrf"):
        assert f"| {name} |" in md


def test_cli_runs_and_prints_json():
    out = subprocess.run([sys.executable, "-m", "retrieval_eval", str(DATA), "--json"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["k"] == 3


def test_example_misses_share_no_term_with_their_answer():
    """The three misses are paraphrases: no exact or stemmed term overlaps."""
    data = load(str(DATA))
    docs = {d["id"]: d["text"] for d in data["docs"]}
    answers = {q["q"]: q["answer"] for q in data["questions"]}
    for q in evaluate(data, k=3)["methods"]["bm25"]["misses"]:
        a = {stem(t) for t in tokenize(q)}
        b = {stem(t) for t in tokenize(docs[answers[q]])}
        assert not (a & b)
