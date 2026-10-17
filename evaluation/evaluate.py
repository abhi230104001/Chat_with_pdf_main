"""
evaluation/evaluate.py — RAG pipeline evaluation script.

Metrics computed:
  - Retrieval: documents_retrieved / expected_in_document questions
  - Citation presence rate: answers that include at least one citation
  - Citation validity: citations that reference retrieved chunk metadata
  - Refusal accuracy: refusals on questions not expected in documents
  - Average retrieval latency
  - Average total latency

NOTE: These are AUTOMATIC metrics. Answer quality (faithfulness, correctness)
requires human judgement and is NOT measured here. Do not conflate automatic
scores with human-judged answer quality.

Usage:
    # Make sure the server is running first:
    python run.py

    # Then in another terminal:
    python evaluation/evaluate.py --base-url http://localhost:8000 --dataset evaluation/dataset.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:
    print("httpx is required for evaluation: pip install httpx")
    sys.exit(1)


# ── Refusal detection heuristics ─────────────────────────────────────────────

REFUSAL_PHRASES = [
    "cannot find",
    "not found in",
    "does not contain",
    "not present in",
    "unable to find",
    "no information",
    "not mentioned",
    "not available in",
    "outside the scope",
    "cannot answer",
]


def is_refusal(answer: str) -> bool:
    """Heuristically detect whether the model refused to answer."""
    a = answer.lower()
    return any(phrase in a for phrase in REFUSAL_PHRASES)


# ── Metrics ───────────────────────────────────────────────────────────────────

def run_evaluation(
    base_url: str,
    dataset: List[Dict[str, Any]],
    session_id: str = "eval_session",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run the evaluation dataset against the live RAG pipeline.

    Returns a dict of aggregated metrics.
    """
    client = httpx.Client(base_url=base_url, timeout=60.0)

    results = []
    total = len(dataset)

    print(f"\nRunning evaluation on {total} questions…\n{'─' * 50}")

    for i, item in enumerate(dataset, start=1):
        qid = item.get("id", f"q{i:03d}")
        question = item["question"]
        expected_in_doc = item.get("expected_in_document", True)
        expected_refusal = item.get("expected_refusal", False)

        print(f"[{i}/{total}] {qid}: {question[:60]}…" if len(question) > 60 else f"[{i}/{total}] {qid}: {question}")

        t_start = time.perf_counter()
        try:
            res = client.post(
                "/api/chat",
                json={"question": question, "session_id": session_id},
            )
            res.raise_for_status()
            data = res.json()
        except Exception as exc:
            print(f"  ✕ Request failed: {exc}")
            results.append({
                "id": qid, "error": str(exc),
                "retrieved": False, "citation_present": False,
                "citation_valid": False, "refusal_correct": None,
                "retrieval_ms": 0, "total_ms": 0,
            })
            continue

        total_ms = (time.perf_counter() - t_start) * 1000
        answer = data.get("answer", "")
        sources = data.get("sources", [])
        debug = data.get("debug", {})

        # Metric: retrieval (did we get any chunks?)
        num_chunks = debug.get("num_chunks_selected", 0)
        retrieved = num_chunks > 0

        # Metric: citation presence
        citation_present = len(sources) > 0

        # Metric: citation validity (all sources have document_id + filename)
        citation_valid = all(
            s.get("document_id") and s.get("filename")
            for s in sources
        ) if sources else (not citation_present)

        # Metric: refusal accuracy
        refused = is_refusal(answer)
        if expected_refusal:
            refusal_correct = refused
        elif not expected_in_doc:
            refusal_correct = refused  # should have refused
        else:
            refusal_correct = not refused  # should have answered

        retrieval_ms = debug.get("retrieval_latency_ms", 0)

        status = "✓" if (retrieved or expected_refusal) else "○"
        print(f"  {status} chunks={num_chunks}, sources={len(sources)}, refusal={refused}, total={total_ms:.0f}ms")

        if verbose:
            print(f"     Answer: {answer[:120]}…" if len(answer) > 120 else f"     Answer: {answer}")

        results.append({
            "id": qid,
            "question": question,
            "answer_preview": answer[:200],
            "retrieved": retrieved,
            "num_chunks": num_chunks,
            "citation_present": citation_present,
            "citation_valid": citation_valid,
            "expected_in_document": expected_in_doc,
            "expected_refusal": expected_refusal,
            "refused": refused,
            "refusal_correct": refusal_correct,
            "retrieval_ms": retrieval_ms,
            "total_ms": total_ms,
        })

        time.sleep(0.5)  # avoid rate limiting

    # ── Aggregate Metrics ────────────────────────────────────────────────────

    n = len(results)
    successful = [r for r in results if "error" not in r]
    n_ok = len(successful)

    retrieval_recall = sum(1 for r in successful if r["retrieved"]) / max(n_ok, 1)
    citation_presence = sum(1 for r in successful if r["citation_present"]) / max(n_ok, 1)
    citation_validity = sum(1 for r in successful if r["citation_valid"]) / max(n_ok, 1)

    refusal_items = [r for r in successful if r.get("expected_refusal")]
    refusal_accuracy = (
        sum(1 for r in refusal_items if r["refusal_correct"]) / max(len(refusal_items), 1)
        if refusal_items else None
    )

    avg_retrieval_ms = sum(r["retrieval_ms"] for r in successful) / max(n_ok, 1)
    avg_total_ms     = sum(r["total_ms"]     for r in successful) / max(n_ok, 1)

    metrics = {
        "total_questions": total,
        "successful_requests": n_ok,
        "failed_requests": n - n_ok,
        "retrieval_recall": round(retrieval_recall, 3),
        "citation_presence_rate": round(citation_presence, 3),
        "citation_validity_rate": round(citation_validity, 3),
        "refusal_accuracy": round(refusal_accuracy, 3) if refusal_accuracy is not None else None,
        "avg_retrieval_latency_ms": round(avg_retrieval_ms, 1),
        "avg_total_latency_ms": round(avg_total_ms, 1),
        "per_question": results,
    }

    return metrics


def print_report(metrics: Dict[str, Any]) -> None:
    """Print a human-readable evaluation report."""
    print(f"\n{'═' * 50}")
    print("  EVALUATION REPORT")
    print(f"{'═' * 50}")
    print(f"  Questions total:        {metrics['total_questions']}")
    print(f"  Successful requests:    {metrics['successful_requests']}")
    print(f"  Failed requests:        {metrics['failed_requests']}")
    print()
    print(f"  Retrieval recall:       {metrics['retrieval_recall']:.1%}")
    print(f"  Citation presence rate: {metrics['citation_presence_rate']:.1%}")
    print(f"  Citation validity rate: {metrics['citation_validity_rate']:.1%}")
    if metrics['refusal_accuracy'] is not None:
        print(f"  Refusal accuracy:       {metrics['refusal_accuracy']:.1%}")
    print()
    print(f"  Avg retrieval latency:  {metrics['avg_retrieval_latency_ms']:.1f} ms")
    print(f"  Avg total latency:      {metrics['avg_total_latency_ms']:.1f} ms")
    print(f"{'═' * 50}")
    print()
    print("NOTE: These are automatic proxy metrics. They do NOT measure answer")
    print("faithfulness, factual accuracy, or human-judged quality. Use these")
    print("metrics as indicators, not as absolute ground truth benchmarks.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Chat with PDF RAG pipeline."
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--dataset", default="evaluation/dataset.json", help="Path to dataset JSON")
    parser.add_argument("--output", default=None, help="Optional path to save results JSON")
    parser.add_argument("--verbose", action="store_true", help="Print answer previews")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Base URL: {args.base_url}")
    print(f"Dataset:  {dataset_path} ({len(dataset)} questions)")

    # Verify server is reachable
    try:
        res = httpx.get(f"{args.base_url}/health", timeout=5.0)
        health = res.json()
        print(f"Server:   {health.get('status', 'unknown')}")
    except Exception as exc:
        print(f"✕ Cannot reach server at {args.base_url}: {exc}")
        print("  Start the server with: python run.py")
        sys.exit(1)

    metrics = run_evaluation(
        base_url=args.base_url,
        dataset=dataset,
        verbose=args.verbose,
    )

    print_report(metrics)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
