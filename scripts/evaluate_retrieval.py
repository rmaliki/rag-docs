"""Offline retrieval evaluation: recall@k on relevant documents, plus an access-control leak check.

Retrieval is evaluated separately from generation: if the right evidence is never
retrieved, no prompt change will fix the answer.

Usage: python -m scripts.evaluate_retrieval [eval/retrieval_eval.jsonl]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.auth.entra import AuthenticatedUser
from app.auth.permissions import resolve_permissions
from app.config import get_settings
from app.database.session import Database
from app.ingestion.embed import SentenceTransformerEmbedder
from app.monitoring.logging import configure_logging
from app.retrieval.hybrid_search import HybridRetriever
from app.retrieval.reranker import create_reranker
from scripts.create_demo_users import DEMO_USERS


async def evaluate(dataset: Path) -> int:
    settings = get_settings()
    configure_logging("WARNING")
    database = Database(settings)
    retriever = HybridRetriever(
        database,
        SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dimension),
        create_reranker(settings),
        top_k=settings.retrieval_top_k,
        candidates=settings.retrieval_candidates,
        rrf_k=settings.rrf_k,
        max_vector_distance=settings.max_vector_distance,
    )
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    recalls: list[float] = []
    leaks = 0
    try:
        for case in cases:
            user = DEMO_USERS[case["user"]]
            permissions = resolve_permissions(
                AuthenticatedUser(object_id=user["object_id"], groups=tuple(user["groups"]))
            )
            retrieved = {chunk.document_id for chunk in await retriever.retrieve(case["question"], permissions)}
            relevant = set(case.get("relevant", []))
            leaked = retrieved & set(case.get("forbidden", []))
            leaks += len(leaked)
            recall = len(retrieved & relevant) / len(relevant) if relevant else None
            if recall is not None:
                recalls.append(recall)
            verdict = "LEAK " + ",".join(sorted(leaked)) if leaked else "ok"
            shown = f"{recall:.2f}" if recall is not None else " n/a"
            print(f"[{case['user']:>9}] recall={shown}  {verdict:<6} {case['question']}")
    finally:
        await database.dispose()

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"\nmean recall@{settings.retrieval_top_k}: {mean_recall:.2f} over {len(recalls)} questions")
    print(f"unauthorized documents retrieved: {leaks}")
    return 1 if leaks else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("eval/retrieval_eval.jsonl"))
    args = parser.parse_args()
    sys.exit(asyncio.run(evaluate(args.dataset)))


if __name__ == "__main__":
    main()
