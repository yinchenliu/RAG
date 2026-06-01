"""CLI: drive the agentic config loop on a single PDF.

    python -m RAG.pre_injection.configure <pdf_path> [--doc-id ID] [--pages 20] [--max-iter 5]

Writes the accepted config to RAG/pre_injection/configs/<doc_id>.yaml. On
failure, writes a draft to RAG/pre_injection/configs/<doc_id>.draft.yaml
(the `.draft.` infix keeps load_config from picking it up).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from RAG.pre_injection.configure_graph import build_graph
from RAG.rag import CLOIndentureRAG


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m RAG.pre_injection.configure")
    p.add_argument("pdf_path")
    p.add_argument("--doc-id", default=None,
                   help="Override; defaults to SHA1 hash of PDF.")
    p.add_argument("--pages", type=int, default=20,
                   help="Pages of front matter to send to the LLM.")
    p.add_argument("--max-iter", type=int, default=5,
                   help="Cap on propose/critique iterations.")
    args = p.parse_args()

    _setup_logging()

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        print(
            "ERROR: GOOGLE_API_KEY or GEMINI_API_KEY env var is required",
            file=sys.stderr,
        )
        return 2

    doc_id = args.doc_id or CLOIndentureRAG._hash_file(args.pdf_path)
    graph = build_graph()
    initial_state = {
        "pdf_path": args.pdf_path,
        "doc_id": doc_id,
        "max_iterations": args.max_iter,
        "front_matter_pages": args.pages,
    }

    print(f"==> configuring doc_id={doc_id!r} pdf={args.pdf_path}")
    final = graph.invoke(initial_state, config={"recursion_limit": 50})

    if final.get("accepted"):
        print(f"OK: config written to {final['final_config_path']}")
        print(
            f"  iterations: {final['iteration']}, "
            f"sections: {final['smoke_metrics']['section_count']}"
        )
        return 0
    print(
        f"FAILED to converge after {final.get('iteration', 0)} iterations.\n"
        f"  Draft written to {final['final_config_path']} for hand-fixup.\n"
        f"  Review the embedded history at the bottom of that file."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
