"""Start a local development Meno server with the deterministic test embedder.

Production Meno supports Google cloud embeddings only; this runner exists so
benchmarks can exercise the full API locally without an embedding provider.
It reuses ``tests.fakes.TestEmbedder`` (the same deterministic hashing double
used by the unit suite) via the public ``create_app(embedder=...)`` injection
point, with SQLite as the canonical store and the in-memory vector store.

Numbers from this server are comparable across Meno revisions and to previous
local deterministic-embedder runs, but NOT to the production Google-embedding
deployment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from meno.api import create_app
from meno.config import Settings
from tests.fakes import TestEmbedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        default="sqlite:///./artifacts/runtime/meno-dev.db",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--api-token", default="")
    args = parser.parse_args()
    settings = Settings(
        database_url=args.database_url,
        vector_mode="memory",
        api_port=args.port,
        api_token=args.api_token,
    )
    settings.validate()
    app = create_app(settings, embedder=TestEmbedder(dimension=args.dimension))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
