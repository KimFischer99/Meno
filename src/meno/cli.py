from __future__ import annotations

import argparse

import uvicorn

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="meno")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Run the Meno HTTP sidecar")
    subparsers.add_parser("process-once", help="Process pending outbox events once")
    rebuild = subparsers.add_parser(
        "rebuild-projection",
        help="Replay active non-sensitive claims into the configured vector collection",
    )
    rebuild.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.command == "serve":
        uvicorn.run(
            "meno.api:create_app",
            host=settings.api_host,
            port=settings.api_port,
            workers=1,
            factory=True,
        )
        return
    from .api import build_service

    service = build_service(settings)
    try:
        if args.command == "process-once":
            print(service.process_outbox())
            return
        print(service.rebuild_projection(args.batch_size))
    finally:
        service.close()


if __name__ == "__main__":
    main()
