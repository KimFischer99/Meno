from __future__ import annotations

import argparse

import uvicorn

from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="meno")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Run the Meno HTTP sidecar")
    subparsers.add_parser("process-once", help="Process pending outbox events once")
    requeue = subparsers.add_parser(
        "requeue-failed",
        help="Move failed outbox rows back to pending for another processing attempt",
    )
    requeue.add_argument("--limit", type=int, default=None)
    rebuild = subparsers.add_parser(
        "rebuild-projection",
        help="Replay active non-sensitive claims into the configured vector collection",
    )
    rebuild.add_argument("--batch-size", type=int, default=None)
    migrate = subparsers.add_parser(
        "migrate",
        help="Apply the claim coordination schema migration (idempotent)",
    )
    migrate.add_argument(
        "--revert-routing",
        action="store_true",
        help="Clear Phase B routing provenance without dropping columns",
    )
    reprocess = subparsers.add_parser(
        "reprocess",
        help="Enqueue replay of existing events under an extractor version",
    )
    reprocess.add_argument(
        "--extractor-version",
        default=None,
        help="version to replay under (default: MENO_EXTRACTOR_VERSION)",
    )
    reprocess.add_argument("--user-id", default=None)
    reprocess.add_argument("--limit", type=int, default=None)
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
        if args.command == "requeue-failed":
            print(service.requeue_failed_outbox(args.limit))
            return
        if args.command == "migrate":
            print(
                service.revert_routing_migration()
                if args.revert_routing
                else service.migrate()
            )
            return
        if args.command == "reprocess":
            print(
                service.reprocess(
                    args.extractor_version or settings.extractor_version,
                    user_id=args.user_id,
                    limit=args.limit,
                )
            )
            return
        print(service.rebuild_projection(args.batch_size))
    finally:
        service.close()


if __name__ == "__main__":
    main()
