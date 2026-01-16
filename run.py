#!/usr/bin/env python3
"""
AI Tutor Application Entry Point

This script starts the AI Tutor server using uvicorn.

Usage:
    python run.py

Or with custom host/port:
    python run.py --host 0.0.0.0 --port 8000

For production deployment, use gunicorn or similar:
    gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config

# Configure basic logging before importing app
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="AI Tutor Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--host",
        type=str,
        default=Config.HOST,
        help="Host to bind to"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=Config.PORT,
        help="Port to bind to"
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (not recommended with --reload)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default=Config.LOG_LEVEL.lower(),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the application.
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Starting AI Tutor Server...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)

    try:
        import uvicorn

        # Configure uvicorn
        config = uvicorn.Config(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True
        )

        # Create and run server
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
