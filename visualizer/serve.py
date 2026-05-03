#!/usr/bin/env python3
"""Serve the local visualizer with the standard library."""

from __future__ import annotations

import http.server
import socketserver
from pathlib import Path


PORT = 8765
ROOT = Path(__file__).resolve().parents[1]


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args,
        directory=str(ROOT),
        **kwargs,
    )
    with ReusableTCPServer(("127.0.0.1", PORT), handler) as server:
        print(f"Serving LLM Study Visualizer at http://127.0.0.1:{PORT}/visualizer/")
        print("Press Ctrl-C to stop.")
        server.serve_forever()


if __name__ == "__main__":
    main()
