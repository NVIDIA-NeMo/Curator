#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Buffered round-robin HTTP proxy for nonstreaming local replica benchmarks."""

from __future__ import annotations

import argparse
import itertools

from aiohttp import ClientError, ClientSession, ClientTimeout, TCPConnector, web

_HOP_HEADERS = {"host", "content-length", "connection", "transfer-encoding", "keep-alive", "te", "trailer", "upgrade"}


def create_app(upstreams: list[str], *, request_timeout_s: float = 300) -> web.Application:
    """Route each request once; the caller owns retries and their accounting."""
    if not upstreams or request_timeout_s <= 0:
        msg = "At least one upstream and a positive timeout are required"
        raise ValueError(msg)
    upstreams = [value.rstrip("/") for value in upstreams]
    choices = itertools.cycle(upstreams)
    session_key = web.AppKey("session", ClientSession)

    async def startup(app: web.Application) -> None:
        app[session_key] = ClientSession(
            connector=TCPConnector(limit=0), timeout=ClientTimeout(total=request_timeout_s), auto_decompress=False
        )

    async def cleanup(app: web.Application) -> None:
        await app[session_key].close()

    async def proxy(request: web.Request) -> web.Response:
        body = await request.read()
        try:
            async with request.app[session_key].request(
                request.method,
                f"{next(choices)}{request.rel_url}",
                data=body,
                headers={key: value for key, value in request.headers.items() if key.lower() not in _HOP_HEADERS},
                allow_redirects=False,
            ) as response:
                payload = await response.read()
                return web.Response(
                    body=payload,
                    status=response.status,
                    headers={key: value for key, value in response.headers.items() if key.lower() not in _HOP_HEADERS},
                )
        except TimeoutError as exc:
            raise web.HTTPGatewayTimeout(text="upstream request timed out") from exc
        except ClientError as exc:
            raise web.HTTPBadGateway(text=f"upstream request failed: {type(exc).__name__}") from exc

    app = web.Application(client_max_size=128 * 1024**2, handler_args={"handler_cancellation": True})
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route("*", "/{path:.*}", proxy)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--request-timeout", type=float, default=300)
    args = parser.parse_args()
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    web.run_app(
        create_app(args.upstream, request_timeout_s=args.request_timeout),
        host=args.listen_host,
        port=args.listen_port,
        access_log=None,
    )


if __name__ == "__main__":
    main()
