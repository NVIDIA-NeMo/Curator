#!/usr/bin/env python3
"""Minimal OpenAI-compatible round-robin proxy for local replica benchmarks."""

from __future__ import annotations

import argparse
import itertools

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument("--upstream", action="append", required=True)
    args = parser.parse_args()

    upstreams = [value.rstrip("/") for value in args.upstream]
    choices = itertools.cycle(range(len(upstreams)))

    async def startup(app: web.Application) -> None:
        app["session"] = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=900))

    async def cleanup(app: web.Application) -> None:
        await app["session"].close()

    async def proxy(request: web.Request) -> web.Response:
        body = await request.read()
        first = next(choices)
        last_error: Exception | None = None
        for offset in range(len(upstreams)):
            upstream = upstreams[(first + offset) % len(upstreams)]
            url = f"{upstream}{request.rel_url}"
            try:
                async with request.app["session"].request(
                    request.method,
                    url,
                    data=body,
                    headers={
                        key: value
                        for key, value in request.headers.items()
                        if key.lower() not in {"host", "content-length", "connection"}
                    },
                ) as response:
                    payload = await response.read()
                    return web.Response(
                        body=payload,
                        status=response.status,
                        headers={
                            key: value
                            for key, value in response.headers.items()
                            if key.lower() not in {"content-length", "transfer-encoding", "connection"}
                        },
                    )
            except Exception as error:  # Retry another local replica.
                last_error = error
        raise web.HTTPBadGateway(text=f"all upstreams failed: {last_error}")

    app = web.Application(client_max_size=128 * 1024**2)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    app.router.add_route("*", "/{path:.*}", proxy)
    web.run_app(app, host=args.listen_host, port=args.listen_port, access_log=None)


if __name__ == "__main__":
    main()
