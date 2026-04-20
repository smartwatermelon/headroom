"""Third-party proxy extension point.

External packages hook into the Headroom proxy at startup by declaring an
entry point in the ``headroom.proxy_extension`` group in their ``pyproject.toml``:

    [project.entry-points."headroom.proxy_extension"]
    my_extension = "my_pkg.extension:install"

Each ``install`` callable is invoked with the FastAPI ``app`` and the
``ProxyConfig`` at app creation time, and is free to:

  * register ASGI middleware (``app.add_middleware(...)``)
  * add routes or health endpoints
  * mutate config
  * raise on license / environment failure to abort startup

OSS makes no assumptions about what extensions do. The interface is
deliberately minimal; extensions own the complexity behind it.

Stability contract: this module is load-bearing for the Enterprise build and
any third-party extensions. Changes to the signature of ``install(app, config)``
or the entry-point group name require a deprecation cycle.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable, Iterator
from typing import Any

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "headroom.proxy_extension"

ProxyExtension = Callable[[Any, Any], None]
"""Signature: ``install(app: FastAPI, config: ProxyConfig) -> None``."""


def discover() -> Iterator[tuple[str, ProxyExtension]]:
    """Yield ``(name, install_callable)`` pairs for every registered extension.

    Entry-point load failures are logged and skipped — a broken third-party
    package must not prevent the proxy from starting. An extension that wants
    to fail-closed can raise from its ``install()``.
    """
    try:
        entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 — importlib.metadata can raise varied types
        log.debug("proxy extensions: entry-point enumeration failed: %s", exc)
        return
    for entry in entries:
        try:
            install = entry.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("proxy extension %r failed to load: %s", entry.name, exc)
            continue
        yield entry.name, install


def install_all(app: Any, config: Any) -> list[str]:
    """Run every discovered extension's ``install(app, config)``.

    Returns the names of successfully installed extensions. If an extension
    raises inside ``install()``, the exception propagates — this is the
    documented fail-closed signal (e.g., a Shield Enterprise license check
    failing should abort startup rather than silently run without protection).
    """
    installed: list[str] = []
    for name, install in discover():
        install(app, config)
        installed.append(name)
        log.info("proxy extension installed: %s", name)
    return installed
