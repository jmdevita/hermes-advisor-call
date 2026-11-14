"""Hermes entry point for the advisor-call plugin.

Wiring lives in advisor_call/plugin.py. Hermes imports this directory as the
package ``hermes_plugins.advisor_call``; the relative import runs inside
register() so that importing this file outside a package (pytest collects
the repo root) does not fail.
"""


def register(ctx):
    from .advisor_call.plugin import register as _register

    _register(ctx)
