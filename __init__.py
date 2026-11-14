"""Hermes entry point for the advisor-call plugin.

Wiring only; logic lives in the advisor_call sub-package. Hermes imports this
directory as the package ``hermes_plugins.advisor_call``, so every import
here must be relative.
"""


def register(ctx):
    """Register the plugin with Hermes.

    Phase 0 placeholder: registers nothing yet. Tools, hooks, middleware,
    the /phone command, the system prompt section and one auxiliary task per
    advisor line are wired in Phase 3.
    """
    return None
