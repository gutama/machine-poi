"""Reference host with named mock tools; no external services or credentials."""

import time
from .guardian import ActionScope, Gateway, TaskGrant, ToolSpec


def build_host(*, audit=None, clock=time.time, slow_job=None):
    """A gateway with mock tools. ``slow_job`` (an asyncio.Event) adds a
    ``long_job`` tool that waits for the event before its side effect."""
    effects = []

    def describe_write(arguments):
        # Production adapters must resolve the real resource, destination and
        # data class, including aliases/redirects/symlinks, immediately before use.
        return ActionScope(
            frozenset({arguments["document"]}), data_classes=frozenset({"internal"})
        )

    def describe_send(arguments):
        # Registry-owned metadata: the agent cannot claim this is a read.
        return ActionScope(
            frozenset({arguments["document"]}),
            frozenset({arguments["recipient"]}),
            frozenset({"internal"}),
            requires_review=True,
        )

    async def write(arguments, context):
        context.checkpoint()
        effects.append({"tool": "write_note", **arguments})
        return {"receipt": len(effects)}

    async def send(arguments, context):
        context.checkpoint()
        effects.append({"tool": "send_note", **arguments})
        return {"receipt": len(effects)}

    tools = [
        ToolSpec(
            "write_note", "1", {"document": str, "text": str}, describe_write, write
        ),
        ToolSpec(
            "send_note",
            "1",
            {"document": str, "recipient": str},
            describe_send,
            send,
        ),
    ]
    if slow_job is not None:

        async def long_job(arguments, context):
            await slow_job.wait()
            context.checkpoint()
            effects.append({"tool": "long_job", **arguments})
            return {"receipt": len(effects)}

        tools.append(
            ToolSpec("long_job", "1", {"document": str}, describe_write, long_job)
        )
    gateway = Gateway(tools, operators={"operator"}, audit=audit, clock=clock)
    grant = TaskGrant(
        "demo",
        "agent",
        clock() + 600,
        frozenset(tool.name for tool in tools),
        frozenset({"draft:1"}),
        frozenset({"reviewer:internal"}),
        frozenset({"internal"}),
        max_actions=5,
        max_attempts=10,
        goal="Draft a note and send it to the internal reviewer after approval",
    )
    gateway.issue("operator", grant)
    return gateway, effects

