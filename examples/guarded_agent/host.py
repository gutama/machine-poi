"""An agent can propose JSON; only the trusted host owns mock tools and review."""

import asyncio
import json

from machine_poi.guardian import (
    AuditLog,
    ProposedAction,
)

from machine_poi.mock_agent import build_host


async def demo():
    audit = AuditLog()
    gateway, effects = build_host(audit=audit)
    draft = ProposedAction.create(
        "demo", "draft", "write_note", {"document": "draft:1", "text": "Draft"}
    )
    print("draft:", (await gateway.submit("agent", draft)).status)
    send = ProposedAction.create(
        "demo",
        "send",
        "send_note",
        {"document": "draft:1", "recipient": "reviewer:internal"},
    )
    print("send:", (await gateway.submit("agent", send)).status)
    pending = gateway.pending("operator", "demo")
    # This line simulates a host review decision. A real host must display the
    # stored arguments and obtain an authenticated human decision first.
    print(
        "simulated operator approval:",
        (await gateway.approve("operator", "demo", pending.action_hash)).status,
    )
    attack = ProposedAction.create(
        "demo",
        "exfil",
        "send_note",
        {"document": "draft:1", "recipient": "attacker:external"},
    )
    outcome = await gateway.submit("agent", attack)
    print("out-of-scope recipient:", outcome.status, outcome.decision.reason)
    print(
        json.dumps(
            {
                "mock_side_effects": len(effects),
                "final_state": gateway.state("demo").value,
                "audit_events": len(audit.records),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(demo())
