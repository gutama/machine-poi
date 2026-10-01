# Integrating the guardian reference runtime

[README](../README.md) · [Architecture](architecture.md) ·
[Testing](testing.md) · [Steering guide](steering_guide.md)

## What runs now

Machine-POI now includes a standard-library-only action gateway in `machine_poi/guardian/`.
Install a checkout with `pip install /path/to/machine-poi`; the base package has no dependencies.
It checks structured proposals against host-issued grants, pauses actions for
operator review, binds approvals to stored arguments and resolved tool scope,
reserves budgets atomically, prevents replay, stops runs and descendants, and
records redacted hash-chained audit events. It also offers shadow evaluation that
never executes a tool.

This is a reference implementation for a **single trusted host process and one
async event loop**. It is not a network authentication service, OS sandbox,
credential vault, distributed state store, or validated rogue-intent detector.
The examples use mock tools only. No live system has been deployed or evaluated.

## Run the examples

From the repository root, with Python 3.10+:

```bash
python -m examples.guarded_agent.host
python -m examples.guarded_agent.process_demo
python -m evals.rogue_agent.run --output /tmp/rogue_agent_results.json
```

The first example simulates an authorized write, operator approval of an internal
send, and denial of an external send. The process example sends JSON proposals
from a fixed worker script to a host that owns the executor. The worker receives
no gateway object or operator API. It demonstrates an interface boundary; the
processes still share the OS account and are not a hardened sandbox.

The fixture runner measures policy enforcement **after an action proposal has
been produced**. The text in a fixture does not get passed through an LLM. The
report must not be interpreted as a prompt-injection success rate or evidence
that steering detects rogue agents. Live model and steering A/B tests require
real host traces and separate held-out tasks.

## Minimal host API example

This standalone example uses a list as its mock tool. The literal identities stand
in for authenticated host context; they must never be copied from an agent request.

```python
import asyncio
import time

from machine_poi.guardian import ActionScope, Gateway, ProposedAction, TaskGrant, ToolSpec

effects = []

def describe_write(arguments):
    return ActionScope(resources=frozenset({arguments["document"]}))

async def write(arguments, context):
    context.checkpoint()
    effects.append(dict(arguments))
    return {"receipt": len(effects)}

async def main():
    gateway = Gateway(
        [ToolSpec("write_note", "1", {"document": str, "text": str},
                  describe_write, write)],
        operators={"operator"},
    )
    gateway.issue("operator", TaskGrant(
        run_id="example", principal="agent", expires_at=time.time() + 60,
        tools=frozenset({"write_note"}), resources=frozenset({"draft:1"}),
        max_actions=1, max_attempts=3,
    ))
    allowed = ProposedAction.create(
        "example", "write-1", "write_note",
        {"document": "draft:1", "text": "Authorized draft"},
    )
    forbidden = ProposedAction.create(
        "example", "write-2", "write_note",
        {"document": "protected:1", "text": "Out of scope"},
    )
    print((await gateway.submit("agent", allowed)).status)
    print((await gateway.submit("agent", forbidden)).status)
    assert len(effects) == 1

asyncio.run(main())
```

Expected output is `executed`, then `blocked`. The second proposal fails the
resource check and stops the run. Cost/token scopes default to zero in this mock;
adapters that invoke paid or token-consuming services must declare enforceable
upper bounds and receive matching grant budgets.

| API | Caller and effect |
| --- | --- |
| `issue(operator, grant)` | Host operator creates a bounded run; does not execute a tool |
| `await submit(caller, action)` | Agent channel proposes one action; host supplies caller identity |
| `pending(operator, run_id)` | Host reads the immutable pending action, resolved scope, hash and expiry |
| `await approve(operator, run_id, action_hash)` | Host approves and executes the stored action once after revalidation |
| `stop(operator, run_id)` | Host stops the run and descendants; rejection also uses this path |
| `observe(operator, run_id, signal)` | Host records a known risk signal without adding authority |
| `preview(caller, action)` | Host shadow inspection; records an audit decision without dispatch or budget reservation |
| `state(run_id)` | Host reads current run state |
| `retire(operator, run_id)` | Host frees a stopped or expired run with no in-flight tasks or live children; the run ID can never be reissued |
| `stats(operator)` | Host reads live and retired run counts, unknown-caller requests and audit event count |

`ActionOutcome` carries `decision`, `status` and optional `result`. Inspect the
status before consuming a result: `pending` needs review; `blocked` did not
execute; `interrupted` or `uncertain` requires reconciliation. Do not automatically
retry uncertain effects. Default review lifetime is 300 seconds, capped by grant
expiry. The host must display the exact pending action and obtain an independent
decision before calling `approve`; the demo's simulated approval is not a UI.

The schema rejects unknown top-level arguments and exact-type mismatches. For
nested values, pass `ToolSpec(..., validators={"arg": rule})`. A rule is either a
callable that raises, or returns False, to reject, or a JSON-Schema subset
(`type`, `enum`, `const`, string and array bounds, `pattern`, `items`,
`uniqueItems`, `properties`, `required`, `additionalProperties`). Schemas compile
when the spec is built; unknown keywords are errors, objects are closed unless
`additionalProperties` allows more, and booleans never count as numbers. A failed
check denies the action as `invalid_arguments` and stops the run. Adapters still
own business rules that a schema cannot express. Scopes use exact string
identifiers; empty sets are not wildcards. `TaskGrant.goal` documents intent but
is not semantically evaluated by policy. The gateway is an importable component,
not an authenticated HTTP service or an automatic wrapper around the research CLI.

## Host integration contract

The host/agent separation described here is related to [Out-of-Band Policy Enforcement at a Trusted Tool Boundary](https://arxiv.org/abs/2608.27646), which places policy enforcement at a trusted boundary outside the agent's unconstrained reasoning path. [Progent](https://arxiv.org/abs/2504.11703) likewise uses host-side policy enforcement over tool calls, while [CaMeL](https://arxiv.org/abs/2503.18813) separates trusted control information from untrusted data. Machine-POI implements a narrower reference gateway: the host issues bounded grants, and the gateway rechecks structured proposals against those grants before execution.

1. Authenticate the user/operator and agent channel outside this library. The
   `operator` and `caller` arguments are trusted transport context, never fields
   copied from an agent JSON request. Only expose `submit` to an agent. Use
   `preview` for explicit shadow inspection; it never permits execution.
2. Keep the `Gateway`, `issue`, `approve`, `pending`, `observe`, `stop`, tool
   registry, and credentials in a separately permissioned host service. Never
   execute untrusted Python/plugins in this process. Do not give the agent direct
   credentials, filesystem/network paths to protected services, or a generic
   tool that can bypass mediation.
3. Implement a `ToolSpec` for every action surface. Its strict argument schema
   rejects unknown fields and type confusion. Its trusted `describe` method
   derives resources, destinations, data classes, cost/token upper bounds and
   review needs from the **actual arguments**. It must not read a model's claimed
   risk level. Resolve aliases, symlinks, URL redirects, recipients and tenant
   ownership correctly; use handles or service-side preconditions to avoid a
   target changing between resolution and execution. Nested tool calls and
   delegated model calls must pass through the gateway as well.
4. The async executor must use only the supplied validated arguments, enforce
   declared token/cost limits in the underlying service, and call
   `context.checkpoint()` immediately before each side effect. It must propagate
   cancellation. Blocking work, subprocesses and external remote jobs need
   their own enforceable deadlines, revocation and cancellation adapters.
5. Issue a frozen `TaskGrant` with exact allowed identifiers. Empty scope sets
   authorize nothing. Child grants can narrow scope and share the ancestor's
   aggregate budgets. Action, cost and token reservations are conservative and
   are not refunded after uncertain failures. The grant expiry supplies a time
   bound; host rate limits also protect the input channel and audit service.
6. On REVIEW, show the operator `pending.action.arguments`, `pending.scope` resolved by
   the trusted tool adapter, recipient/resource, reason, and expiry. Approval
   calls `approve(operator, run_id, pending.action_hash)`; that method executes
   the stored action once after checking scope again. Do not accept a fresh
   action body along with an approval. Rejection uses `stop` and a fresh task
   grant if a narrower retry is later authorized. A policy denial also stops the
   run. Changes to task intent require an operator-issued new run.
7. Supply a synchronous, bounded `on_stop(run_id)` callback that revokes scoped
   credentials and alerts the operator through the host's existing incident
   system. Logical stopping happens first and remains effective if that callback
   fails. A callback failure marks the gateway unavailable, stops all local runs,
   and cancels their tasks; an explicit `stop` reports the failure to the host.
   The incident owner must reconcile credential revocation. Integrate an
   independent kill switch for process, network and remote job control.
8. Persist the audit sink to host-only storage. Events exclude arguments, tool
   results, model text and raw exceptions. The hash chain detects ordinary edits
   against a known chain, not malicious rewriting/truncation by the storage
   owner; externally anchor receipts or use an append-only service for stronger
   integrity. By default the audit keeps every record in memory. For long-running
   hosts, pass `AuditLog(path, tail=N)` or `AuditLog(sink=callable, tail=N)`: every
   event is written through to the sink (fsynced for a path) and only the last N
   stay in memory. `count` and `head` continue the chain, and `verify()` re-checks
   the whole file. Retention and access controls for the sink are host
   responsibilities.
9. Retire finished runs. `issue` refuses new grants once `max_runs` runs are live.
   `retire` frees a stopped or expired run once its tasks have finished and its
   child runs are retired; an expired run is stopped first, so `on_stop` still
   revokes its credentials. A 16-byte tombstone per retired ID keeps it from being
   reissued, so replay protection outlives the run.

## State and failure semantics

| Event | Result |
| --- | --- |
| Unknown/missing grant or wrong authenticated caller | Deny; do not grant access or stop another principal's run. Such requests are counted, and only the 1st, 2nd, 4th, 8th... are audited, with the running count |
| Bad schema, forbidden tool/resource/destination/data class, or replay | Block and stop that run and its descendants |
| Sensitive action within scope | Pause; execute only after bound host review |
| Grant/approval expiry, exhausted budget, policy outage | Block and stop. Stopping also cancels actions already reserved but not yet started, so a request that exceeds a budget under concurrent load can leave none of the concurrent actions executed; this is intended |
| Audit append fails before dispatch | Stop; no tool execution |
| Tool failure, timeout, cancellation, or outcome audit failure | Stop; return interrupted/uncertain status, never retry automatically |
| Explicit kill switch | Mark stopped, cancel queued/in-flight tasks cooperatively, revoke descendants, invoke host callback |
| Restart | No automatic grant or approval restoration; issue fresh unique run IDs and reconcile uncertain effects |

No library can reverse an external action that already committed. Cancellation
can race with remote completion or be ignored by a faulty adapter. Treat such
outcomes as uncertain and use service-side idempotency, transaction receipts,
credential revocation and reconciliation. The gateway's in-process lock does not
coordinate multiple replicas. Use a transactional shared authorization/budget/
replay store before scaling out. The same audit path must have one writer.

## Steering changes and migration

The [steering guide](steering_guide.md) contains complete Python/CLI examples,
injection semantics and known CLI routing limits. The relevant migration points
for an embedding host are:

- One model serializes inference and hook mutation through an RLock. High-level
  temporary MRA/graph steering uses a synchronous session that restores the exact
  prior hooks and enabled flags, even after errors. Async retrieval completes
  before entering that session; no lock spans an await. Direct mutation of the
  wrapped model bypasses this contract.
- Repeated registration replaces the previous handle for that layer. Activation
  extraction runs with steering disabled and releases temporary hooks on error.
- Clamp now receives the effective coefficient explicitly. Its coefficient is a
  **target projection on the unit vector**; zero removes the existing projection
  and is not the same as disabling steering. Diagnostics compute the actual
  change for add, blend, clamp and replace. Replace remains a research mode.
- `QuranSteerer.last_run_diagnostics` retains scalar summaries after `generate`,
  `compare` and `generate_with_graph`, averaged over every steered token of that
  run. Hooks keep running statistics instead of copies of hidden states.
- Retrieval is quoted, bounded reference data. Dynamic steering from retrieval
  now defaults off. A trusted-corpus experiment must explicitly pass both
  `use_dynamic_steering=True` and `trusted_retrieval=True`. Quoting is not an
  injection detector; the tool gateway still enforces authority.
- Model and embedding remote code defaults off. `QuranSteerer` accepts
  `llm_revision`; `SteeredLLM` and `QuranEmbeddings` accept `revision`. Remote code
  opt-in requires a full commit hash. The CLI exposes `--revision` and
  `--trust-remote-code` for the LLM. Review the model code and snapshot
  before opt-in; a revision alone is not a safety review.
- Steering caches now contain numeric arrays plus JSON metadata for model,
  revision, corpus hash and recipe. Old/mismatched caches are rejected and
  recomputed when supported; object arrays are never loaded. Metadata prevents
  accidental reuse, not deliberate forgery. Protect cache storage. Pin a model
  revision for reproducible deployments; an unresolved local model revision is
  recorded explicitly and is not cryptographic model authentication.

## Rollout checklist for a specific host

Complete a deployment profile before enabling real effects:

| Field | Required evidence |
| --- | --- |
| Host/framework and owner | Named runtime and incident owner |
| Complete action inventory | Every tool, nested call, credential and direct network path |
| Adapter mapping | Canonical resource IDs, tenant/data classification, destinations, side effects and cost bounds |
| Isolation | Different host/agent identities, deny direct tool access, restricted egress/filesystem |
| Operator review | Authenticated reviewers, exact action display, approval expiry and rejection exercise |
| Stop and recovery | Queued, in-flight and descendant cancellation drills; external reconciliation procedure |
| State and audit | Crash/restart policy, replay store, single writer or transactional coordination, retention |
| Evaluation | Held-out benign/adversarial tasks, task completion, false blocks, review burden, latency and containment time |
| Promotion | Shadow observations reviewed by owner, scoped canary enforcement, rollback and incident criteria |

No values for these fields are assumed in this repository. Slice 5 remains a
deployment task until a real host and its permitted tools are supplied.
