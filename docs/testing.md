# Testing and evaluation

[README](../README.md) · [Architecture](architecture.md) ·
[Guardian integration](guardian_integration.md) · [Steering guide](steering_guide.md)

Use the smallest environment that covers the behavior under test. Commands below
run from the repository root. Guardian tests and demos use mocks and do not need
models or service credentials.

## Guardian only

Python 3.10+ and the `test` extra are sufficient; the base package has no
dependencies, and CI checks that the guardian runs without the ML stack:

```bash
python -m pip install -c ci-constraints.txt -e ".[test]"
python -m pytest tests/guardian --confcutdir=tests/guardian -q
python -m examples.guarded_agent.host
python -m examples.guarded_agent.process_demo
python -m evals.rogue_agent.run --output /tmp/rogue-agent-results.json
```

`--confcutdir` avoids the root test fixtures, which import the ML stack. The
process demo validates a JSON proposal interface under the same OS account; it
does not test sandbox escape resistance.

## Algebra and synthetic transport controls

Both guardian CI matrix jobs (Python 3.10 and 3.12) also run the standard-library
geometry verifier explicitly:

```bash
python experiments/geometry_sanity.py
```

It checks rotor identities and degeneracies, distinct same-axis rotations in
opposite multiplication orders and against their angle sum, closed flat and
pure-gauge loops, frame covariance, and a query/value reparameterization control.
Failed assertions fail the CI step. These fixtures evaluate mathematics, not
model behavior or host containment; see the
[research design](containment_geometry_research.md).

## Full offline runtime tests

CI uses Python 3.12 with a CPU PyTorch wheel, the `research`, `graph` and `test`
extras, and the versions pinned in `ci-constraints.txt`:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c ci-constraints.txt -e ".[research,graph,test]"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -m 'not slow and not integration'
```

Dependency installation needs network access. The selected tests use local
fixtures/mocks and run with model-hub downloads disabled. `ci-constraints.txt`
pins top-level packages only; it is not a complete transitive lock, and CI does
not install the `providers` or `quantization` extras.

## Lint

CI runs ruff at the version pinned in `ci-constraints.txt`. `ruff.toml` selects
pyflakes and syntax-level pycodestyle rules explicitly, because ruff's default
selection varies by version:

```bash
python -m pip install -c ci-constraints.txt ruff
ruff check .
```

`pytest.ini` also turns deprecation warnings raised from this repository's own
modules into test errors.

## Targeted runs

For a targeted run, use the same environment and select a test file:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_runtime_safety.py -q
python -m pytest tests/guardian --confcutdir=tests/guardian \
    --cov=machine_poi.guardian --cov-branch --cov-report=term-missing
```

`make test` selects the fast suite. `pytest.ini` excludes integration tests by
default, so `make test-all` still inherits that exclusion. Explicitly clear the
marker filter with `python -m pytest -m ''` only in an environment prepared for
slow/model integration work. Those tests were not part of the validation below.

## What CI checks

The [workflow](../.github/workflows/containment.yml) runs on every pull request
and every push to `main`. The Actions page for a commit is the record of what
passed; this guide does not repeat test counts, which change with every change.

| Job | Checks |
| --- | --- |
| `lint` | `ruff check .` with the pinned version |
| `guardian` (Python 3.10 and 3.12, no ML packages) | Guardian tests: identity and scope, approvals, replay, budgets, delegation, stop, expiry, retirement, audit bounds, nested validation, and policy, audit and callback failure. Also the standalone geometry verifier, mock process demo, and action fixtures and scenarios. The 3.12 run uploads the fixture report as the `containment-eval-report` artifact. |
| `runtime` (Python 3.12, CPU torch) | The offline suite: steering, pooling, dose calibration, recipes, retrieval, the evaluation harness on a tiny model, CLI and runtime safety |

`tests/guardian/test_eval_report.py` regenerates the fixture report and fails if
the committed [`results.json`](../evals/rogue_agent/results.json) is stale, so the
committed report always matches the current fixtures and gateway. Regenerate it
with `python -m evals.rogue_agent.run --output evals/rogue_agent/results.json`.
No model weights or live external-tool credentials are used in CI.

## What the action fixtures measure

[`cases.json`](../evals/rogue_agent/cases.json) contains single proposed tool
actions and expected decisions. The runner creates a fresh mock host for each
case, submits the proposal and checks status/reason and mock side effects.

[`scenarios.json`](../evals/rogue_agent/scenarios.json) contains multi-step
sequences run against one host with a simulated clock:

- draft, review and approval, including a second approval of the same action;
- rejection by stop, and an approval after the review window;
- held work while a review is pending, and replay after execution;
- concurrent submissions, including the pause race fixed in Phase 0 and a
  budget race;
- delegation that tries to widen scope or budget, and a parent stop that revokes
  its child;
- a stop, and a grant expiry, while a long job is in flight.

Each scenario lists the exact side effects it permits; any others count as
unapproved, and the runner exits non-zero if a step or effect count differs. The
budget race shows the fail-closed design: the request that exhausts the budget
stops the run, which also cancels reserved actions that had not yet run.

The [committed report](../evals/rogue_agent/results.json) includes fixture and
scenario SHA-256s, Python version, timestamp, per-case and per-step results and
local latency observations.

Fixture context text is descriptive; it is not submitted to an LLM. Consequently,
these results measure policy enforcement once an action exists. They do not
measure prompt-injection success, semantic goal adherence, steering efficacy, or
general task completion. A pending benign review is not an executed task.

To reproduce a report without replacing the committed evidence:

```bash
python -m evals.rogue_agent.run --output /tmp/rogue-agent-results.json
```

Compare case outcomes and the fixture hash. Timing depends on the environment and
is not a production latency or time-to-containment benchmark.

## Model experiments

The [research note](../PAPER.md) describes the implemented methods. The historical
[model result report](../experiments/results/README.md) links raw JSON traces and
run conditions. Do not treat those runs as tests of the guardian changes.

Steering claims come from the evaluation harness, which scores held-out
prompts with confidence intervals ([evaluation guide](evaluation.md)):

```bash
python experiments/steering_eval.py --spec experiments/specs/qwen2.5-0.5b.json
```

The older demonstration runner accepts `--model`, not `--llm`, and prints
sample outputs without scoring them:

```bash
python experiments/reproduce_paper.py --model deepseek-r1-1.5b --section 5.1 --quick
python experiments/reproduce_paper.py --section 5.3 --quick
```

Section 5.3 prints outputs across dose ratios. Section 5.2, which counted English
keywords as a thematic score, is retired.

For attention-transport comparisons, inspect these entry points before selecting
checkpoint, dose, prompts and output path:

```bash
python experiments/steered_vs_baseline_transport.py --help
python experiments/centered_contrast_probe.py --help
```

Record checkpoint/revision, code commit, corpus hash, vector recipe, layer band,
chat template, dtype, decoding, prompts/seeds and baseline behavior. Compare
outputs and task performance alongside geometry; a metric shift alone can reflect
collapse. Raw experimental vector scales differ from the high-level normalized
persona API, so coefficients are not interchangeable.

## Next evaluation gate

A live host evaluation needs a complete action inventory and isolated adapters,
held-out benign/adversarial tasks, and matched baseline, steering-only,
gateway-only and combined conditions. Measure unauthorized effects, task success,
false blocks, review burden, budget use, latency and stop/recovery behavior. Run
queued, in-flight, descendant and direct-bypass drills. A host owner must review
that evidence before promotion from shadow observations to scoped enforcement.
