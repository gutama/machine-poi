# Machine-POI: Agent Action Containment and Steering Research

Machine-POI combines a **host-side gateway for agent tool actions** with a research
library for Quran-derived activation steering, retrieval, and model diagnostics.
The components work independently: the guardian uses Python's standard library;
steering experiments use PyTorch, Transformers, and optional retrieval services.

The guardian checks what an agent is permitted to do at the tool boundary.
Steering changes model activations and can affect language, style, and task
performance. Neither steering nor a diagnostic score grants tool permissions.

For reproducible **Quran-grounded behavioral comparisons**, see the
[guidance pipeline](docs/quran_guidance.md): pinned configuration, cited RAG,
centered/paired steering, an opt-in bounded Euclidean rotor, and model proposals
mediated by the independent guardian against mock tools. The committed evidence
separates scripted integration from unavailable model runs. Quran guidance and
norm preservation are not evidence of ethical behavior or containment.

## Start with the guardian

From a checkout of this repository, with Python 3.10 or later:

```bash
python -m examples.guarded_agent.host
python -m examples.guarded_agent.process_demo
python -m evals.rogue_agent.run --output /tmp/machine-poi-evaluation.json
```

These commands need no model downloads, ML packages, API keys, or external tools.
To use the gateway from another project, `pip install /path/to/checkout` installs
`machine_poi` with no dependencies; import it as `from machine_poi.guardian import Gateway`.
The first demo executes an authorized mock write, pauses an internal send for
simulated operator approval, then blocks an external recipient. The second sends
JSON proposals from a separate worker process to the host. All effects are
in-memory mocks; the process demo is not an OS sandbox.

| Control | Implemented behavior |
| --- | --- |
| Task grants | Host-issued, expiring grants with exact tool, resource, destination and data-class scopes |
| Tool adapters | Strict argument fields/types; trusted code resolves scope from the actual arguments |
| Operator review | Sensitive actions pause; approval binds to stored arguments, resolved scope and versions |
| Budgets and replay | Atomic action/cost/token reservations, bounded attempts, shared ancestor budgets and single-use action IDs |
| Stop and recovery | Stop descendants, cancel queued/in-flight work cooperatively, invoke a host revocation callback |
| Audit and shadow mode | Redacted hash-chained events; preview decisions without executing tools |

**Deployment boundary:** this is a reference runtime for one trusted host process
and one async event loop. The host must authenticate callers, isolate the agent,
keep credentials outside its reach, and route every protected action through the
gateway. The research CLI is not automatically connected to the guardian. Real
credential revocation, durable state, remote-job cancellation and production
rollout remain host integration work. See the [guardian guide](docs/guardian_integration.md).

## Steering and retrieval research

The research library supports mean-activation and contrastive vectors, weighted
verse/passage/surah profiles, ChromaDB retrieval, optional LightRAG graph
retrieval, and per-layer/per-head diagnostics. Model weights are unchanged.

Install the research extras in a virtual environment:

```bash
python -m venv venv
. venv/bin/activate
python -m pip install -e ".[research]"
machine-poi --help        # or: python main.py --help
```

| Extra | Adds |
| --- | --- |
| `research` | PyTorch, Transformers, sentence-transformers and ChromaDB for steering and vector retrieval |
| `graph` | LightRAG graph retrieval and the Ollama provider |
| `providers` | OpenAI and Gemini clients for graph entity extraction |
| `quantization` | bitsandbytes for 4-bit and 8-bit loading |
| `test` | pytest and ruff |
| `all` | Every runtime extra; `requirements.txt` installs this |

For the CPU-only test environment used in CI, follow the [testing guide](docs/testing.md).
Actual inference requires model downloads and memory appropriate to the selected
checkpoint, dtype and context length.

A basic steering comparison is available through the CLI:

```bash
python main.py --llm qwen2.5-0.5b --dose-ratio 0.05 \
    --prompt "How should we resolve a disagreement?"
```

The dose ratio sets each steered layer's update to a fraction of that layer's
typical token norm, so the same ratio means a similar relative push on any model.
It is an experimental setting, not a validated safe dose. For
model-specific chat formatting, MRA/graph retrieval, dynamic steering opt-in,
cache migration, and the complete CLI reference, use the
[steering guide](docs/steering_guide.md). Comparison modes give the steered and
baseline runs the same retrieved context and random seed.

Recent runtime changes serialize model use and hook mutation, restore temporary
steering after failures, replace duplicate layer hooks, and correct clamp
strength and diagnostics. Remote model code defaults off; explicit opt-in
requires a full commit revision. Steering caches use numeric arrays with
model/corpus/recipe metadata and reject object arrays. Retrieval-derived dynamic
steering defaults off and requires an explicit trusted-corpus opt-in.

## Evidence and current limits

- **Runtime correctness:** CI runs lint, the guardian suite on Python 3.10 and 3.12
  without ML packages, and the offline runtime suite on every pull request; see
  [testing](docs/testing.md) for what each job covers.
- **Action containment fixtures:** the committed
  [report](evals/rogue_agent/results.json) covers single forbidden and benign
  actions and multi-step scenarios (review and approval, concurrency, delegation,
  stops in flight), all passing with zero unapproved mock side effects. A test
  fails CI if the report is stale. It evaluates already-proposed actions; it does
  not measure a model's resistance to prompt injection.
- **Steering behavior:** the [evaluation harness run](experiments/results/README.md)
  on Qwen2.5-0.5B-Instruct (48 held-out English and Arabic prompts, 95% intervals)
  found a trade-off: centered steering at ratio 0.05 left ARC-Easy accuracy
  unchanged within its interval, 0.1 raised a religious register but cost 15
  points of accuracy, and 0.2 degenerated 81% of outputs. Human relevance ratings
  are pending. These results do not establish preserved general capabilities,
  rogue-agent detection, or a universally safe dose.
- **Pending deployment:** no live agent host or real external side effects were
  evaluated in the guardian implementation. Held-out model comparisons and host
  bypass/kill-switch drills remain acceptance gates.

## Research directions and how to help

The current priority is [agent containment with bounded geometric interventions](docs/containment_geometry_research.md): test whether low-rank, norm-preserving
rotor steering reduces unauthorized proposals and review burden at matched benign
task utility. Host grants and adapters retain authority. The design specifies
factorial controls, adaptive attacks, service-receipt scoring, and gates before
learned metrics or mixed-curvature architectures.

The existing transport fields remain reproducible heuristics: the three-rotation
product does not define closed-loop holonomy. Run the mathematical controls with
`python experiments/geometry_sanity.py`; these do not evaluate an LLM or live host.


A [literature map](docs/research_directions.md) of 7,322 recent arXiv papers
provides a historical topic-growth analysis and collaborator problems. Topic
counts do not decide the current research priority. Arabic/Islamic steering
remains a language-matched ablation with its own evaluation needs. Several
existing open problems need no code:

- rating blinded steering outputs, and reviewing the Arabic prompts and controls;
- adding QuranicMMLU, IslamicMMLU and PalmX to the evaluation harness;
- running the gateway on AgentDojo and InjecAgent.

Each problem has a [help-wanted issue](https://github.com/gutama/machine-poi/labels/help%20wanted);
comment on it to claim the work. The [good first issues](https://github.com/gutama/machine-poi/labels/good%20first%20issue)
need no code.

## Documentation

| Document | What it covers |
| --- | --- |
| [Architecture](docs/architecture.md) | Components, authority boundary, execution flow and state ownership |
| [Guardian integration](docs/guardian_integration.md) | Runnable API example, grants, approvals, failures and host rollout |
| [Steering guide](docs/steering_guide.md) | Python/CLI usage, model aliases, injection semantics and migration |
| [Testing and evaluation](docs/testing.md) | Minimal and full test environments, evidence and experiment limits |
| [Steering evaluation](docs/evaluation.md) | Evaluation harness spec, held-out prompts, metrics, rating rubric and provenance |
| [Containment plan](docs/rogue_agent_containment_plan.md) | Baseline findings, delivered slices and remaining deployment gates |
| [Research note](PAPER.md) | Implemented steering methods and the evidence supporting current claims |
| [Research directions](docs/research_directions.md) | Current priorities, historical literature map and collaborator problems |
| [Containment and geometry design](docs/containment_geometry_research.md) | Threat model, rotor/metric ladder, transport corrections and falsifiable agent experiments |
| [Workspace research roadmap](docs/global_workspace_improvement_plan.md) | Diagnostic work and experiments still planned |
| [Improvement plan](docs/improvement_plan.md) | Whole-repository review findings and phased fixes |
| [Geometry literature notes](docs/curvature_literature_roadmap_review.md) | Research leads; proposed connections require validation |

## Repository map

| Path | Role |
| --- | --- |
| `machine_poi/guardian/` | Standard-library policy gateway, contracts, review, state, recovery and audit |
| `examples/guarded_agent/` | Mock host and JSON proposal worker |
| `evals/rogue_agent/` | Synthetic action cases, runner and committed report |
| `machine_poi/steerer.py`, `machine_poi/llm_wrapper.py` | Research orchestration, model loading and steering hooks |
| `machine_poi/retrieval_context.py`, `machine_poi/steering_cache.py` | Quoted/bounded context and numeric steering caches |
| `machine_poi/knowledge_base.py`, `machine_poi/hybrid_knowledge_base.py` | Vector and optional graph retrieval |
| `machine_poi/workspace_diagnostics.py`, `machine_poi/transport_stats.py` | Activation/transport summaries and paired statistics |
| `machine_poi/evaluation.py`, `experiments/steering_eval.py` | Evaluation metrics and the harness that produces reported steering results |
| `machine_poi/cli.py`, `machine_poi/config.py`, `main.py` | Research CLI, model aliases and presets; `main.py` launches the CLI from a checkout |
| `pyproject.toml`, `ci-constraints.txt` | Package metadata and extras; versions pinned in CI |
| `experiments/` | Model experiments and historical results |
| `tests/`, `.github/workflows/containment.yml` | Regression tests and CI |

## Research references

- Turner et al., [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248).
- Rimsky et al., [Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/).
- Lewis et al., [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401).

These are methodological references; their findings do not validate Machine-POI's
particular vectors, checkpoints, or containment implementation.

## License

Machine-POI is licensed under the [Apache License 2.0](LICENSE); see
[NOTICE](NOTICE). Contributions are accepted under the same license (section 5).

The license does not cover third-party material:

- `al-quran.txt` is third-party text. Its source, edition and terms are not yet
  recorded; see decision 1 in the [improvement plan](docs/improvement_plan.md#decisions-needed-from-the-owner).
- Models, datasets and benchmarks that the code downloads keep their own terms.
