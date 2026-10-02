# Steering and retrieval guide

[README](../README.md) · [Architecture](architecture.md) ·
[Guardian integration](guardian_integration.md) · [Testing](testing.md)

This guide covers the research API. Run examples from the repository root after
installing research dependencies. Inference examples load model weights; graph
indexing/querying can invoke the configured provider. They do not automatically
pass through the guardian.

## Quran-guidance comparisons and experimental rotor

Use the [reproducible guidance pipeline](quran_guidance.md) for pinned revisions,
verse-cited RAG, matched behavioral pairs, development dose sweeps and opt-in G1
rotor comparisons submitted to mock guardian tools. `QuranGuidanceConfig` and
`QuranGuidance` are public APIs. The CLI accepts `--guidance-config`,
`--guidance-mode validate|mock|model`, and `--guidance-output`.
Sentence embeddings select references; model-native activations supply directions;
only the independent guardian authorizes effects. Norm preservation and Quran
context establish neither ethical behavior nor containment.

## Generate with mean-activation steering

```python
from machine_poi import QuranSteerer

steerer = QuranSteerer(
    llm_model="qwen2.5-0.5b",
    embedding_model="paraphrase-minilm",
)
steerer.load_models()
steerer.config.dose_ratio = 0.05  # the default; see "Dose" below
steerer.prepare_quran_steering(
    chunk_by="verse", sample_size=8, cache_path="vectors/example_mean.npz"
)

# Both arms share the prompt, chat template and seed.
steered, baseline = steerer.compare(
    "How should we resolve a disagreement?", max_new_tokens=100
)
print(steered)
print(steerer.last_run_diagnostics)
print(baseline)
```

This is a small API demonstration, not a calibrated behavioral evaluation.
`prepare_quran_steering` pools unsteered activations and, by default
(`recipe="centered"`), normalizes `mean(Quran) - mean(neutral Arabic control)` at
each layer. Without centering, the mean is dominated by the component every hidden
state shares; `recipe="raw_mean"` keeps that older vector for reproducing earlier
results and warns. `prepare_quran_persona` instead combines centered verse,
paragraph and surah directions with default weights 0.50, 0.35 and 0.15, then
normalizes the result. Use one preparation method for the experiment being measured.

`ContrastiveQuranSteerer.prepare_contrastive_steering(positive_texts,
negative_texts)` constructs normalized differences between activation means.
`prepare_quran_contrastive()` contrasts sampled verses with a language-matched
control: up to 50 distinct sentences from `machine_poi/data/neutral_arabic.txt`,
120 short Modern Standard Arabic sentences written for this project about weather,
science, daily life and similar topics. `machine_poi.controls.neutral_texts("en")`
returns the older ten-sentence English set for an explicit cross-language run.
Repeated negatives are dropped before pooling, and cached vectors record a hash of
both text sets. Matching the language removes the largest confound, but the
contrast still mixes register (classical versus modern prose) and topic, so it
does not isolate moral behavior. A native speaker has not yet reviewed the Arabic
set.

Generation wraps the prompt as one user turn in the tokenizer's chat template
whenever the tokenizer has one, so pass plain text rather than templated text.
Pass `chat_template=False` to send a prompt unchanged, for example a transcript
you have already formatted. Templated text is tokenized without adding special
tokens again, which avoids a doubled BOS. For Qwen3, `reasoning_mode` switches the
template's thinking on or off; DeepSeek-R1 reasoning starts the response with
`<think>`. The transport experiments keep their recorded prompt formatting. A
fluent baseline is a prerequisite for interpreting a steering comparison.

## Retrieval and dynamic steering

Continue with the `steerer` instance above to build a vector index and use MRA:

```python
steerer.initialize_knowledge_base()
steerer.knowledge_base.build_index("al-quran.txt")
answer = steerer.generate(
    "How should I handle team conflict?",
    mra_mode=True,
    use_dynamic_steering=False,
)
print(answer)
```

MRA retrieves verse, passage and surah context and adds it to the prompt, each
item prefixed with its reference, such as `[2:255]` or `[2:254-272]`. Passages are
windows of up to 19 verses that never cross a surah boundary. The corpus file must
hold exactly 6,236 lines, one verse per line in mushaf order; any other layout
raises `CorpusError` rather than misnumbering verses. An explicit `--quran-path`
that does not exist is an error; only the default `al-quran.txt` falls back to the
checkout's copy. Each index collection records the embedding model, its
dimension, the corpus SHA-256 and a schema version. If any differ from the current
configuration, for example after changing `--embedding`, queries and builds raise
`StaleIndexError` until you run `machine-poi --init-db --rebuild`. Indexes built
before this check existed need one rebuild. The steerer shares its loaded embedder
with the index instead of loading a second copy. This path assembles its own
prompt; check checkpoint formatting when designing an experiment. Retrieval-derived activation steering is a separate, explicit opt-in:

```python
answer = steerer.generate(
    "How should I handle team conflict?",
    mra_mode=True,
    use_dynamic_steering=True,
    trusted_retrieval=True,
    dynamic_blend_ratio=0.3,
)
```

`trusted_retrieval=True` is an assertion by the integrator, not a corpus integrity
check. Use it only for an intentionally trusted research corpus. Text returned by
MRA and graph retrieval is quoted as reference data with a 12,000-character bound
per MRA resolution, or for the combined graph context. Oversized context raises
an error instead of silently truncating. Arabic and other scripts stay readable
inside the JSON; control, zero-width, bidirectional-override and separator
characters are escaped, so they cannot hide or reorder text. Earlier versions
escaped all non-ASCII text, which made Arabic context unreadable to the model and
several times longer in tokens. Quoting does not detect prompt injection.

## Graph retrieval

Use the async API for graph-enhanced generation. This standalone example expects
`GRAPH_MODEL` to name a model accessible to the configured OpenAI account and
`OPENAI_API_KEY` to be available to its client. Indexing can make many provider
calls. Ollama and Gemini adapter factories are also available in
`machine_poi/llm_adapters.py`; configure their model, endpoint and credentials for your host.

```python
import asyncio
import os

from machine_poi import QuranSteerer
from machine_poi.llm_adapters import create_openai_adapter

async def main():
    steerer = QuranSteerer(
        llm_model="qwen2.5-0.5b",
        use_graph_kb=True,
        llm_func=create_openai_adapter(model_name=os.environ["GRAPH_MODEL"]),
    )
    steerer.load_models()
    steerer.prepare_quran_steering(sample_size=8)
    await steerer.initialize_hybrid_knowledge_base()
    # Build once, then reuse the index on later runs.
    await steerer.hybrid_kb.build_index("al-quran.txt", build_graph=True)
    print(await steerer.generate_with_graph(
        "How should I handle team conflict?",
        query_mode="hybrid",
        use_dynamic_steering=False,
    ))

asyncio.run(main())
```

`query_mode` accepts `vector`, `graph`, `hybrid` or `auto`. Graph bridges start
from seed entities: concepts mapped from query terms, plus graph labels named in the
query. They are the seeds' direct neighbors in the LightRAG graph, ranked by edge
weight, with thematic relation types (such as "requires" or "leads to") counted
double. `HybridQueryResult.graph_entities` and `graph_relationships` report the
seeds and edges used. Without a usable graph the bridges fall back to embedding
similarity with the curated themes, then to the unverified seed concepts; the
confidence scores show which source applied. Async retrieval completes before the
synchronous steering session starts; do not hold that session across an `await`. Provider calls and index storage need their own authorization boundary
when incorporated into an agent host.

## Injection semantics

Let `h` be a token's hidden state, `v` the supplied vector, and `a` the hook
coefficient. Clamp uses `u = v / (norm(v) + 1e-8)`.

| Mode | Hook operation | Interpretation |
| --- | --- | --- |
| `add` | `h + a * v` | Vector addition |
| `blend` | `(1 - a) * h + a * v` | Interpolation; coefficient must be in [0, 1] |
| `replace` | `v` at every position | Erases the original hidden state; the low-level hook ignores its coefficient |
| `clamp` | `h - dot(h, u) * u + a * u` | Sets a projection along the normalized direction, up to numerical epsilon |

The high-level API applies a layer-distribution scale to the hook coefficient.
For `replace`, it scales the vector before registering the hook.
`SteeringConfig` accepts dose ratios in [-1, 1] and raw coefficients in [0, 2],
with [0, 1] for blend; the low-level hook accepts any finite coefficient, with the
same blend constraint. These are configuration bounds, not validated safety
thresholds.

## Dose

The high-level dose is a **target relative perturbation**, `SteeringConfig.dose_ratio`
(default 0.05). At each steered layer the hook coefficient is

    a_l = dose_ratio * scale_l * n_l / norm(v_l)

where `scale_l` is the layer-distribution scale and `n_l` is the layer's median
per-token hidden-state norm on a calibration set. Add mode then moves every token
by `dose_ratio * scale_l * n_l`. The first application calibrates on the English
control sentences plus ten Arabic ones (`machine_poi.controls.calibration_texts`);
call `steerer.calibrate_dose(texts)` to use other texts. Calibration pools content
tokens only, and the median ignores the first-position attention-sink token,
whose norm can be orders of magnitude above the rest. `last_run_settings` records
the per-layer coefficients and a hash of the calibration texts.

Ratios, not raw coefficients, transfer between models. High-level vectors are
unit-norm, and the norms of committed mean-activation vectors range from 59–96
(Gemma 4) to about 2,375 (SmolLM2), so a coefficient that moves one model
noticeably is inert on another. Presets are ratios: `gentle` 0.02, `moderate` and
`focused` and `workspace` 0.05, `strong` 0.1. In the
[evaluation run](../experiments/results/README.md) on Qwen2.5-0.5B-Instruct,
centered vectors at 0.05 left ARC-Easy accuracy unchanged within its interval, 0.1
cost 15 points while adding a religious register even to neutral answers, and 0.2
degenerated most outputs. These presets are starting points, not validated doses;
measure a new model with the [evaluation harness](evaluation.md) before relying on
one. A negative ratio steers away from the direction, for ablations.

Dose ratios apply to add mode. For blend, replace or clamp, set
`dose_ratio=None` and a raw `coefficient`; `set_steering_strength(c)` does this,
and `set_dose_ratio(r)` switches back. The achieved ratio varies with the prompt,
since calibration is fixed: `last_run_diagnostics[layer].dose_ratio` reports the
mean update norm divided by the run's median token norm
(`median_activation_norm`). The older `relative_perturbation` divides by the
mean token norm instead, which the attention-sink token inflates.

**Clamp coefficient zero still removes the existing projection.** Use
`steering_disabled()` or `generate_unsteered()` for an unsteered baseline. No
injection mode is established as universally more fluent or more stable. Read the
[committed results](../experiments/results/README.md) before interpreting a dose.

## Diagnostics and state lifetime

After `QuranSteerer.generate`, `compare` or `generate_with_graph`, read
`steerer.last_run_diagnostics`. Each steered layer keeps running statistics over
every token it steered in the latest generation, prompt and decode steps alike,
so the summary describes the whole output rather than its last token. A new
high-level generation resets this field; it is not a per-request history.

At the low level, `SteeredLLM.get_steering_diagnostics()` summarizes enabled
hooks' statistics since the last `generate` call began. It uses the actual
add/blend/replace/clamp delta. Hooks no longer copy hidden states; register with
`capture=True` to keep the latest hidden states in `captured_activation`.
Vector preparation pools activations in batches with `pooled_layer_means`, which
right-pads each batch, masks padding out of the mean and matches one-text-at-a-time
pooling to floating-point tolerance. BOS, EOS and other special-token positions are
also left out of the mean, because the first position carries a large activation
shared by every text. Set `STEERING_DEFAULTS.pool_exclude_special_tokens = False`
to average every token as before; cache metadata records which pooling was used. `get_attention_transport_diagnostics(prompt)` makes a separate
forward pass; wrap it in `steering_disabled()` for its baseline. Geometry and
perturbation metrics are research measurements, not action authorization signals.

`QuranSteerer.compare` accepts the same options as `generate`. With
`mra_mode=True` it retrieves context once, then generates the steered and baseline
outputs from the same final prompt and random seed (`seed` defaults to
`STEERING_DEFAULTS.random_seed`), so the arms differ only in steering. It stores
the steered run's scalar summaries in `last_run_diagnostics`. It does not assemble
graph context; use `generate_with_graph` for that. `SteeredLLM.generate` raises
`TypeError` for retrieval options such as `mra_mode` instead of ignoring them.
`last_run_settings` records what the latest `generate`, `compare` or graph run
actually used: seed, greedy or sampling (with the effective temperature, which
reasoning mode can override), chat templating, retrieval, a SHA-256 of the final
prompt, the steering configuration and the per-layer hook coefficients. Record it
next to any output you report.

## Model loading and cache migration

| Area | Current behavior | Migration action |
| --- | --- | --- |
| Remote code | Off by default for LLMs and embedders | Review code before opt-in; supply a full 40-character commit revision |
| LLM revision | `QuranSteerer(llm_revision=...)` or `SteeredLLM(revision=...)` | Pin the checkpoint for reproducible runs |
| Embedding revision | `QuranEmbeddings(revision=...)` or `QuranSteerer(embedding_revision=...)` | Pin separately from the LLM; remote embedder code still defaults off |
| Steering caches | Numeric NPZ arrays and JSON model/revision/corpus/recipe metadata, format 3 | Older formats and mismatched caches are recomputed, with a log line naming the reason; do not convert them by loading pickle |
| Corrupt artifacts | Invalid arrays/metadata are rejected; supported cache errors trigger recomputation | Other corruption can raise; investigate and rebuild from a trusted source |
| Dynamic retrieval steering | Off by default | Explicitly pass both opt-in flags for trusted-corpus experiments |
| Temporary hooks | Restored after high-level generation, including failure | Use scalar diagnostics instead of relying on retained activation tensors |

Metadata detects accidental cache reuse; it is not a signature. Protect model and
cache storage from agent writes. An unresolved local revision is recorded as
`unresolved`. Repeated registration replaces a layer's previous hook. Use public
wrapper APIs for serialized inference; direct model/hook mutation bypasses them.

## CLI reference

```bash
python main.py --help
python main.py --llm qwen2.5-0.5b --dose-ratio 0.05 --prompt "What is justice?"
python main.py --quran-persona --interactive
python main.py --preset workspace --layer-distribution workspace --interactive
python main.py --init-db
python main.py --mra --interactive
python compare_models.py --list-models
```

Every generation path forwards `--max-tokens`, `--temperature`, `--mra` and
`--reasoning`. Single-prompt, default, `--compare` and interactive comparison runs
call `compare`, so `--mra` adds the same MRA context to both arms once the vector
index exists. Interactive mode with comparison toggled off calls `generate`. `--graph-kb`
configures the graph provider and enables graph index building with
`--init-db --build-graph`; current CLI generation does not call
`generate_with_graph`. Use the async API above for graph generation.

| Flag | Behavior |
| --- | --- |
| `--llm`, `--llm-path` | Registered alias, or `--llm custom --llm-path MODEL_PATH`; default `deepseek-r1-1.5b` |
| `--embedding` | Registered embedding alias; default `paraphrase-minilm` |
| `--revision`, `--trust-remote-code` | LLM revision and reviewed-code opt-in; opt-in requires a full commit hash |
| `--preset` | `gentle`, `moderate`, `strong`, `focused`, `workspace`; when omitted, `moderate` with a registered model's recommended layers |
| `--dose-ratio` | Target relative perturbation per layer, in [-1, 1]; negative steers away; add mode only |
| `--coefficient` | Raw coefficient instead of a ratio, including `0`: [0, 2], blend [0, 1]; required for blend, replace and clamp |
| `--injection-mode` | `add`, `blend`, `replace`, `clamp` |
| `--layer-distribution` | `uniform`, `bell`, `focused`, `workspace` |
| `--chunk-by`, `--quran-persona`, `--theme` | Select text resolution (default from preset), weighted persona, or thematic preparation |
| `--recipe` | `centered` (default) or `raw_mean` for mean and persona vectors |
| `--quran-path`, `--cache-dir` | Corpus and steering-cache paths |
| `--device`, `--quantize` | Device (`cpu`, `cuda`, `mps`) and optional `4bit`/`8bit` loading |
| `--max-tokens`, `--temperature` | Generation options, forwarded on every CLI path |
| `--seed`, `--greedy` | Seed shared by both comparison arms (default 42); decode greedily instead of sampling. Comparisons print the settings used |
| `--interactive`, `--compare`, `--prompt` | Interactive generation, predefined comparisons, or one comparison prompt |
| `--reasoning` | Model-specific prompt/decoding behavior; inspect it when matching experimental conditions |
| `--init-db`, `--rebuild`, `--mra` | Build the vector index (`--rebuild` replaces an existing one); add MRA context on every generation path |
| `--graph-kb`, `--build-graph` | Configure graph provider; build graph with `--init-db` |
| `--llm-provider`, `--llm-api-model` | Provider (`openai`, `gemini`, `ollama`) and its model name |

Settings resolve in this order: an explicit flag, then `--preset`, then the
model's recommended layers, then `moderate`. `--dose-ratio` and `--coefficient`
are mutually exclusive, and a non-add `--injection-mode` needs `--coefficient`.
Both are validated before models load. `--layer-distribution` selects layers
from that distribution instead of a model's recommended layers. A zero
coefficient is applied as given; remember that zero clamp is not an unsteered
baseline. In interactive mode, `strength <value>` changes whichever kind of dose
the run uses.

## Registered model aliases

These are the repository's convenience mappings, not a current compatibility or
quality certification for every checkpoint/dependency combination. Custom paths
also require a supported model layout and sufficient memory. `LLM_MODELS` and
`EMBEDDING_MODELS` in `machine_poi/config.py` are the only registries; the CLI,
`SteeredLLM`, `QuranEmbeddings` and `compare_models.py` read them. Hidden size and
layer count come from the loaded checkpoint.

| LLM alias | Checkpoint |
| --- | --- |
| `deepseek-r1-1.5b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `phi4-mini` | `microsoft/Phi-4-mini-reasoning` |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` |
| `smollm3` | `HuggingFaceTB/SmolLM3-3B` |
| `gemma-270m` | `google/gemma-3-270m-it` |
| `gemma-4-e2b` | `google/gemma-4-E2B-it` (no recommended layers; uses the preset's distribution) |
| `gemma-4-e4b` | `google/gemma-4-E4B-it` (no recommended layers; uses the preset's distribution) |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `smollm2-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| `smollm2-360m` | `HuggingFaceTB/SmolLM2-360M-Instruct` |

| Embedding alias | Checkpoint |
| --- | --- |
| `paraphrase-minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| `paraphrase-mpnet` | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `bge-m3` | `BAAI/bge-m3` |
| `multilingual-e5` | `intfloat/multilingual-e5-large-instruct` |
| `multilingual-e5-large` | `intfloat/multilingual-e5-large` |
| `qwen-embedding` | `Alibaba-NLP/gte-Qwen2-7B-instruct` |
