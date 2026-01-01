# Implementation Plan - Quran Persona Integration

We will implement a "Quran Persona" mode that aggregates **all** embedding vectors (Verse, Passage, and Surah levels) to construct a comprehensive "Quran Character" vector. This adapts the "Persona Vector" concept (consistent character steering) using the rich semantic data of the entire Quran.

## Status: ✅ Implemented

> [!NOTE]
> This replaces the generic `--persona` proposal. We created a single, robust "Quran Persona" by fusing embeddings from all resolutions.
> 
> **New in latest update**: Added `clamp` injection mode (inspired by [Eiffel Tower LLaMA](https://github.com/scienceetonnante/eiffel-tower-llama)) for more stable steering at higher coefficients.

## Proposed Changes

### [src/steerer.py](src/steerer.py)

#### [MODIFY] `QuranSteerer` class
- **Add method**: `prepare_quran_persona()`
    - Loads embeddings for ALL resolutions: `verse`, `paragraph` (passage), and `surah`.
    - Computes a global weighted mean of all these embeddings to capture the "Holographic" character of the Quran.
    - Projects this "Persona Embedding" to a steering vector.
    - Applies this vector globally.

### [main.py](main.py)

#### [MODIFY] CLI Arguments
- Add `--quran-persona` flag.
- Add `--injection-mode` flag with choices: `add`, `blend`, `replace`, `clamp`.
- When enabled alongside `--mra`, the "Persona" provides the global character, while MRA provides local context steering.

### [src/llm_wrapper.py](src/llm_wrapper.py)

#### [MODIFY] `ActivationHook` class
- Add `clamp` injection mode:
  - Removes the existing projection of activations onto the steering direction.
  - Adds back a controlled amount (`coefficient * normalized_vector`).
  - More stable than naive addition for strong steering.

## Verification Plan

### Manual Verification
1. **Quran Persona Test**:
   ```bash
   python3 main.py --quran-persona --interactive
   ```
   *   Verify that `vectors/quran_bge-m3_verse.npz`, `_paragraph.npz`, and `_surah.npz` are all loaded/generated.
   *   Prompt: "Who are you?" -> Expecting a response reflecting the Quranic character (guidance, wisdom).

2.  **Persona + MRA Combination**:
    ```bash
    python3 main.py --quran-persona --mra --interactive
    ```
    *   Verify both global steering (Persona) and dynamic context (MRA) are active.

3.  **Clamp Injection Mode Test**:
    ```bash
    python3 main.py --quran-persona --injection-mode clamp --coefficient 0.8 --interactive
    ```
    *   Verify that high-coefficient steering remains fluent and coherent.
    *   Compare with `--injection-mode add` to observe stability difference.
