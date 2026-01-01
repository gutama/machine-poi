# Implementation Plan - Quran Persona Integration

We will implement a "Quran Persona" mode that aggregates **all** embedding vectors (Verse, Passage, and Surah levels) to construct a comprehensive "Quran Character" vector. This adapts the "Persona Vector" concept (consistent character steering) using the rich semantic data of the entire Quran.

## User Review Required

> [!NOTE]
> This replaces the generic `--persona` proposal. We will create a single, robust "Quran Persona" by fusing embeddings from all resolutions.

## Proposed Changes

### [src/steerer.py](file:///home/ginanjar/repositories/machine-poi/src/steerer.py)

#### [MODIFY] `QuranSteerer` class
- **Add method**: `prepare_quran_persona()`
    - Loads embeddings for ALL resolutions: `verse`, `paragraph` (passage), and `surah`.
    - Computes a global weighted mean of all these embeddings to capture the "Holographic" character of the Quran.
    - Projects this "Persona Embedding" to a steering vector.
    - Applies this vector globally.

### [main.py](file:///home/ginanjar/repositories/machine-poi/main.py)

#### [MODIFY] CLI Arguments
- Add `--quran-persona` flag.
- When enabled alongside `--mra`, the "Persona" provides the global character, while MRA provides local context steering.

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
