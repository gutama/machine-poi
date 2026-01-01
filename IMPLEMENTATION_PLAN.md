# Implementation Plan - Quranic MRA & Multidomain Thinking

The goal is to move beyond static activation steering towards a system where the Quran actively shapes the "thinking" and "analogy" process of the LLM using **Multi-Resolution Analysis (MRA)** and **Multidomain** mapping, powered by **ChromaDB**.

## User Review Required

> [!IMPORTANT]
> This plan introduces a dependency on `chromadb`.
> MRA is defined here as retrieving and synthesizing from multiple granularities (Verse Level, Passage Level, Theme Level).

## Proposed Changes

### 1. Database & Indexing (ChromaDB)
We will replace the simple numpy cache with ChromaDB to manage multiple collections (Resolutions).

#### [NEW] [src/knowledge_base.py](file:///home/ginanjar/repositories/machine-poi/src/knowledge_base.py)
This new module will handle ChromaDB interactions.
- **Collections**:
    - `quran_verses`: Fine-grained (Micro resolution).
    - `quran_passages`: Thematic groups of verses (Meso resolution).
    - `quran_surahs`: Chapter summaries/intros (Macro resolution).
- **Methods**:
    - `initialize_db()`: Chunk text at different levels and embed.
    - `multiresolution_query(query, n_results)`: Query all collections simultaneously.

#### [MODIFY] [requirements.txt](file:///home/ginanjar/repositories/machine-poi/requirements.txt)
- Add `chromadb`.

### 2. Multidomain Analogy Logic
We want to map user queries (e.g., "coding bug") to Quranic domains (e.g., "patience", "correction").

#### [MODIFY] [src/steerer.py](file:///home/ginanjar/repositories/machine-poi/src/steerer.py)
- **Domain Shifting**:
    - Before querying Chroma, the LLM (or a heuristic) generates "domain bridges".
    - Example: User "Fixing a bug" -> Domain Bridge "Correction and Improvement".
    - Query Chroma with both original and bridge terms.
- **Dynamic Steering**:
    - Compute steering vectors from the *weighted average* of retrieved Verse and Passage embeddings.

### 3. Explicit Reasoning (MRA Style)
The "Chain of Thought" prompt will be structured to force Multi-Resolution thinking.

#### [MODIFY] [src/steerer.py](file:///home/ginanjar/repositories/machine-poi/src/steerer.py)
- **Prompt Template**:
    > **Context (Micro-Verse)**: {verse_content}
    > **Context (Macro-Theme)**: {passage_content}
    > **Task**: {user_prompt}
    > **Instruction**: Perform a Multi-Resolution Analysis.
    > 1. Matches: How does the specific verse apply?
    > 2. Themes: How does the general theme apply?
    > 3. Multidomain Analogy: Draw an analogy between the Quranic principle (e.g., generic patience) and the specific user domain (e.g., debugging code).
    > 4. Synthesis: Answer the user.

### 4. CLI Updates

#### [MODIFY] [main.py](file:///home/ginanjar/repositories/machine-poi/main.py)
- Add `init-db` command to build ChromaDB.
- Update `generate` to support `--mra` flag.

## Verification Plan

### Automated Tests
- Test `knowledge_base.py` stores and retrieves from multiple collections.
- Verify `multiresolution_query` returns mixed results (verses and passages).

### Manual Verification
1.  Run `python main.py init-db`
2.  Run `python main.py --interactive --mra`
3.  Prompt: "My team is arguing about code style."
4.  Expectation:
    -   **Retrieval**: Verses on "Reconciliation" (Surah Hujurat) + Passages on "Unity".
    -   **Reasoning**: "Just as the Quran advises making peace between brothers (Micro), and holding fast to the rope of Allah (Macro), you should unify your team on a shared standard..."
