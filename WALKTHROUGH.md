# Quran Persona Walkthrough

We have implemented a **Quran Persona** mode that aggregates embeddings from the entire Quran across multiple resolutions (Verse, Passage, and Surah) to create a comprehensive "character" vector for steering.

## Changes
- **Modified `src/steerer.py`**: Added `prepare_quran_persona()` which:
    - Loads/Generates embeddings for all resolutions: `verse`, `paragraph`, `surah`.
    - Computes a global mean "Persona Embedding".
    - Projects this into a steering vector.
- **Modified `main.py`**: Added `--quran-persona` CLI flag.

## How to Use

Run the following command to activate the Quran Persona:

```bash
python3 main.py --quran-persona --interactive
```

This will:
1.  Load the LLM and Embedding models.
2.  Generate/Load embeddings for Verses, Passages, and Surahs.
3.  Construct the unified "Quran Persona" vector.
4.  Start an interactive chat where the LLM is steered by this persona.

## Verification
We verified that the application successfully:
- Detects the `--quran-persona` flag.
- Loads embeddings for all three resolutions.
- Prints `Quran Persona activated.`
