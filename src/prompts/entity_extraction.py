"""
Quranic Entity Extraction Prompts

Specialized prompts for extracting Islamic concepts, prophets,
virtues, and relationships from Quranic text.
"""

ENTITY_EXTRACTION_PROMPT = """
You are an expert in Islamic studies and Quranic analysis. Your task is to extract
entities and relationships from Quranic text.

## Entity Types to Extract:

1. **PROPHET**: Prophets and messengers mentioned (e.g., Ibrahim, Musa, Isa, Muhammad)
2. **ANGEL**: Angels mentioned (e.g., Jibreel, Mikael, Israfil)
3. **VIRTUE**: Moral virtues and positive qualities (e.g., sabr/patience, shukr/gratitude, taqwa/God-consciousness)
4. **COMMAND**: Divine commands and obligations (e.g., prayer, charity, fasting)
5. **PROHIBITION**: Prohibited actions (e.g., lying, oppression, usury)
6. **CONCEPT**: Theological concepts (e.g., tawhid/monotheism, risalah/prophethood, akhirah/hereafter)
7. **EVENT**: Historical events (e.g., creation of Adam, flood of Nuh, exodus of Musa)
8. **PLACE**: Sacred or significant places (e.g., Makkah, Madinah, Mount Sinai)
9. **GROUP**: Groups of people (e.g., believers, disbelievers, people of the book)
10. **PRACTICE**: Religious practices (e.g., salah, sawm, hajj, dhikr)

## Relationship Types:

- exemplifies: Prophet X exemplifies Virtue Y
- teaches: Verse teaches Concept
- leads_to: Action leads to Consequence
- requires: Practice requires Condition
- contrasts_with: Virtue contrasts with Vice
- manifests_as: Concept manifests as Practice
- is_aspect_of: Specific is aspect of General
- practiced_by: Practice practiced by Group

## Input Text:
{text}

## Output Format:
Extract all entities and relationships in the following JSON format:

```json
{
  "entities": [
    {"name": "entity_name", "type": "ENTITY_TYPE", "arabic": "Arabic term if applicable", "description": "brief description"}
  ],
  "relationships": [
    {"source": "entity1", "target": "entity2", "relationship": "relationship_type", "description": "brief explanation"}
  ]
}
```

Focus on extracting meaningful, theologically significant entities and relationships.
Preserve Arabic terms where they carry important meaning (sabr, tawakkul, etc.).
"""


RELATIONSHIP_ENHANCEMENT_PROMPT = """
Given the following Quranic entities, identify meaningful relationships between them:

Entities:
{entities}

Context:
{context}

For each relationship, explain:
1. The nature of the connection
2. The Quranic basis for this relationship
3. The practical implication

Output relationships in JSON format:
```json
{
  "relationships": [
    {
      "source": "entity1",
      "target": "entity2",
      "relationship": "relationship_type",
      "quranic_basis": "verse reference or theme",
      "implication": "practical significance"
    }
  ]
}
```
"""


THEME_BRIDGING_PROMPT = """
You are helping connect modern concepts to Quranic themes.

User Query: {query}

Available Quranic Themes and Entities:
{available_themes}

Task: Identify which Quranic themes are most relevant to the user's query.
Explain the conceptual bridge between the modern context and Quranic wisdom.

Output format:
```json
{
  "bridges": [
    {
      "modern_concept": "concept from query",
      "quranic_theme": "relevant Quranic theme",
      "bridge_explanation": "how they connect",
      "confidence": 0.0-1.0
    }
  ]
}
```
"""
