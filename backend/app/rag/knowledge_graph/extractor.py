import logging
from copy import deepcopy
import pandas as pd
import dspy
from dspy.functional import TypedPredictor
from typing import Mapping, Optional, List, Dict, Any
from llama_index.core.schema import BaseNode

from app.rag.knowledge_graph.schema import (
    Entity,
    Relationship,
    KnowledgeGraph,
    EntityCovariateInput,
    EntityCovariateOutput,
)

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_relation_metadata_from_node(node: BaseNode) -> Mapping[str, str]:
    """Extracts and cleans metadata from a BaseNode."""
    metadata = deepcopy(node.metadata)
    keys_to_remove = [
        "_node_content",
        "_node_type",
        "excerpt_keywords",
        "questions_this_excerpt_can_answer",
        "section_summary",
    ]
    for key in keys_to_remove:
        metadata.pop(key, None)
    metadata["chunk_id"] = node.node_id
    logger.info(f"Extracted metadata from node '{node.node_id}': {metadata}")
    return metadata

# --- Extraction Agents ---

class HighLevelEntityRelationshipExtractionAgent(dspy.Signature):
    """
**High-Level Entity and Relationship Extraction**

**Objective**: Extract high-level entities (sections, chapters, paragraphs, sentences) and relationships between them, to build a broad, structural representation of the document’s content.


"""

    text = dspy.InputField(
        desc="The text from which to extract high-level entities."
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the high-level entities extracted from the text."
    )

class MidLevelEntityRelationshipExtractionAgent(dspy.Signature):
    """ Domain-Specific Entity and Relationship Extraction

Objective: Extract detailed, domain-specific entities  and their relationships, capturing the essential components and interactions in the  domain.

Instructions:

Exhaustive Domain-Specific Entity Extraction:

Identify all relevant domain-specific entities:

Extract Relationships Between Domain-Specific Entities:

Identify and define detailed relationships between domain entities:

Document Intermediate Reasoning:

For each identified relationship, explain why and how it exists in the context of the subject. Justify each dependency or containment relationship and how they contribute to the subject.
**Output**:
    - A JSON object containing 'entities' and 'relationships' extracted from the text.

Output Example:
```json
{
    "entities": [
        {"name": "Gserver Process", "type": "Component", "description": "Main server process managing client connections and executing queries"}
    ],
    "relationships": [
        {"source_entity": "Gserver Process", "target_entity": "Log Buffer", "relationship_desc": "writes transaction logs to"},
    ]
}

Goal: Capture domain-specific entities and their relationships to create a detailed subject subgraph

"""
    text = dspy.InputField(
        desc="The text from which to extract mid-level entities."
    )
    high_level_entities: List[Entity] = dspy.InputField(
        desc="List of high-level entities previously extracted."
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the mid-level entities extracted from the text."
    )

class LowLevelEntityRelationshipExtractionAgent(dspy.Signature):
    """
**Low-Level Entity and Relationship Extraction**

**Objective**: Extract highly detailed, granular entities and relationships specific to the subject domain, focusing on fine-grained elements.

**Instructions**:

1. **Granular Entity Extraction**:
    - **Identify low-level entities** related to the subject:


2. **Identify Specific Relationships Between Low-Level Entities**:
    - Focus on the detailed, system-level relationships between low-level entities
    
3. **Document Intermediate Reasoning**:
    - Provide a detailed justification for each entity and relationship identified:

      **Output**:
    - A JSON object containing 'entities' and 'relationships' extracted from the text.

**Output Example**:
```json
{
    "entities": [
        {"name": "Gserver Process", "type": "Process", "description": "Main process managing client connections and queries."},

    ],
    "relationships": [
        {"source_entity": "Gserver Process", "target_entity": "Log Writer Process", "relationship_desc": "writes transaction logs to"},

    ]
}
Goal: To produce a highly detailed, low-level subgraph that maps the fine-grained interactions and relationships within the subject. 

"""

    text = dspy.InputField(
        desc="The text from which to extract domain-specific entities."
    )
    mid_level_entities: List[Entity] = dspy.InputField(
        desc="List of mid-level entities previously extracted."
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the domain-specific entities extracted from the text."
    )

class CovariateExtractionAgent(dspy.Signature):
   
    """
**Objective**: Extract and link detailed covariates for each identified entity. These covariates should offer a comprehensive and precise summary of each entity's characteristics, ensuring factual accuracy and verifiability within the text.

**Instructions**:

1. **Review Provided Entities**:
    - Start by reviewing the list of **entities** already identified in the provided text. 
    - For each entity, extract **covariates**—attributes that provide additional context and detail about its characteristics and role in the subject.

2. **Extract Detailed Covariates**:


3. **Structure Covariates Clearly**:
    - Ensure each entity has a **topic** field as the first key in the covariate JSON structure, which should summarize the entity or its primary function.
    - Use a hierarchical structure where possible to represent nested attributes, related parameters, or grouped covariates.

4. **Link Covariates to Entities**:
    - **Ensure Each Covariate is Correctly Linked** to its corresponding entity.
    - Provide clarity by associating **each covariate** with its relevant **entity type** and **description**.

5. **Verification and Factual Integrity**:
    - Only include **covariates** that are directly extracted from the provided text. Avoid assumptions or relying on external sources.
    - Each covariate must be **factually accurate** and **verifiable** based on the provided source text.

6. **Comprehensive Attribute Coverage**:
    - Aim for **completeness** in covering all aspects of the entity’s attributes.
    - Ensure the extracted covariates provide **a thorough and precise understanding** of the entity’s characteristics and functionalities within the Subject system.

7. **Example Output**:
```json
{
    "topic": "Gserver Process",
    "Description": "The primary process responsible for handling client connections and executing queries.",
    "Dependencies": ["Gmaster Process", "Log Buffer", "Shared Pool"],
    "Operational States": ["Active", "Idle", "Busy"],
    "Performance Metrics": {
        "Current Load": "75%",
        "Average Response Time": "200ms"
    },
    "Configuration Parameters": {
        "Max Connections": 1000,
        "Listening Port": 1521,
        "Timeout Settings": "30s"
    },
    "Security": {
        "Encryption": "TLS-1.2",
        "Access Control": "Role-Based",
        "Permissions": ["Read", "Write"]
    }
}

    
    Goal:

The goal is to create a rich and detailed covariate structure for each entity, ensuring that every relevant attribute is captured and accurately linked to the correct entity. The covariates should help enrich the knowledge graph and provide a complete picture of each entity's role within the subject.
Final Note: Ensure all covariates extracted align with the domain and are directly supported by the text provided. The depth and detail of covariates should be sufficient to understand the entity in a real-world operational context. 
    """

    text = dspy.InputField(
        desc="The text from which to extract covariates."
    )
    entities: List[EntityCovariateInput] = dspy.InputField(
        desc="List of entities for which to extract covariates."
    )
    covariates: List[EntityCovariateOutput] = dspy.OutputField(
        desc="List of entities with extracted covariates."
    )

# --- Main Extractor Module ---

class Extractor(dspy.Module):
    """
    Orchestrates the extraction process across multiple levels of granularity to build a comprehensive knowledge graph.
    """

    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm

        # Initialize TypedPredictors for each agent
        self.high_level_extractor = TypedPredictor(HighLevelEntityRelationshipExtractionAgent)
        self.mid_level_extractor = TypedPredictor(MidLevelEntityRelationshipExtractionAgent)
        self.low_level_extractor = TypedPredictor(LowLevelEntityRelationshipExtractionAgent)
        self.covariate_extractor = TypedPredictor(CovariateExtractionAgent)

    def get_llm_output_config(self):
        if "openai" in self.dspy_lm.provider.lower():
            return {
                "response_format": {"type": "json_object"},
            }
        elif "ollama" in self.dspy_lm.provider.lower():
            return {}
        else:
            return {
                "response_mime_type": "application/json",
            }

    def forward(self, text):
        with dspy.settings.context(lm=self.dspy_lm):
            # Step 1: High-Level Entity Extraction
            pred_high_level = self.high_level_extractor(
                text=text,
                config=self.get_llm_output_config(),
            )
            logger.info("High-level entities extracted.")

            # # Step 2: Mid-Level Entity Extraction
            high_level_entities = pred_high_level.knowledge.entities
            # pred_mid_level = self.mid_level_extractor(
            #     text=text,
            #     high_level_entities=high_level_entities,
            #     config=self.get_llm_output_config(),
            # )
            # logger.info("Mid-level entities extracted.")

            # # Step 3: Low-Level Entity Extraction
            # mid_level_entities = pred_mid_level.knowledge.entities
            # pred_low_level = self.low_level_extractor(
            #     text=text,
            #     mid_level_entities=mid_level_entities,
            #     config=self.get_llm_output_config(),
            # )
            # logger.info("Low-level entities extracted.")

            # Combine entities from all levels
            all_entities = high_level_entities 
            # + mid_level_entities + pred_low_level.knowledge.entities
            all_relationships = (
                pred_high_level.knowledge.relationships 
                # +
                # pred_mid_level.knowledge.relationships +
                # pred_low_level.knowledge.relationships
            )

            # Step 4: Covariate Extraction
            entities_for_covariates = [
                EntityCovariateInput(
                    name=entity.name,
                    description=entity.description,
                )
                for entity in all_entities
            ]

            pred_covariates = self.covariate_extractor(
                text=text,
                entities=entities_for_covariates,
                config=self.get_llm_output_config(),
            )
            logger.info("Covariates extracted.")

            # Update entities with covariates
            for entity in all_entities:
                for covariate in pred_covariates.covariates:
                    if entity.name == covariate.name:
                        entity.metadata = covariate.covariates

            # Construct the final knowledge graph
            knowledge_graph = KnowledgeGraph(
                entities=all_entities,
                relationships=all_relationships,
            )

            return knowledge_graph

# --- Simple Graph Extractor Class ---

class SimpleGraphExtractor:
    """
    Interface for executing the extraction process and converting results into DataFrames.
    """

    def __init__(
        self,
        dspy_lm: dspy.LM,
        compiled_extract_program_path: Optional[str] = None
    ):
        self.extractor = Extractor(dspy_lm=dspy_lm)
        if compiled_extract_program_path is not None:
            self.extractor.load(compiled_extract_program_path)
            logger.info(f"Loaded compiled extraction program from '{compiled_extract_program_path}'.")

    def extract(self, text: str, node: BaseNode) -> (pd.DataFrame, pd.DataFrame):
        """
        Executes the extraction process and returns DataFrames for entities and relationships.
        """
        try:
            knowledge_graph = self.extractor.forward(text=text)
            logger.info("Knowledge graph extraction successful.")
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise e

        metadata = get_relation_metadata_from_node(node)
        entities_df, relationships_df = self._to_df(
            knowledge_graph.entities, knowledge_graph.relationships, metadata
        )
        logger.info("Converted knowledge graph to DataFrames.")
        return entities_df, relationships_df

    def _to_df(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        extra_meta: Mapping[str, str],
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Converts lists of entities and relationships into pandas DataFrames.
        """
        # Process entities
        entities_data = []
        for entity in entities:
            entity_dict = {
                "name": entity.name,
                "description": entity.description,
                "meta": entity.metadata
            }
            entities_data.append(entity_dict)

        entities_df = pd.DataFrame(entities_data)
        logger.info(f"Entities DataFrame created with {len(entities_df)} records.")

        # Map entity names to their descriptions
        mapped_entities = {entity["name"]: entity for entity in entities_data}

        # Process relationships
        relationships_data = []
        for relationship in relationships:
            source_entity_desc = mapped_entities.get(relationship.source_entity, {}).get("description", "")
            target_entity_desc = mapped_entities.get(relationship.target_entity, {}).get("description", "")

            relationship_dict = {
                "source_entity": relationship.source_entity,
                "source_entity_description": source_entity_desc,
                "target_entity": relationship.target_entity,
                "target_entity_description": target_entity_desc,
                "relationship_desc": relationship.relationship_desc,
                "meta": deepcopy(extra_meta)
            }
            relationships_data.append(relationship_dict)

        relationships_df = pd.DataFrame(relationships_data)
        logger.info(f"Relationships DataFrame created with {len(relationships_df)} records.")

        return entities_df, relationships_df

