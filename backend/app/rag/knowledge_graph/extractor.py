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
    logger.debug(f"Extracted metadata from node '{node.node_id}': {metadata}")
    return metadata

# --- Extraction Agents ---

class HighLevelEntityRelationshipExtractionAgent(dspy.Signature):
    """
**High-Level Entity and Relationship Extraction**

**Objective**: Extract high-level entities (sections, chapters, paragraphs, sentences) and relationships between them, to build a broad, structural representation of the document’s content.

**Instructions**:

1. **Extract High-Level Entities**:
    - Identify the **top-level entities** in the document, such as:
      - **Sections**: Core sections that divide the document into broad themes.
      - **Chapters**: Major subdivisions under each section, typically containing multiple topics.
      - **Paragraphs**: Smaller subdivisions within chapters, often discussing one concept or element.
      - **Sentences**: The smallest unit that conveys individual thoughts or points.

2. **Extract Relationships Between High-Level Entities**:
    - Identify the relationships between these entities:
      - **Containment**: Sections contain chapters, chapters contain paragraphs, paragraphs contain sentences.
      - **Dependency**: Some sections or chapters depend on others for logical continuity or context.
      - **Hierarchical**: Sections are higher-level entities, and chapters or paragraphs are sub-levels, reflecting the document's structure.

3. **Document Intermediate Reasoning**:
    - Provide reasoning behind how these entities and relationships are identified. Explain why certain sections contain specific chapters, or why one paragraph depends on another.    
    
**Output**:
    - A JSON object containing 'entities' and 'relationships' extracted from the text.
    
**Output Example**:
```json
{
    "entities": [
        {"name": "Chapter 1", "type": "Chapter", "description": "Introduction to SunDB architecture"},
        {"name": "Section 1.1", "type": "Section", "description": "Overview of SunDB components"},
        {"name": "Paragraph 1", "type": "Paragraph", "description": "Describes SunDB’s core processes and components"}
    ],
    "relationships": [
        {"source_entity": "Chapter 1", "target_entity": "Section 1.1", "relationship_desc": "contains"},
        {"source_entity": "Section 1.1", "target_entity": "Paragraph 1", "relationship_desc": "contains"}
    ]
}
Goal: Provide a broad structural overview of the document, identifying the top-level entities and their relationships to form a global subgraph.



"""

    text = dspy.InputField(
        desc="The text from which to extract high-level entities."
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the high-level entities extracted from the text."
    )

class MidLevelEntityRelationshipExtractionAgent(dspy.Signature):
    """ Domain-Specific Entity and Relationship Extraction (SunDB)

Objective: Extract detailed, domain-specific entities (e.g., processes, configurations, components) and their relationships, capturing the essential components and interactions in the SunDB domain.

Instructions:

Exhaustive Domain-Specific Entity Extraction:

Identify all relevant domain-specific entities related to SunDB, including but not limited to:
Components: Processes like Gserver, Gmaster, and Log Buffer.
Configurations: System parameters like BUFFER_CACHE_SIZE, LOG_BUFFER_SIZE, Shared Pool.
Data Structures: Objects such as Tablespaces, Control Files, Data Files.
Operations: Specific actions or commands like SQL Statements, Backup Commands, Restore Procedures.
States: Operational states like Active, Inactive, Pending.
Security: Elements such as Roles, Permissions, and Access Control.
Extract Relationships Between Domain-Specific Entities:

Identify and define detailed relationships between domain entities:
Dependency: E.g., "Gserver depends on Log Buffer for transaction consistency".
Containment: E.g., "Shared Pool contains memory buffers".
Operational Flow: E.g., "Log Buffer receives logs from Gserver Process".
Interconnection: E.g., "Tablespace connects to Data Files".
Document Intermediate Reasoning:

For each identified relationship, explain why and how it exists in the context of SunDB. Justify each dependency or containment relationship and how they contribute to SunDB’s operation.
**Output**:
    - A JSON object containing 'entities' and 'relationships' extracted from the text.

Output Example:
```json
{
    "entities": [
        {"name": "Gserver Process", "type": "Component", "description": "Main server process managing client connections and executing queries"},
        {"name": "Log Buffer", "type": "Data Structure", "description": "Buffer that stores transaction logs to ensure data consistency"},
        {"name": "BUFFER_CACHE_SIZE", "type": "Configuration", "description": "Defines the size of the buffer cache in memory"}
    ],
    "relationships": [
        {"source_entity": "Gserver Process", "target_entity": "Log Buffer", "relationship_desc": "writes transaction logs to"},
        {"source_entity": "Log Buffer", "target_entity": "BUFFER_CACHE_SIZE", "relationship_desc": "affected by"}
    ]
}

Goal: Capture domain-specific entities and their relationships to create a detailed SunDB subgraph, representing the architecture, configurations, and processes.

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
**Low-Level Entity and Relationship Extraction for SunDB**

**Objective**: Extract highly detailed, granular entities and relationships specific to the SunDB domain, focusing on fine-grained elements like internal processes, memory allocations, and intricate system-level interactions.

**Instructions**:

1. **Granular Entity Extraction**:
    - **Identify low-level entities** related to SunDB operations and configurations:
      - **Processes and Threads**: Specific processes like **Gserver**, **Gmaster**, and threads like **Log Writer**, **Checkpoint Process**.
      - **Memory Allocations**: Detailed memory structures such as **Buffer Cache**, **Shared Pool**, **Log Buffer**.
      - **File Structures**: Identify **data files**, **control files**, **log files**, **archive logs**, **temporary files**, etc.
      - **Execution Units**: Extract details about **SQL Execution Plans**, **Transaction Logs**, and specific database commands.
      - **Network Entities**: Connections, protocols, **listener configurations**, **ports**, **IP addresses**.
      - **Runtime Metrics**: Metrics like **CPU Utilization**, **Disk I/O**, **Query Execution Time**, **Response Times**.

2. **Identify Specific Relationships Between Low-Level Entities**:
    - Focus on the detailed, system-level relationships between low-level entities, such as:
      - **Memory Allocation Dependencies**: E.g., "Buffer Cache relies on **Shared Pool** for memory management".
      - **Process-File Interactions**: E.g., "Gserver Process writes to **Data Files**".
      - **File-System Interactions**: E.g., "Transaction Logs are archived to **Log Archive**".
      - **System Resource Relationships**: E.g., "Log Writer Process interacts with **Disk I/O** for log writing".
      - **Performance Dependencies**: E.g., "Buffer Cache Size impacts **Query Execution Time**".
    
3. **Document Intermediate Reasoning**:
    - Provide a detailed justification for each entity and relationship identified:
      - Explain why a certain **process** requires specific **memory allocations** or why an **SQL query** depends on certain **data files**.
      - Describe how **network protocols** are linked to **connection strings** and **port configurations**.
      - Clarify why certain **system metrics** are directly tied to performance (e.g., "CPU utilization increases with **heavy disk I/O**").

      **Output**:
    - A JSON object containing 'entities' and 'relationships' extracted from the text.

**Output Example**:
```json
{
    "entities": [
        {"name": "Gserver Process", "type": "Process", "description": "Main process managing client connections and queries."},
        {"name": "Log Writer Process", "type": "Process", "description": "Responsible for writing transaction logs to disk."},
        {"name": "Buffer Cache", "type": "Memory Allocation", "description": "Cache memory that stores recently used data blocks."},
        {"name": "Data File", "type": "File", "description": "File containing database records for permanent storage."}
    ],
    "relationships": [
        {"source_entity": "Gserver Process", "target_entity": "Log Writer Process", "relationship_desc": "writes transaction logs to"},
        {"source_entity": "Log Writer Process", "target_entity": "Disk I/O", "relationship_desc": "writes log data to"},
        {"source_entity": "Buffer Cache", "target_entity": "Shared Pool", "relationship_desc": "depends on for memory management"},
        {"source_entity": "Data File", "target_entity": "Disk I/O", "relationship_desc": "interacts with for data retrieval"}
    ]
}
Goal: To produce a highly detailed, low-level subgraph that maps the fine-grained interactions and relationships within the SunDB system. This level of granularity allows for precise tracking of operational and resource dependencies, system behavior, and performance metrics at runtime.

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
**Objective**: Extract and link detailed covariates for each identified entity in the provided SunDB documentation. These covariates should offer a comprehensive and precise summary of each entity's characteristics, ensuring factual accuracy and verifiability within the text.

**Instructions**:

1. **Review Provided Entities**:
    - Start by reviewing the list of **entities** already identified in the provided text. These entities may represent system components, processes, configurations, or any other detailed elements related to SunDB.
    - For each entity, extract **covariates**—attributes that provide additional context and detail about its characteristics and role in the system.

2. **Extract Detailed Covariates**:
    - **Technical Specifications**:
      - Extract details such as **sizes**, **capacities**, **default values**, **allowable ranges**, **data types**, and **units of measurement**.
    - **Configurations and Settings**:
      - Extract specific **configuration parameters**, **operational modes**, **policies**, and **strategies** related to the entity.
    - **Operational Details**:
      - Identify **states**, **statuses**, **behaviors**, **lifecycle stages**, and **runtime metrics** that describe the entity’s functioning.
    - **Dependencies and Interactions**:
      - Detail **dependencies** the entity has on other system components, such as memory, processes, files, or external resources.
    - **Functional Descriptions**:
      - Capture the **primary roles**, **responsibilities**, and **functions** the entity performs within SunDB.
    - **Security and Permissions**:
      - For entities involved in access control or data protection, extract relevant details about **encryption methods**, **access control models**, and **security policies**.
    - **Performance Metrics**:
      - Extract metrics such as **throughput**, **latency**, **resource utilization**, and **optimization settings** associated with the entity’s performance.
    - **Metadata**:
      - Identify and include **version numbers**, **timestamps**, **authors**, and any other metadata relevant to the entity.

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
    - Ensure the extracted covariates provide **a thorough and precise understanding** of the entity’s characteristics and functionalities within the SunDB system.

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

The goal is to create a rich and detailed covariate structure for each entity, ensuring that every relevant attribute is captured and accurately linked to the correct entity. The covariates should help enrich the knowledge graph and provide a complete picture of each entity's functionality, interactions, and properties within the SunDB system.
Final Note: Ensure all covariates extracted align with the SunDB domain and are directly supported by the text provided. The depth and detail of covariates should be sufficient to understand the entity in a real-world operational context. 
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

            # Step 2: Mid-Level Entity Extraction
            high_level_entities = pred_high_level.knowledge.entities
            pred_mid_level = self.mid_level_extractor(
                text=text,
                high_level_entities=high_level_entities,
                config=self.get_llm_output_config(),
            )
            logger.info("Mid-level entities extracted.")

            # Step 3: Low-Level Entity Extraction
            mid_level_entities = pred_mid_level.knowledge.entities
            pred_low_level = self.low_level_extractor(
                text=text,
                mid_level_entities=mid_level_entities,
                config=self.get_llm_output_config(),
            )
            logger.info("Low-level entities extracted.")

            # Combine entities from all levels
            all_entities = high_level_entities + mid_level_entities + pred_low_level.knowledge.entities
            all_relationships = (
                pred_high_level.knowledge.relationships +
                pred_mid_level.knowledge.relationships +
                pred_low_level.knowledge.relationships
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
        logger.debug(f"Entities DataFrame created with {len(entities_df)} records.")

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
        logger.debug(f"Relationships DataFrame created with {len(relationships_df)} records.")

        return entities_df, relationships_df

