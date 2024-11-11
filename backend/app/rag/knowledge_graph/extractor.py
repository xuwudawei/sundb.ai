
import logging
from copy import deepcopy
import pandas as pd
import dspy
from dspy.functional import TypedPredictor
from typing import Mapping, Optional, List
from llama_index.core.schema import BaseNode

from app.rag.knowledge_graph.schema import (
    Entity,
    Relationship,
    KnowledgeGraph,
    EntityCovariateInput,
    EntityCovariateOutput,
)

logger = logging.getLogger(__name__)


class ExtractGraphTriplet(dspy.Signature):
    """**Objective**: Extract as many entities and relationships as possible from the provided SunDB documentation text to construct a comprehensive and detailed knowledge graph that captures the full breadth and intricacies of SunDB's architecture, components, configurations, and functionalities.

        **Instructions**:

        1. **Exhaustive Entity Extraction**:
        - **Identify All Relevant Entities**:
            - Scrutinize the text to extract all possible entities, including but not limited to:
            - **Components**: Hardware components, software modules, processes, threads, services, daemons within SunDB.
            - **Configurations and Parameters**: System settings, configuration files, parameters, environment variables, default values, and tunable settings.
            - **Data Structures**: Tablespaces, control files, log files, data files, indexes, buffers, caches, memory pools, schemas, and database objects.
            - **Operations and Commands**: Administrative commands, SQL statements, operational modes (e.g., ARCHIVELOG mode), utilities, and scripts.
            - **Events and States**: Error codes, statuses, operational states (e.g., Active, Inactive, Pending), events, and triggers.
            - **Concepts and Terminologies**: Replication strategies, failover mechanisms, backup methods, sharding techniques, consistency models, and any domain-specific terminologies.
            - **Users and Roles**: User accounts, roles, permissions, and authentication methods.
            - **Networks and Connections**: Network interfaces, protocols, ports, connection strings, and communication channels.
        - **Use Specific and Descriptive Names**:
            - Ensure each entity has a unique and descriptive name that reflects its role and purpose within SunDB.
            - Avoid generic terms; prefer specific identifiers (e.g., "Gserver Process" instead of "Process").

        - **正例子 (Positive Example)**:
            ```json
            {
                "name": "BUFFER_CACHE_SIZE",
                "type": "Parameter",
                "attributes": {
                    "topic": "Memory Configuration Parameter",
                    "Default Value": "256MB",
                    "Allowed Range": "128MB - 1024MB",
                    "Description": "Specifies the size of the buffer cache in memory.",
                    "Related Parameters": ["SHARED_POOL_SIZE", "LOG_BUFFER_SIZE"],
                    "Impact": "Affects the amount of data that can be cached in memory, influencing performance."
                }
            }
            ```

        - **反例子 (Negative Example)**:
            ```json
            {
                "name": "Buffer Size",
                "type": "Parameter",
                "attributes": {
                    "topic": "Memory Parameter",
                    "Default Value": "Unknown",
                    "Allowed Range": "None",
                    "Description": "Unknown impact",
                    "Related Parameters": [],
                    "Impact": "None"
                }
            }
            ```
            - **Explanation**: The above is a poor example because the name "Buffer Size" is too generic, and the attributes lack specificity (e.g., "Unknown" values and non-descriptive impact).

        2. **Comprehensive Covariate Extraction for Entities**:
        - **Populate `attributes` with Detailed Covariates**:
            - For each entity, extract all relevant covariates and include them in the `attributes` field as a nested JSON object.
            - **Include**:
            - **Technical Specifications**: Sizes, capacities, limits, thresholds, versions, and performance metrics.
            - **Configurations**: Settings, modes, policies, strategies, parameter values, and configurations.
            - **Dependencies and Interactions**: Other entities that this entity depends on, interacts with, or is associated with.
            - **Functional Descriptions**: Roles, responsibilities, behaviors, algorithms implemented, and operational logic.
            - **Default Values and Ranges**: Any default settings, acceptable value ranges, and configurable options.
            - **Metadata**: Creation dates, authors, last modified times, and other relevant metadata.

        - **正例子 (Positive Example)**:
            ```json
            {
                "topic": "Gserver Process",
                "Description": "Primary server process handling client connections and query execution.",
                "Dependencies": ["Gmaster Process", "Log Buffer", "Shared Pool"],
                "Configuration Parameters": {
                    "Max Connections": 1000,
                    "Listening Port": 1521,
                    "Timeout Settings": "30s"
                },
                "Operational States": ["Active", "Idle", "Busy"],
                "Performance Metrics": {
                    "Current Load": "75%",
                    "Average Response Time": "200ms"
                }
            }
            ```

        - **反例子 (Negative Example)**:
            ```json
            {
                "topic": "Process",
                "Description": "Handles requests",
                "Dependencies": [],
                "Configuration Parameters": {},
                "Operational States": ["Idle"],
                "Performance Metrics": {}
            }
            ```
            - **Explanation**: This example lacks sufficient details, such as dependencies, specific configurations, and performance metrics, making it incomplete.

        3. **Detailed Relationship Extraction**:
        - **Identify All Possible Relationships Between Entities**:
            - Capture every possible relationship, including:
            - **Dependencies**: Which entities rely on others to function.
            - **Hierarchies**: Parent-child relationships, ownerships, containment.
            - **Interactions**: Communication paths, data flows, synchronization mechanisms.
            - **Associations**: Groupings, categorizations, affiliations.
            - **Sequences and Workflows**: Order of operations, execution sequences.
            - **Equivalencies and Aliases**: Entities that are equivalent or have aliases.
        - **Use Clear and Descriptive `relationship_desc`**:
            - Provide detailed descriptions of how entities are related.
            - Use action verbs and phrases that specify the nature of the relationship.

        - **正例子 (Positive Example)**:
            ```json
            {
                "source_entity": "Gserver Process",
                "target_entity": "Log Buffer",
                "relationship_desc": "writes transaction logs to"
            }
            ```

        - **反例子 (Negative Example)**:
            ```json
            {
                "source_entity": "Process",
                "target_entity": "Memory",
                "relationship_desc": "works with"
            }
            ```
            - **Explanation**: The description "works with" is too vague. Instead, a more specific relationship (e.g., "allocates memory for") would be more accurate.

        4. **Verification and Accuracy**:
        - Ensure that all required fields (e.g., `name`, `type`, `attributes`, `source_entity`, `target_entity`, `relationship_desc`) are populated correctly.
        - Avoid redundancies and inconsistencies by consolidating similar entities.
        - Ensure factual integrity by extracting only the information present in the provided text, not assumptions or external knowledge.

    **Goal**: Produce a deeply structured, index-ready JSON object representing the SunDB knowledge graph, capturing all possible entities and their interrelationships to build the most comprehensive knowledge graph ever built for SunDB.
    """

    text = dspy.InputField(
        desc="A paragraph of SunDB documentation text to extract entities and relationships from"
    )
    knowledge: KnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the knowledge extracted from the text."
    )


class ExtractCovariate(dspy.Signature):
    """**Objective**: From the provided SunDB documentation text and the list of identified entities, extract as many detailed covariates as possible for each entity to create a complete and rich JSON representation of each entity's attributes.

    **Instructions**:

    1. **Exhaustive Covariate Identification**:
       - **For Each Entity**:
         - Extract all possible covariates related to:
           - **Technical Specifications**: Sizes, capacities, default values, allowable ranges, data types.
           - **Configurations and Settings**: Modes, policies, strategies, parameter values, configuration options.
           - **Operational Details**: States, statuses, behaviors, processes, lifecycle stages.
           - **Dependencies and Interactions**: Other entities it interacts with, depends on, or affects.
           - **Functional Descriptions**: Roles, responsibilities within SunDB, algorithms implemented.
           - **Security and Permissions**: Access controls, authentication methods, encryption settings.
           - **Performance Metrics**: Throughput, latency, resource utilization, optimization settings.
           - **Metadata**: Version numbers, authorship, timestamps, identifiers.
       - **Multiple Levels of Detail**:
         - Include both high-level overviews and low-level specifics.
         - Capture nested attributes and hierarchies where applicable.

    2. **Structured JSON Representation**:
       - **Use `"topic"` Field**:
         - Start the `attributes` JSON with a `"topic"` field summarizing the entity or its primary function.
       - **Organize Covariates Clearly**:
         - Use clear and consistent key-value pairs for each attribute.
         - Group related attributes under subtopics or nested objects if necessary.
       - **正例子 (Positive Example)**:
         ```json
         {
           "topic": "Gserver Process",
           "Description": "Primary server process handling client connections and query execution.",
           "Dependencies": ["Gmaster Process", "Log Buffer", "Shared Pool"],
           "Configuration Parameters": {
             "Max Connections": 1000,
             "Listening Port": 1521,
             "Timeout Settings": "30s"
           },
           "Operational States": ["Active", "Idle", "Busy"],
           "Performance Metrics": {
             "Current Load": "75%",
             "Average Response Time": "200ms"
           }
         }
         ```

       - **反例子 (Negative Example)**:
         ```json
         {
           "topic": "Process",
           "Description": "Handles requests",
           "Dependencies": [],
           "Configuration Parameters": {},
           "Operational States": ["Idle"],
           "Performance Metrics": {}
         }
         ```
         - **Explanation**: This example is incomplete. It lacks sufficient detail, such as specific dependencies, configurations, and performance metrics. It uses a very vague "Process" description, which doesn't describe its role properly in the context of SunDB.

    3. **Verification and Accuracy**:
       - **Link Covariates to Correct Entities**:
         - Ensure that each covariate is accurately associated with its corresponding entity.
       - **Factual Integrity**:
         - Only include covariates that are verifiable within the provided text.
         - Do not include assumptions or information not present in the text.

    4. **Comprehensive Attribute Coverage**:
       - **Aim for Completeness**:
         - The collection of covariates should provide a full understanding of the entity's characteristics and functionalities.
       - **Detail Operational Intricacies**:
         - Include unique operational details, configurations, and any special behaviors specific to SunDB.

    5. **Enhance Knowledge Graph Richness**:
       - **Maximize Detail**:
         - Provide as much detail as possible to enrich the knowledge graph.
       - **Ensure Consistency and Clarity**:
         - Use consistent terminology and formatting for easier integration into the knowledge graph.

    **Goal**: Provide a detailed and precise summary of each entity's characteristics as described in the source material, enhancing the richness and depth of the knowledge graph.

    **Instructions Continued**:

    - **Additional Positive Example (正例子)**:
      This is an expanded example that demonstrates a more complex entity with various covariates across multiple categories:
      ```json
      {
        "topic": "Data File",
        "Description": "Represents a physical file that stores user data in SunDB.",
        "Size": "50GB",
        "File Type": "Binary",
        "Creation Date": "2024-01-15",
        "Last Modified": "2024-10-22",
        "Dependencies": ["Tablespace", "Control File"],
        "Operational States": ["Active", "Archived"],
        "Performance Metrics": {
          "Read Speed": "500MB/s",
          "Write Speed": "300MB/s"
        },
        "Security": {
          "Encryption": "AES-256",
          "Access Control": "Role-Based",
          "Permissions": ["Read", "Write"]
        }
      }
      ```

    - **Additional Negative Example (反例子)**:
      This is a poor example because it contains too little information and lacks meaningful attributes:
      ```json
      {
        "topic": "File",
        "Description": "A file.",
        "Size": "Unknown",
        "File Type": "Unknown",
        "Creation Date": "Unknown",
        "Operational States": ["Inactive"],
        "Performance Metrics": {}
      }
      ```
      - **Explanation**: This example is missing critical details such as file size, type, creation date, and operational metrics. The attributes are vague, and it doesn't provide enough value for constructing a knowledge graph.

    **Goal**: The goal here is to create a well-structured, detailed, and accurate representation of each entity's attributes. The richer and more complete the covariates, the more effective the knowledge graph will be for representing SunDB's architecture and components.

    """


    text = dspy.InputField(
        desc="A paragraph of SunDB documentation text to extract covariates from."
    )

    entities: List[EntityCovariateInput] = dspy.InputField(
        desc="List of entities identified in the text."
    )
    covariates: List[EntityCovariateOutput] = dspy.OutputField(
        desc="Graph representation of the knowledge extracted from the text."
    )


def get_relation_metadata_from_node(node: BaseNode):
    metadata = deepcopy(node.metadata)
    for key in [
        "_node_content",
        "_node_type",
        "excerpt_keywords",
        "questions_this_excerpt_can_answer",
        "section_summary",
    ]:
        metadata.pop(key, None)
    metadata["chunk_id"] = node.node_id
    return metadata


class Extractor(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog_graph = TypedPredictor(ExtractGraphTriplet)
        self.prog_covariates = TypedPredictor(ExtractCovariate)

    def get_llm_output_config(self):
        if "openai" in self.dspy_lm.provider.lower():
            return {
                "response_format": {"type": "json_object"},
            }
        elif "ollama" in self.dspy_lm.provider.lower():
            # ollama support set format=json in the top-level request config, but not in the request's option
            # https://github.com/ollama/ollama/blob/5e2653f9fe454e948a8d48e3c15c21830c1ac26b/api/types.go#L70
            return {}
        else:
            return {
                "response_mime_type": "application/json",
            }

    def forward(self, text):
        with dspy.settings.context(lm=self.dspy_lm):
            pred_graph = self.prog_graph(
                text=text,
                config=self.get_llm_output_config(),
            )

            # extract the covariates
            entities_for_covariates = [
                EntityCovariateInput(
                    name=entity.name,
                    description=entity.description,
                )
                for entity in pred_graph.knowledge.entities
            ]

            pred_covariates = self.prog_covariates(
                text=text,
                entities=entities_for_covariates,
                config=self.get_llm_output_config(),
            )

            # replace the entities with the covariates
            for entity in pred_graph.knowledge.entities:
                for covariate in pred_covariates.covariates:
                    if entity.name == covariate.name:
                        entity.metadata = covariate.covariates

            return pred_graph


class SimpleGraphExtractor:
    def __init__(
        self, dspy_lm: dspy.LM, complied_extract_program_path: Optional[str] = None
    ):
        self.extract_prog = Extractor(dspy_lm=dspy_lm)
        if complied_extract_program_path is not None:
            self.extract_prog.load(complied_extract_program_path)

    def extract(self, text: str, node: BaseNode):
        pred = self.extract_prog(text=text)
        metadata = get_relation_metadata_from_node(node)
        return self._to_df(
            pred.knowledge.entities, pred.knowledge.relationships, metadata
        )

    def _to_df(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        extra_meta: Mapping[str, str],
    ):
        # Create lists to store dictionaries for entities and relationships
        entities_data = []
        relationships_data = []

        # Iterate over parsed entities and relationships to create dictionaries
        for entity in entities:
            entity_dict = {
                "name": entity.name,
                "description": entity.description,
                "meta": entity.metadata,
            }
            entities_data.append(entity_dict)

        mapped_entities = {entity["name"]: entity for entity in entities_data}

        for relationship in relationships:
            source_entity_description = ""
            if relationship.source_entity not in mapped_entities:
                new_source_entity = {
                    "name": relationship.source_entity,
                    "description": (
                        f"Derived from from relationship: "
                        f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                    ),
                    "meta": {"status": "need-revised"},
                }
                entities_data.append(new_source_entity)
                mapped_entities[relationship.source_entity] = new_source_entity
                source_entity_description = new_source_entity["description"]
            else:
                source_entity_description = mapped_entities[relationship.source_entity][
                    "description"
                ]

            target_entity_description = ""
            if relationship.target_entity not in mapped_entities:
                new_target_entity = {
                    "name": relationship.target_entity,
                    "description": (
                        f"Derived from from relationship: "
                        f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                    ),
                    "meta": {"status": "need-revised"},
                }
                entities_data.append(new_target_entity)
                mapped_entities[relationship.target_entity] = new_target_entity
                target_entity_description = new_target_entity["description"]
            else:
                target_entity_description = mapped_entities[relationship.target_entity][
                    "description"
                ]

            relationship_dict = {
                "source_entity": relationship.source_entity,
                "source_entity_description": source_entity_description,
                "target_entity": relationship.target_entity,
                "target_entity_description": target_entity_description,
                "relationship_desc": relationship.relationship_desc,
                "meta": {
                    **extra_meta,
                },
            }
            relationships_data.append(relationship_dict)

        # Create DataFrames for entities and relationships
        entities_df = pd.DataFrame(entities_data)
        relationships_df = pd.DataFrame(relationships_data)
        return entities_df, relationships_df


