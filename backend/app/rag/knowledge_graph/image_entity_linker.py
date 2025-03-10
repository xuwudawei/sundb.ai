# import logging
# import json
# import requests
# from pathlib import Path
# from functools import lru_cache
# from typing import List
# from fuzzywuzzy.fuzz import token_sort_ratio
# from app.models.image import Image
# from app.rag.knowledge_graph.schema import Entity, Relationship
# from app.utils.image import image_base64_url

# logger = logging.getLogger(__name__)

# PROMPTS = {
#     "EI_LINK_SYSTEM": """You are an expert at analyzing images and their relationships to entities.
# Your task is to determine meaningful connections between images and entities based on visual content and textual descriptions.
# For each valid connection, provide a clear label and description of how the image relates to the entity.""",
    
#     "EI_LINK": """Given the following entities and image information, determine which entities are meaningfully represented or related to the image.
# For each valid connection, explain how the image relates to the entity.

# Entities:
# {entities}

# Image Information:
# {image}

# Provide the relationships in JSON format with the following fields for each connection:
# - entity: name of the related entity
# - label: descriptive label for the relationship (e.g. "depicts", "illustrates", "contains")
# - references: list of specific elements in the image that support this relationship
# - description: detailed explanation of how the image relates to the entity
# """
# }

# def _search_related_entities(entities: List[Entity], image: Image) -> List[Entity]:
#     """
#     Search for related entities to the image and sort by relevance

#     Args:
#         entities (List[Entity]): The entities to search for
#         image (Image): The image to search in

#     Returns:
#         List[Entity]: The related entities sorted by relevance score
#     """

#     @lru_cache(maxsize=None)
#     def compute_similarity(s1: str, s2: str) -> float:
#         return token_sort_ratio(s1.upper(), s2.upper()) / 100.0

#     def compute_avg_similarity(list1: List[str], list2: List[str]) -> float:
#         """
#         Calculate average similarity between two lists of strings
#         """
#         if not list1 or not list2:
#             return 0.0

#         similarities = [compute_similarity(s1, s2) for s1 in list1 for s2 in list2]
#         return sum(similarities) / len(similarities)

#     # def compute_entity_relevance(entity: Entity) -> float:
#     #     """
#     #     Calculate entity's relevance score to the image
#     #     """
#     #     entity_terms = [entity.name] + (entity.meta.get("topic", []) or [])

#     #     # Calculate similarity with image texts
#     #     text_similarity = (
#     #         compute_avg_similarity(entity_terms, [image.text_snippets])
#     #         if image.text_snippets
#     #         else 0.0
#     #     )

#     #     # Calculate similarity with image caption
#     #     caption_similarity = (
#     #         compute_avg_similarity(entity_terms, [image.caption])
#     #         if image.caption
#     #         else 0.0
#     #     )

#     #     # Weight caption similarity slightly higher than text similarity
#     #     score = caption_similarity * 0.6 + text_similarity * 0.4
#     #     return score
#     def compute_entity_relevance(entity: Entity) -> float:
#         """
#         Calculate an entity's relevance score to the image based on its name, topic (if available),
#         and description.

#         Returns:
#             float: The relevance score.
#         """
#         # Extract the 'topic' from the custom metadata (stored in the 'meta' field)
#         topic = entity.meta.get("topic", "") if isinstance(entity.meta, dict) else ""
        
#         # Combine the relevant attributes into a list of terms
#         entity_terms = [entity.name, topic, entity.description]

#         # Calculate similarity with image text snippets
#         text_similarity = (
#             compute_avg_similarity(entity_terms, [image.text_snippets])
#             if image.text_snippets
#             else 0.0
#         )

#         # Calculate similarity with the image caption
#         caption_similarity = (
#             compute_avg_similarity(entity_terms, [image.caption])
#             if image.caption
#             else 0.0
#         )

#         # Weight caption similarity slightly higher than text similarity
#         score = caption_similarity * 0.6 + text_similarity * 0.4
#         return score


#     related_entities = []
#     if not image.text_snippets and not image.caption:
#         return related_entities

#     # Calculate relevance scores for each entity
#     scored_entities = [
#         (entity, compute_entity_relevance(entity)) for entity in entities
#     ]

#     # Filter entities with minimum relevance and sort by score
#     MIN_RELEVANCE = 0.1
#     related_entities = [
#         entity
#         for entity, score in sorted(scored_entities, key=lambda x: x[1], reverse=True)
#         if score >= MIN_RELEVANCE
#     ]

#     return related_entities


# async def link_image_to_entities(
#     related_entities: List[Entity], image: Image, llm
# ) -> List[Relationship]:
#     """
#     Link image to entities

#     Args:
#         related_entities (List[Entity]): The related entities
#         image (Image): The image to link
#         llm: Language model for analyzing image-entity relationships

#     Returns:
#         List[Relationship]: The image-entity relations
#     """

#     def _entity_json_str(entity: Entity) -> str:
#         return entity.model_dump_json(
#             include={"name", "description", "metadata"}
#         )

#     def _image_json_str(image: Image) -> str:
#         return {
#             "caption": image.caption,
#             "description": image.description,
#             "text_snippets": image.text_snippets
#         }

#     image_path = Path(image.path)
#     if not image_path.exists():
#         logger.error(f"Image not found at {image_path}")
#         return []

#     messages = [
#         {"role": "system", "content": PROMPTS["EI_LINK_SYSTEM"]},
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": PROMPTS["EI_LINK"].format(
#                         entities="["
#                         + ",\n".join([_entity_json_str(e) for e in related_entities])
#                         + "]",
#                         image=_image_json_str(image),
#                     ),
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"{image_base64_url(str(image_path))}",
#                     },
#                 },
#             ],
#         },
#     ]

#     try:
#         payload_dict = {
#             "model": "gpt-4o-mini",
#             "messages": messages
#         }
#         payload = json.dumps(payload_dict)
        
#         headers = {
#             'Authorization': f'Bearer {llm.api_key}',
#             'Content-Type': 'application/json'
#         }
        
#         response = requests.request("POST", llm.api_url, headers=headers, data=payload)
#         response.raise_for_status()
#         response_json = response.json()
#         res = response_json["choices"][0]["message"]["content"].strip()
        
#         logger.info(
#             f'Link image to entities: entities: {",".join([e.name for e in related_entities])}, image: {image_path}, \nLLM res:\n{res}'
#         )
        
#         # Parse LLM response to extract relationships
#         try:
#             rels = json.loads(res)
#             if not isinstance(rels, list):
#                 logger.error(f"Invalid LLM response format: {res}")
#                 return []
#         except json.JSONDecodeError:
#             logger.error(f"Failed to parse LLM response as JSON: {res}")
#             return []
            
#         # Create relationships from parsed data
#         relationships = []
#         for rel in rels:
#             if not all(k in rel for k in ["entity", "label", "references", "description"]):
#                 logger.warning(f"Skipping incomplete relationship data: {rel}")
#                 continue
                
#             relationship = Relationship(
#                 source_entity=rel["entity"],
#                 target_entity=f"Image_{image.id}",  # Use image entity name format from image_integration.py
#                 relationship_desc=rel["description"]
#             )
#             relationships.append(relationship)
            
#         return relationships
        
#     except Exception as e:
#         logger.error(f"Error linking image to entities: {str(e)}")
#         logger.exception(e)
#         return []


import logging
import json
import httpx
import os
import re
from pathlib import Path
from functools import lru_cache
from typing import List
from fuzzywuzzy.fuzz import token_sort_ratio
from app.models.image import Image
from app.rag.knowledge_graph.schema import Entity, Relationship
from app.utils.image import image_base64_url

logger = logging.getLogger(__name__)

PROMPTS = {
    "EI_LINK_SYSTEM": """You are an expert at analyzing images and their relationships to entities.
Your task is to determine meaningful connections between images and entities based on visual content and textual descriptions.
For each valid connection, provide a clear label and description of how the image relates to the entity.""",
    
    "EI_LINK": """Given the following entities and image information, determine which entities are meaningfully represented or related to the image.
For each valid connection, explain how the image relates to the entity.

Entities:
{entities}

Image Information:
{image}

Provide the relationships in JSON format with the following fields for each connection:
- entity: name of the related entity
- label: descriptive label for the relationship (e.g. "depicts", "illustrates", "contains")
- references: list of specific elements in the image that support this relationship
- description: detailed explanation of how the image relates to the entity
"""
}

def _search_related_entities(entities: List[Entity], image: Image) -> List[Entity]:
    """
    Search for related entities to the image and sort by relevance

    Args:
        entities (List[Entity]): The entities to search for
        image (Image): The image to search in

    Returns:
        List[Entity]: The related entities sorted by relevance score
    """

    @lru_cache(maxsize=None)
    def compute_similarity(s1: str, s2: str) -> float:
        return token_sort_ratio(s1.upper(), s2.upper()) / 100.0

    def compute_avg_similarity(list1: List[str], list2: List[str]) -> float:
        """
        Calculate average similarity between two lists of strings
        """
        if not list1 or not list2:
            return 0.0

        similarities = [compute_similarity(s1, s2) for s1 in list1 for s2 in list2]
        return sum(similarities) / len(similarities)

    def compute_entity_relevance(entity: Entity) -> float:
        """
        Calculate an entity's relevance score to the image based on its name, topic (if available),
        and description.
        """
        topic = entity.meta.get("topic", "") if isinstance(entity.meta, dict) else ""
        entity_terms = [entity.name, topic, entity.description]

        text_similarity = (
            compute_avg_similarity(entity_terms, [image.text_snippets])
            if image.text_snippets
            else 0.0
        )

        caption_similarity = (
            compute_avg_similarity(entity_terms, [image.caption])
            if image.caption
            else 0.0
        )

        score = caption_similarity * 0.6 + text_similarity * 0.4
        return score

    if not image.text_snippets and not image.caption:
        return []

    scored_entities = [
        (entity, compute_entity_relevance(entity)) for entity in entities
    ]

    MIN_RELEVANCE = 0.1
    related_entities = [
        entity
        for entity, score in sorted(scored_entities, key=lambda x: x[1], reverse=True)
        if score >= MIN_RELEVANCE
    ]

    return related_entities


async def link_image_to_entities(
    related_entities: List[Entity], image: Image, llm
) -> List[Relationship]:
    """
    Link image to entities

    Args:
        related_entities (List[Entity]): The related entities
        image (Image): The image to link
        llm: Language model for analyzing image-entity relationships

    Returns:
        List[Relationship]: The image-entity relations
    """

    def _entity_json_str(entity: Entity) -> str:
        return entity.model_dump_json(
            include={"name", "description", "metadata"}
        )

    def _image_json_str(image: Image) -> dict:
        return {
            "caption": image.caption,
            "description": image.description,
            "text_snippets": image.text_snippets
        }

    image_path = image.path
    
    # Check if the path is a URL (starts with http:// or https://)
    if image_path.startswith(('http://', 'https://')):
        # For URLs, we can't use Path.exists(), so we'll just log and proceed
        # Fix URL format if needed (ensure proper protocol separator)
        if '://' not in image_path:
            # Fix malformed URLs like https:/example.com to https://example.com
            image_path = image_path.replace('https:/', 'https://')
            image_path = image_path.replace('http:/', 'http://')
            logger.info(f"Fixed malformed URL: {image_path}")
        
        # For S3 or other remote URLs, we'll skip the existence check
        # and let the downstream code handle any connection issues
        logger.info(f"Using remote image URL: {image_path}")
    else:
        # For local files, check if they exist
        local_path = Path(image_path)
        if not local_path.exists():
            logger.error(f"Image not found at {local_path}")
            return []

    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL")
    messages = [
        {"role": "system", "content": PROMPTS["EI_LINK_SYSTEM"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": PROMPTS["EI_LINK"].format(
                        entities="["
                        + ",\n".join([_entity_json_str(e) for e in related_entities])
                        + "]",
                        image=_image_json_str(image),
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_path if image_path.startswith(('http://', 'https://')) else f"{image_base64_url(str(image_path))}",
                    },
                },
            ],
        },
    ]

    try:
        payload_dict = {
            "model": "gpt-4o-mini",
            "messages": messages
        }
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        timeout = httpx.Timeout(60.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, headers=headers, json=payload_dict)
            response.raise_for_status()
            response_json = response.json()
            res = response_json["choices"][0]["message"]["content"].strip()
        
        logger.info(
            f'Link image to entities: entities: {",".join([e.name for e in related_entities])}, image: {image_path}, \nLLM res:\n{res}'
        )
        # Strip markdown code fences if present
        cleaned_res = re.sub(r"^```(?:json)?\s*", "", res)
        cleaned_res = re.sub(r"\s*```$", "", cleaned_res)
        
        try:
            rels = json.loads(cleaned_res)
            if not isinstance(rels, list):
                logger.error(f"Invalid LLM response format: {cleaned_res}")
                return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {cleaned_res}")
            return []
            
        relationships = []
        for rel in rels:
            if not all(k in rel for k in ["entity", "label", "references", "description"]):
                logger.warning(f"Skipping incomplete relationship data: {rel}")
                continue
                
            relationship = Relationship(
                source_entity=rel["entity"],
                target_entity=f"Image_{image.id}",
                relationship_desc=rel["description"]
            )
            relationships.append(relationship)
            
        return relationships
        
    except httpx.ReadTimeout as e:
        logger.error(f"ReadTimeout error linking image to entities: {str(e)}")
        return []
    
    except Exception as e:
        logger.error(f"Error linking image to entities: {str(e)}")
        logger.exception(e)
        return []
