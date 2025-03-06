import logging
from typing import List, Dict, Any
from app.models.image import Image
from app.rag.knowledge_graph.schema import Entity, Relationship, KnowledgeGraph
from app.models.knowledge_graph import EntityType

logger = logging.getLogger(__name__)

def create_image_entities_and_relationships(image: Image) -> KnowledgeGraph:
    """
    Creates a simplified image entity with consolidated metadata and relationships.
    
    Args:
        image (Image): The image model containing analysis results
        
    Returns:
        KnowledgeGraph: A knowledge graph containing the image entity and document relationship
    """
    # Create a single comprehensive image entity with all information
    image_entity = Entity(
        name=f"Image_{image.id}",
        description=image.description or "Image without description",
        metadata={
            "topic": "Image Analysis",
            "type": "image",
            "path": image.path,
            "source_document_id": image.source_document_id,
            "caption": image.caption,
            "text_content": image.text_snippets if image.text_snippets else None,
            "visual_elements": {
                "description": image.description,
                "type": "diagram" if "diagram" in (image.description or "").lower() else "image",
                "elements": image.description.split(", ") if image.description else []
            } if image.description else None
        },
        entity_type="image",
        image_url=image.path,
        visual_content=image.description
    )
    
    # Create only the document relationship
    document_relationship = Relationship(
        source_entity=image_entity.name,
        target_entity=f"Document_{image.source_document_id}",
        relationship_desc=f"Image from document {image.source_document_id}"
    )
    
    return KnowledgeGraph(
        entities=[image_entity],
        relationships=[document_relationship]
    )