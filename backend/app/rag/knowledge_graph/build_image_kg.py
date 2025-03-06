import logging
import time
from typing import List, Optional
from sqlmodel import Session

from app.models.image import Image
from app.rag.knowledge_graph.image_integration import create_image_entities_and_relationships
from app.rag.knowledge_graph.image_entity_linker import _search_related_entities, link_image_to_entities
from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

def build_kg_from_images(session: Session, document_id: int, dspy_lm, embed_model) -> None:
    """
    Builds knowledge graph entities and relationships from images associated with a document.
    
    Args:
        session (Session): Database session
        document_id (int): ID of the document whose images should be processed
        dspy_lm: DSPy language model for knowledge graph operations
        embed_model: Embedding model for vector representations
        
    Returns:
        None
    """
    start_time = time.time()
    logger.info(f"Starting knowledge graph building process for document ID: {document_id}")
    
    # Query all images associated with the document
    try:
        images = session.query(Image).filter(Image.source_document_id == document_id).all()
        logger.info(f"Successfully queried images for document ID: {document_id}")
    except Exception as e:
        logger.error(f"Failed to query images for document ID: {document_id}. Error: {str(e)}")
        logger.exception(e)
        return
    
    if not images:
        logger.info(f"No images found for document ID: {document_id}")
        return
    
    logger.info(f"Processing {len(images)} images for document ID: {document_id} to build knowledge graph")
    
    # Initialize graph store
    try:
        logger.info(f"Initializing TiDBGraphStore for document ID: {document_id}")
        graph_store = TiDBGraphStore(
            session=session, 
            dspy_lm=dspy_lm, 
            embed_model=embed_model
        )
        logger.info(f"Successfully initialized TiDBGraphStore for document ID: {document_id}")
    except Exception as e:
        logger.error(f"Failed to initialize TiDBGraphStore for document ID: {document_id}. Error: {str(e)}")
        logger.exception(e)
        return
    
    try:
        # Initialize knowledge graph index
        try:
            logger.info(f"Initializing KnowledgeGraphIndex for document ID: {document_id}")
            graph_index = KnowledgeGraphIndex.from_existing(
                dspy_lm=dspy_lm, 
                kg_store=graph_store
            )
            logger.info(f"Successfully initialized KnowledgeGraphIndex for document ID: {document_id}")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphIndex for document ID: {document_id}. Error: {str(e)}")
            logger.exception(e)
            return
        
        # Process each image and add to knowledge graph
        logger.info(f"Starting to process {len(images)} images for document ID: {document_id}")
        processed_count = 0
        success_count = 0
        error_count = 0
        for idx, image in enumerate(images):
            try:
                logger.info(f"Processing image {idx+1}/{len(images)}, ID: {image.id}")
                processed_count += 1
                
                # Create knowledge graph from image
                logger.info(f"Creating entities and relationships for image ID: {image.id}")
                image_kg = create_image_entities_and_relationships(image)
                
                if not image_kg.entities or not image_kg.relationships:
                    logger.warning(f"No entities or relationships generated for image ID: {image.id}")
                    error_count += 1
                    continue
                
                logger.info(f"Generated {len(image_kg.entities)} entities and {len(image_kg.relationships)} relationships for image ID: {image.id}")
                    
                # Convert entities and relationships to nodes and insert them into the graph
                logger.info(f"Converting and inserting entities for image ID: {image.id}")
                entity_insert_count = 0
                for entity in image_kg.entities:
                    try:
                        # Convert entity to LlamaIndex node with all entity information
                        # Ensure image entities are properly marked with their type
                        metadata = {
                            "name": entity.name,
                            "entity_type": "image" if entity.entity_type == "image" else "original",
                            **entity.metadata
                        }
                        if entity.entity_type == "image":
                            metadata["image_url"] = entity.image_url
                            metadata["visual_content"] = entity.visual_content
                        
                        node = TextNode(
                            text=entity.description,
                            metadata=metadata
                        )
                        graph_index.insert_nodes([node])
                        entity_insert_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting entity {entity.name} for image ID: {image.id}. Error: {str(e)}")
                        logger.exception(e)
                
                logger.info(f"Successfully inserted {entity_insert_count}/{len(image_kg.entities)} entities for image ID: {image.id}")
                    
                logger.info(f"Converting and inserting relationships for image ID: {image.id}")
                relationship_insert_count = 0
                for relationship in image_kg.relationships:
                    try:
                        # Convert relationship to LlamaIndex node
                        node = TextNode(
                            text=relationship.relationship_desc,
                            metadata={
                                "source_entity": relationship.source_entity,
                                "target_entity": relationship.target_entity
                            }
                        )
                        graph_index.insert_nodes([node])
                        relationship_insert_count += 1
                    except Exception as e:
                        logger.error(f"Error inserting relationship between {relationship.source_entity} and {relationship.target_entity} for image ID: {image.id}. Error: {str(e)}")
                        logger.exception(e)
                
                logger.info(f"Successfully inserted {relationship_insert_count}/{len(image_kg.relationships)} relationships for image ID: {image.id}")
                
                # Link image to existing entities in the knowledge graph
                try:
                    # Get all existing entities from the graph store
                    logger.info(f"Retrieving existing entities from graph store for image ID: {image.id}")
                    existing_entities = graph_store.get_all_entities()
                    logger.info(f"Retrieved {len(existing_entities)} existing entities from graph store for image ID: {image.id}")
                    
                    # Find entities related to the image
                    logger.info(f"Searching for entities related to image ID: {image.id}")
                    related_entities = _search_related_entities(existing_entities, image)
                    
                    if related_entities:
                        logger.info(f"Found {len(related_entities)} related entities for image ID: {image.id}")
                        
                        # Create relationships between image and related entities
                        # Using asyncio.run to properly call the async function
                        logger.info(f"Creating relationships between image ID: {image.id} and {len(related_entities)} related entities")
                        import asyncio
                        image_entity_relationships = asyncio.run(link_image_to_entities(
                            related_entities=related_entities,
                            image=image,
                            llm=dspy_lm
                        ))
                        logger.info(f"Created {len(image_entity_relationships)} relationships for image ID: {image.id}")
                        
                        # Add the new relationships to the graph
                        for rel in image_entity_relationships:
                            # Convert relationship to TextNode before insertion
                            rel_node = TextNode(
                                text=rel.relationship_desc,
                                metadata={
                                    "source_entity": rel.source_entity,
                                    "target_entity": rel.target_entity,
                                    "relationship_type": "image_entity"
                                }
                            )
                            graph_index.insert_nodes([rel_node])
                        
                        logger.info(f"Added {len(image_entity_relationships)} image-entity relationships for image ID: {image.id}")
                except Exception as e:
                    logger.error(f"Error linking image {image.id} to entities: {str(e)}")
                    logger.exception(e)
                    
                logger.info(f"Added {len(image_kg.entities)} entities and {len(image_kg.relationships)} relationships for image ID: {image.id}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image ID {image.id} for knowledge graph: {str(e)}")
                logger.exception(e)
                error_count += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"Completed building knowledge graph from images for document ID: {document_id}")
        logger.info(f"Processing summary for document ID: {document_id}:")
        logger.info(f"  - Total images processed: {processed_count}/{len(images)}")
        logger.info(f"  - Successfully processed: {success_count}")
        logger.info(f"  - Errors encountered: {error_count}")
        logger.info(f"  - Total processing time: {processing_time:.2f} seconds")
    finally:
        # Always close the session, even if an exception occurs
        graph_store.close_session()