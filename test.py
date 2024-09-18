import os
import uuid
import hashlib
from sqlalchemy import create_engine, Column, Text, JSON
from sqlmodel import SQLModel, Field, Session, select
from pgvector.sqlalchemy import Vector
from openai import OpenAI
from typing import List, Dict
from enum import Enum


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

DATABASE_URL = "postgresql+psycopg2://myuser:mypassword@localhost:5432/mydb"
engine = create_engine(DATABASE_URL)

class KgIndexStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Chunk(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, nullable=False)  
    text: str = Field(sa_column=Column(Text))  
    hash: str = Field(max_length=64, nullable=False)  
    meta: Dict = Field(default={}, sa_column=Column(JSON))  
    embedding: List[float] = Field(sa_column=Column(Vector(1536), comment="Embedding vector"))  
    source_uri: str = Field(max_length=512, nullable=True)  
    index_status: KgIndexStatus = Field(default=KgIndexStatus.NOT_STARTED, nullable=False)  

    __tablename__ = "chunks"


SQLModel.metadata.create_all(engine)


def generate_embedding(text: str) -> List[float]:
    try:
        response = client.embeddings.create(model="text-embedding-ada-002", input=text)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return [0.0] * 1536  


def generate_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def add_chunk_to_db(session: Session, text: str, metadata: Dict) -> None:
    embedding = generate_embedding(text)
    hash_value = generate_text_hash(text)  # Generate a hash for the text
    chunk = Chunk(text=text, meta=metadata, embedding=embedding, hash=hash_value)
    session.add(chunk)
    session.commit()
    print(f"Added chunk with ID: {chunk.id} and Hash: {chunk.hash}")


def query_by_similarity(session: Session, query_embedding: List[float]) -> None:
    # Query chunks by cosine similarity
    stmt = select(Chunk).order_by(Chunk.embedding.cosine_distance(query_embedding))
    results = session.exec(stmt).all()

    if results:
        for result in results:
            similarity = 1 - cosine_distance(result.embedding, query_embedding)
            print(f"Chunk ID: {result.id}, Text: {result.text}, Similarity: {similarity}")
    else:
        print("No similar chunks found.")


# Function to calculate cosine distance between two vectors
def cosine_distance(v1: List[float], v2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = sum(a * a for a in v1) ** 0.5
    norm_v2 = sum(b * b for b in v2) ** 0.5
    return 1 - (dot_product / (norm_v1 * norm_v2))


def run_demo():
    with Session(engine) as session:
        text_samples = [
            "Artificial intelligence is transforming the world.",
        ]
        for text in text_samples:
            add_chunk_to_db(session, text, {"source": "demo"})

        query_text = "AI is changing everything."
        query_embedding = generate_embedding(query_text)
        
        query_by_similarity(session, query_embedding)


if __name__ == "__main__":
    run_demo()
