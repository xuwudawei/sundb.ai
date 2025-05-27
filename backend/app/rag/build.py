import logging
import re
from typing import List

import dspy
from llama_index.core import VectorStoreIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import TextSplitter
from llama_index.core.schema import Document as LlamaDocument
from sqlmodel import Session

from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.chat_config import get_default_embedding_model
from app.models import (
    Document as DBDocument,
    Chunk as DBChunk,
)

logger = logging.getLogger(__name__)

class ParagraphAwareTextSplitter(TextSplitter):
    """自定义段落感知分割器，保证段落完整性并按 512 字符切割"""

    def split_text(self, text: str) -> List[str]:
        chunks = []
        current_chunk = []
        current_length = 0
        paragraphs = text.split('~!')  # 按照段落结束符 `~!` 来分割

        for para in paragraphs:
            para = para.strip()
            if not para:  # 忽略空段落
                continue

            # 如果段落本身已经超过 512 字符，直接放入一个新的 chunk
            if len(para) > 512:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 将该段落分割成多个小于 512 字符的部分
                for i in range(0, len(para), 512):
                    chunk = para[i:i+512]
                    chunks.append(chunk)
                continue

            # 否则，将段落加入到当前的 chunk 中
            if current_length + len(para) > 512:
                # 如果当前块已经超过了 512 字符，先提交当前块
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += len(para)

            # 如果当前块已经达到 512 字符，提交当前块
            if current_length >= 512:
                chunks.append(''.join(current_chunk))
                current_chunk = []
                current_length = 0

        # 最后处理剩余的内容
        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks


class BuildService:
    """
    Service class for building vector index and graph index.
    """

    def __init__(
            self,
            llm: LLM,
            dspy_lm: dspy.LM,
    ):
        self._llm = llm
        self._dspy_lm = dspy_lm

    def build_vector_index_from_document(
            self, session: Session, db_document: DBDocument
    ):
        embed_mode = get_default_embedding_model(session)

        # 使用自定义段落感知分割器
        splitter = ParagraphAwareTextSplitter()

        _transformations = [splitter]

        vector_store = TiDBVectorStore(session=session)
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embed_mode,
            transformations=_transformations,
        )
        document = db_document.to_llama_document()
        logger.info(f"Start building index for document {document.doc_id}")
        vector_index.insert(document, source_uri=db_document.source_uri)
        logger.info(f"Finish building vector index for document {document.doc_id}")
        vector_store.close_session()
        return vector_index

    def build_kg_index_from_chunk(self, session: Session, db_chunk: DBChunk):
        embed_mode = get_default_embedding_model(session)

        graph_store = TiDBGraphStore(
            session=session, dspy_lm=self._dspy_lm, embed_model=embed_mode
        )
        graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
            dspy_lm=self._dspy_lm, kg_store=graph_store
        )
        node = db_chunk.to_llama_text_node()
        logger.info(f"Start building graph index for chunk {db_chunk.id}")
        graph_index.insert_nodes([node])
        logger.info(f"Finish building graph index for chunk {db_chunk.id}")
        graph_store.close_session()
        return graph_index
