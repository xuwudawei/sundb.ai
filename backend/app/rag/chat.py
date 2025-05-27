import json
import time
import logging
import re
import dspy
from sqlalchemy.exc import SQLAlchemyError
from uuid import UUID
from typing import List, Generator, Optional, Tuple, Type, Callable
from datetime import datetime, UTC
from urllib.parse import urljoin

import requests
import jinja2
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from pydantic import BaseModel
from sqlmodel import Session, select, func
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.base import ChatMessage
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.callbacks.schema import EventPayload
from llama_index.core.callbacks import CallbackManager
from llama_index.core.response_synthesizers import get_response_synthesizer
from langfuse import Langfuse
from langfuse.llama_index import LlamaIndexCallbackHandler

from app.models import (
    User,
    Document,
    Chunk,
    ChatVisibility,
    Chat as DBChat,
    ChatMessage as DBChatMessage,
    LLM as DBLLM,
    EmbeddingModel as DBEmbeddingModel,
    DataSource as DBDataSource,
    RerankerModel as DBRerankerModel,
)
from app.core.config import settings
from app.models.recommend_question import RecommendQuestion
from app.rag.chat_stream_protocol import (
    ChatStreamMessagePayload,
    ChatStreamDataPayload,
    ChatEvent,
)
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.knowledge_graph.graph_store import (
    TiDBGraphStore,
    tidb_graph_editor as editor,
)
from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.rag.chat_config import ChatEngineConfig, get_default_embedding_model, KnowledgeGraphOption
from app.rag.types import (
    MyCBEventType,
    ChatMessageSate,
    ChatEventType,
    MessageRole,
)
from app.repositories import chat_repo
from app.site_settings import SiteSetting
from app.exceptions import ChatNotFound

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
            self,
            *,
            db_session: Session,
            user: User,
            browser_id: str,
            origin: str,
            chat_messages: List[ChatMessage],
            engine_name: str = "default",
            chat_id: Optional[UUID] = None,
    ) -> None:
        self.db_session = db_session
        self.user = user
        self.browser_id = browser_id
        self.engine_name = engine_name

        self.user_question, self.chat_history = self._parse_chat_messages(chat_messages)
        if chat_id:
            # FIXME:
            #   only chat owner or superuser can access the chat,
            #   anonymous user can only access anonymous chat by track_id
            self.db_chat_obj = chat_repo.get(self.db_session, chat_id)
            if not self.db_chat_obj:
                raise ChatNotFound()
            try:
                self.chat_engine_config = ChatEngineConfig.load_from_db(db_session, self.db_chat_obj.engine.name)
                self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            except Exception as e:
                logger.error(f"Failed to load chat engine config: {e}")
                self.chat_engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
                self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            logger.info(f"ChatService - chat_id: {chat_id}, chat_engine: {self.db_chat_obj.engine.name}")
            self.chat_history = [
                ChatMessage(role=m.role, content=m.content, additional_kwargs={})
                for m in chat_repo.get_messages(self.db_session, self.db_chat_obj)
            ]
        else:
            self.chat_engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
            self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
            self.db_chat_obj = chat_repo.create(
                self.db_session,
                DBChat(
                    title=self.user_question[:100],
                    engine_id=self.db_chat_engine.id,
                    engine_options=self.chat_engine_config.screenshot(),
                    user_id=self.user.id if self.user else None,
                    browser_id=self.browser_id,
                    origin=origin,
                    visibility=ChatVisibility.PUBLIC.value
                    if not self.user
                    else ChatVisibility.PRIVATE.value,
                ),
            )
            chat_id = self.db_chat_obj.id
            # slack/discord may create a new chat with history messages
            now = datetime.now(UTC)
            for i, m in enumerate(self.chat_history):
                chat_repo.create_message(
                    session=self.db_session,
                    chat=self.db_chat_obj,
                    chat_message=DBChatMessage(
                        role=m.role,
                        content=m.content,
                        ordinal=i + 1,
                        created_at=now,
                        updated_at=now,
                        finished_at=now,
                    ),
                )

        self._reranker = self.chat_engine_config.get_reranker(db_session)
        self._metadata_filter = self.chat_engine_config.get_metadata_filter()
        if self._reranker:
            self._node_postprocessors = [self._metadata_filter, self._reranker]
            # Set initial similarity_top_k to a large number,
            # reranker will filter out irrelevant nodes after the retrieval
            self._similarity_top_k = 60
        else:
            self._node_postprocessors = [self._metadata_filter]
            self._similarity_top_k = 10

        self.langfuse_host = SiteSetting.langfuse_host
        self.langfuse_secret_key = SiteSetting.langfuse_secret_key
        self.langfuse_public_key = SiteSetting.langfuse_public_key
        self.enable_langfuse = (
                self.langfuse_host and self.langfuse_secret_key and self.langfuse_public_key
        )

    def chat(self) -> Generator[ChatEvent | str, None, None]:
        try:
            if self.chat_engine_config.external_engine_config:
                for event in self._external_chat():
                    yield event
            else:
                for event in self._chat():
                    yield event
        except Exception as e:
            logger.exception(e)
            yield ChatEvent(
                event_type=ChatEventType.ERROR_PART,
                payload="Encountered an error while processing the chat. Please try again later.",
            )

    def _search_kg(
            self,
            kg_config: KnowledgeGraphOption,
            fast_dspy_lm: dspy.LM,
            embed_model: BaseEmbedding,
            get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
            trace_url: str,
            annotation_silent: bool = False,
    ) -> Generator[ChatEvent | str, None, Tuple[List[dict], List[dict], List[dict], dict, str]]:
        """
        Search the knowledge graph for relevant entities, relationships, and chunks.
        Args:
            kg_config: KnowledgeGraphOption
            fast_dspy_lm: dspy.LM
            embed_model: BaseEmbedding
            get_llamaindex_callback_manager: Callable[[], CallbackManager]
            trace_url: str
            annotation_silent: bool, if True, do not send annotation events

        Returns:
            List[dict]: entities
            List[dict]: relationships
            List[dict]: chunks
            dict: graph_data_source_ids
            str: graph_knowledges_context
        """

        entities, relations, chunks = [], [], []
        graph_data_source_ids = {}
        graph_knowledges_context = ""

        if kg_config.enabled:
            graph_store = TiDBGraphStore(
                dspy_lm=fast_dspy_lm,
                session=self.db_session,
                embed_model=embed_model,
            )
            graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
                dspy_lm=fast_dspy_lm,
                kg_store=graph_store,
                callback_manager=get_llamaindex_callback_manager(),
            )

            if kg_config.using_intent_search:
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.KG_RETRIEVAL,
                            display="我正在理解您问题的核心意图",
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                sub_queries = graph_index.intent_analyze(
                    self.user_question,
                    self.chat_history,
                )
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.TRACE,
                            display="我正在从知识图谱中搜索相关信息",
                            context={"langfuse_url": trace_url},
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                result = graph_index.graph_semantic_search(
                    sub_queries,
                    include_meta=True,
                    relationship_meta_filters=kg_config.relationship_meta_filters,
                )

                entities = result["graph"]["entities"]
                relations = result["graph"]["relationships"]
                graph_data_source_ids = {
                    "entities": [e["id"] for e in entities],
                    "relationships": [r["id"] for r in relations],
                }

                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    sub_queries=result["queries"],
                )
                graph_knowledges_context = graph_knowledges.template
            else:
                if not annotation_silent:
                    yield ChatEvent(
                        event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                        payload=ChatStreamMessagePayload(
                            state=ChatMessageSate.TRACE,
                            display="我正在从知识图谱中搜索相关信息",
                            context={"langfuse_url": trace_url},
                        ),
                    )
                graph_index._callback_manager = get_llamaindex_callback_manager()
                entities, relations, chunks = graph_index.retrieve_with_weight(
                    self.user_question,
                    [],
                    depth=kg_config.depth,
                    include_meta=kg_config.include_meta,
                    with_degree=kg_config.with_degree,
                    relationship_meta_filters=kg_config.relationship_meta_filters,
                    with_chunks=False,
                )
                graph_data_source_ids = {
                    "entities": [e["id"] for e in entities],
                    "relationships": [r["id"] for r in relations],
                }
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.normal_graph_knowledge,
                    entities=entities,
                    relationships=relations,
                )
                graph_knowledges_context = graph_knowledges.template

        return entities, relations, chunks, graph_data_source_ids, graph_knowledges_context

    def _get_llamaindex_callback_manager(
            self,
            langfuse: Optional[Langfuse] = None,
            trace_id: Optional[str] = None,
            llm: Optional[LLM] = None,
            fast_llm: Optional[LLM] = None,
            embed_model: Optional[BaseEmbedding] = None,
    ) -> CallbackManager:
        # Why we don't use high-level decorator `observe()` as \
        #   `https://langfuse.com/docs/integrations/llama-index/get-started` suggested?
        # track:
        #   - https://github.com/langfuse/langfuse/issues/2015
        #   - https://langfuse.com/blog/2024-04-python-decorator
        if self.enable_langfuse and langfuse and trace_id:
            observation = langfuse.trace(id=trace_id)
            langfuse_handler = LlamaIndexCallbackHandler()
            langfuse_handler.set_root(observation)
            callback_manager = CallbackManager([langfuse_handler])
        else:
            callback_manager = CallbackManager([])

        if llm:
            llm.callback_manager = callback_manager
        if fast_llm:
            fast_llm.callback_manager = callback_manager
        if embed_model:
            embed_model.callback_manager = callback_manager

        return callback_manager

    def _get_langfuse_config(self):
        langfuse = None
        trace_id = ""
        trace_url = ""

        if self.enable_langfuse:
            langfuse = Langfuse(
                host=self.langfuse_host,
                secret_key=self.langfuse_secret_key,
                public_key=self.langfuse_public_key,
            )
            observation = langfuse.trace(
                name="chat",
                user_id=self.user.email
                if self.user
                else f"anonymous-{self.browser_id}",
                metadata={
                    "chat_engine_config": self.chat_engine_config.screenshot(),
                },
                tags=[f"chat_engine:{self.engine_name}"],
                release=settings.ENVIRONMENT,
                input={
                    "user_question": self.user_question,
                    "chat_history": self.chat_history,
                },
            )
            trace_id = observation.trace_id
            trace_url = observation.get_trace_url()

        return langfuse, trace_id, trace_url

    class ClarityResult(BaseModel):
        clarity_needed: bool
        clarifying_question: str

    def _refine_or_early_stop(
            self,
            get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
            fast_llm: LLM,
            graph_knowledges_context: str,
            refined_question_prompt: Optional[str] = None,
            annotation_silent: bool = False,
    ) -> Generator[ChatEvent | str, None, Tuple[bool, str, str]]:
        """
        Determine whether to refine the user question or early stop the conversation with a clarifying question.

        Args:
            get_llamaindex_callback_manager: Callable[[], CallbackManager]
            fast_llm: LLM
            graph_knowledges_context: str
            refined_question_prompt: Optional[str], if it's None, use the default condense_question_prompt
            annotation_silent: bool, if True, do not send annotation events

        Returns:
            bool: whether to early stop the conversation
            str: clarifying question
            str: refined question
        """
        if refined_question_prompt is None:
            refined_question_prompt = self.chat_engine_config.llm.condense_question_prompt

        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.REFINE_QUESTION,
                    display="我在优化您的查询，以提升信息检索效果",
                ),
            )
        callback_manager = get_llamaindex_callback_manager()

        # 1. Refine the question
        with callback_manager.as_trace("condense_question"):
            with callback_manager.event(
                    MyCBEventType.CONDENSE_QUESTION,
                    payload={EventPayload.QUERY_STR: self.user_question},
            ) as event:
                refined_question = fast_llm.predict(
                    get_prompt_by_jinja2_template(
                        refined_question_prompt,
                        graph_knowledges=graph_knowledges_context,
                        chat_history=self.chat_history,
                        question=self.user_question,
                    ),
                )
                event.on_end(payload={EventPayload.COMPLETION: refined_question})
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.REFINE_QUESTION,
                    message=refined_question,
                ),
            )

        # 2. Check if we have enough information to answer the user question or not
        if self.chat_engine_config.clarify_question:
            with callback_manager.as_trace("check_question"):
                with callback_manager.event(
                        MyCBEventType.CLARIFYING_QUESTION,
                        payload={EventPayload.QUERY_STR: refined_question},
                ) as event:
                    clarity_result = fast_llm.structured_predict(
                        output_cls=self.ClarityResult,
                        prompt=get_prompt_by_jinja2_template(
                            self.chat_engine_config.llm.clarifying_question_prompt,
                            graph_knowledges=graph_knowledges_context,
                            chat_history=self.chat_history,
                            question=refined_question,
                        ),
                    )
                    event.on_end(payload={
                        EventPayload.COMPLETION: f"Need Clarification: {clarity_result.clarity_needed}, "
                                                 f"Clarifying Question: {clarity_result.clarifying_question}"
                    })

                    if clarity_result.clarity_needed:
                        if not annotation_silent:
                            yield ChatEvent(
                                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                                payload=ChatStreamMessagePayload(
                                    state=ChatMessageSate.GENERATE_ANSWER,
                                    display="我需要向您提出一个澄清性的问题",
                                ),
                            )

                        yield ChatEvent(
                            event_type=ChatEventType.TEXT_PART,
                            payload=clarity_result.clarifying_question,
                        )

                        return True, clarity_result.clarifying_question, ""

        return False, "", refined_question

    def _gen_answer_via_llama_index(
            self,
            get_llamaindex_callback_manager: Callable[[], Optional[CallbackManager]],
            refined_question: str,
            graph_knowledges_context: str,
            llm: LLM,
            embed_model: BaseEmbedding,
            annotation_silent: bool = False,
    ) -> Generator[ChatEvent | str, None, Tuple[StreamingResponse, List[dict]]]:
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SEARCH_RELATED_DOCUMENTS,
                    display="我正在检索并重新排序最匹配的数据"
                    if self._reranker
                    else "我正在检索最相关的数据",
                ),
            )
        callback_manager = get_llamaindex_callback_manager()
        #JSON 文件路径
        json_file_path = '/hd1/workspace/sundb-ai/project/code/sundbai_midterm/sundb.ai/backend/app/rag/structured_.json'
        logger.info("原始问题: %s", self.user_question)



        # 读取 JSON 文件并赋值给 manual_structure
        with open(json_file_path, 'r', encoding='utf-8') as file:
            manual_structure = json.load(file)
        # # 动态初始化章节索引（新增）
        # if hasattr(self, '_manual_sections_index'):
        #     logging.info("索引已经建立过。")
        # else:
        #     logging.info("索引未建立，开始建立索引...")
        #     self._initialize_manual_sections_index(embed_model)  # 使用传入的embed_model
        #     logging.info("索引建立完成。")
        # 新增：匹配相关章节
        matched_sections = self._match_manual_sections(llm, self.user_question, manual_structure)
        # matched_sections = self._match_manual_sections(refined_question)
        # 输出匹配的章节
        if matched_sections:
            logger.info("Matched sections based on the question: %s", self.user_question)
            for section in matched_sections:
                logger.info(
                    "Part: %s, Chapter: %s, Section: %s, Title: %s",
                    section["part_number"],
                    section["chapter_number"],
                    section["section_number"],
                    section["title"],
                )
        else:
            logger.warning("No sections matched for the question: %s", self.user_question)
        document_ids = self._fetch_section_document_ids(matched_sections)

        # 输出 document_id
        logger.info(f"Matched document IDs: {document_ids}")
        start_time = datetime.now()
        logger.info(f"Process started at: {start_time}")




        text_qa_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.text_qa_prompt,
            current_date=datetime.now().strftime("%Y-%m-%d"),
            graph_knowledges=graph_knowledges_context,
            original_question=self.user_question,
        )
        refine_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.refine_prompt,
            graph_knowledges=graph_knowledges_context,
            original_question=self.user_question,
        )
        # 记录日志
        logging.info(f"text_qa_template: {text_qa_template}")
        logging.info(f"refine_template: {refine_template}")

        def create_query_engine(filter_ids: List[str] = None) -> Tuple[StreamingResponse, List[dict]]:
            """创建带文档过滤的查询引擎"""
            # 创建带过滤条件的向量存储
            vector_store = TiDBVectorStore(
                session=self.db_session,
                filter_document_ids=filter_ids  # 支持文档过滤
            )

            # 创建向量索引
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
                callback_manager=callback_manager,
            )

            # 配置响应合成器
            response_synthesizer = get_response_synthesizer(
                llm=llm,
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                streaming=True,
                callback_manager=callback_manager,
            )

            # 创建查询引擎
            query_engine = vector_index.as_query_engine(
                llm=llm,
                response_synthesizer=response_synthesizer,
                node_postprocessors=self._node_postprocessors,
                similarity_top_k=self._similarity_top_k,
            )

            # 设置回调管理器
            query_engine.callback_manager = callback_manager
            for _np in self._node_postprocessors:
                _np.callback_manager = callback_manager

            return query_engine

        def is_local_search_effective(local_response) -> bool:
            """判断局部检索是否有效"""
            # 空结果直接无效
            if not local_response.source_nodes:
                return False

            # 关键指标提取
            node_count = len(local_response.source_nodes)
            top_score = local_response.source_nodes[0].score

            # 双重条件判断
            score_condition = top_score >= 0.8  # 最高相关性达标
            quantity_condition = node_count >= 3  # 结果数量充足

            # 任意条件不满足即视为效果不佳
            return score_condition and quantity_condition

        final_response = None
        source_documents = []
        if document_ids:
            # 创建局部查询引擎（只加载相关文档）
            local_query_engine = create_query_engine(document_ids)
            response: StreamingResponse = local_query_engine.query(refined_question)

            # 调试信息改为仅记录元数据，不消费生成器
            logger.info("相关节点数量: %d", len(response.source_nodes))
            if response.source_nodes:
                logger.info("最高相关性得分: %.4f", response.source_nodes[0].score)
                # 记录前3个节点的部分信息用于调试
                for i, node in enumerate(response.source_nodes[:3]):
                    logger.debug("节点 %d 元数据: %s", i + 1, str(node.metadata)[:100])
            else:
                logger.warning("没有找到相关节点")

            # 判断检索效果（基于元数据，不消费生成器）
            if is_local_search_effective(response):
                logger.info("局部检索效果达标，使用本地结果")
                final_response = response
                source_documents = self._get_source_documents(response)
                logger.info("局部检索得到的 source_documents: %s", source_documents)
            else:
                logger.warning(
                    "局部检索效果不佳 (节点数=%d, 最高分=%.2f)，触发回退机制",
                    len(response.source_nodes),
                    response.source_nodes[0].score if response.source_nodes else 0
                )
        if not final_response:
            logger.info("执行全局检索")

            vector_store = TiDBVectorStore(session=self.db_session)
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store,
                embed_model=embed_model,
                callback_manager=callback_manager,
            )
            response_synthesizer = get_response_synthesizer(
                llm=llm,
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                streaming=True,
                callback_manager=callback_manager,
            )
            query_engine = vector_index.as_query_engine(
                llm=llm,
                response_synthesizer=response_synthesizer,
                node_postprocessors=self._node_postprocessors,
                similarity_top_k=self._similarity_top_k,
            )
            query_engine.callback_manager = callback_manager
            for _np in self._node_postprocessors:
                _np.callback_manager = callback_manager
            response: StreamingResponse = query_engine.query(refined_question)
            final_response = response
            source_documents = self._get_source_documents(response)
            logger.info("全局检索得到的 source_documents: %s", source_documents)

        # 记录结束时间
        end_time = datetime.now()
        logger.info(f"Process ended at: {end_time}")

        # 计算并输出总耗时
        total_time = end_time - start_time
        logger.info(f"Total time taken: {total_time}")
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.SOURCE_NODES,
                    context=source_documents,
                ),
            )
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.GENERATE_ANSWER,
                    display="我正在为您生成精确的答案",
                ),
            )

        return final_response, source_documents

    def _fetch_section_document_ids(self, matched_sections: List[dict]) -> List[str]:
        """根据匹配的章节获取对应的文档ID集合（修正会话方法版本）

        Args:
            matched_sections: 匹配的章节列表，每个章节应包含
                part_number/chapter_number/section_number 字段

        Returns:
            去重后的文档ID列表
        """
        document_ids = []

        if not matched_sections:
            logger.debug("输入 matched_sections 为空列表，直接返回空结果")
            return []

        logger.info(f"开始处理 {len(matched_sections)} 个匹配章节的文档查询")

        try:
            for idx, section in enumerate(matched_sections, 1):
                # 参数校验与日志
                required_keys = ["part_number", "chapter_number", "section_number"]
                if not all(key in section for key in required_keys):
                    logger.error(f"第 {idx} 个章节缺少必要字段，跳过处理 | 数据: {section}")
                    continue

                part = section["part_number"]
                chapter = section["chapter_number"]
                section_num = section["section_number"]

                logger.debug(
                    f"正在查询章节 {idx}/{len(matched_sections)}: Part={part}, Chapter={chapter}, Section={section_num}")

                # 构建查询
                stmt = select(Document.id).where(
                    Document.part_number == part,
                    Document.chapter_number == chapter,
                    Document.section_number == section_num
                )

                # 正确执行查询
                result = self.db_session.execute(stmt)  # 修正点1
                records = result.scalars().all()  # 修正点2

                logger.debug(f"查询到 {len(records)} 个相关文档")

                # 处理结果
                if records:
                    current_ids = [str(doc_id) for doc_id in records]
                    document_ids.extend(current_ids)
                    logger.debug(f"新增文档ID列表: {current_ids}")
                else:
                    logger.warning(f"未找到匹配文档 | Part={part}, Chapter={chapter}, Section={section_num}")

        except SQLAlchemyError as e:
            logger.critical("数据库查询异常！错误信息：%s", str(e), exc_info=True)
            raise
        except AttributeError as e:
            logger.error("会话方法调用错误，请检查SQLAlchemy版本：%s", str(e), exc_info=True)
            raise
        except Exception as e:
            logger.error("处理过程中出现未预期错误：%s", str(e), exc_info=True)
            raise

        # 去重处理（保持顺序）
        seen = set()
        unique_ids = [x for x in document_ids if not (x in seen or seen.add(x))]

        logger.info(f"最终获取到 {len(unique_ids)} 个唯一文档ID")
        return unique_ids

    def _match_manual_sections(
            self,
            llm,
            question: str,
            manual_structure: list[dict]
    ) -> list[dict]:
        """使用LLM匹配手册章节（带语义标签版）"""
        from llama_index.core.llms import ChatMessage


        # 构造带信息的章节列表（格式优化）
        sections_str = "\n".join([
            f"{s['part_number']}.{s['chapter_number']}.{s['section_number']}  {s['title']}\n摘要：{s['abstract']}\n关键词：{', '.join(s['keywords'])}"
            for s in manual_structure
        ])

        # 优化提示词，新增匹配依据输出要求
        prompt = f"""请根据用户提出的问题，从下面的手册目录里找出与之相关的章节。在匹配时，需要综合考量章节的标题、摘要以及关键词所蕴含的语义信息。

        问题：{question}

        手册目录格式说明：
        每一条目录信息包含章节编号、标题、摘要和关键词，各部分的具体说明如下：
        - 章节编号：采用标准的三级编号格式，即 Part.Chapter.Section，用于唯一标识一个章节。
        - 标题：对该章节核心内容的简要概括。
        - 摘要：对该章节具体内容的详细概述，能帮助你了解章节的主要信息。
        - 关键词：该章节的核心要点，以逗号分隔列出。

        手册目录内容：
        {sections_str}

        匹配原则：
        1. 优先选择整体语义（包括标题、摘要和关键词的综合语义）与问题最匹配的章节。
        2. 若整体语义匹配度相近，再考虑关键词与问题的匹配程度。
        3. 若关键词匹配度也相近，最后考虑标题中的关键词与问题的匹配情况。
        4. 返回的结果必须采用标准的三级编号（Part.Chapter.Section）。

        输出要求：
        请先用自然语言说明匹配理由，再输出JSON格式结果。匹配理由需要包含：
        - 问题与各章节在语义层面的关联性分析
        - 关键词匹配情况的说明
        - 最终选择这些章节的理由

        输出格式示例：
        问题主要涉及设备初始化操作，以下章节的摘要和关键词与问题相关：
        - 第1.2.3节的摘要包含设备初始化流程说明，关键词"初始化"、"配置"与问题直接相关
        - 第2.1.5节虽然包含配置相关内容，但主要针对网络设置，与设备初始化关联度较低

        ```json
        [{{"part_number": "1", "chapter_number": "2", "section_number": "3", "title": "设备初始化流程"}}]
        """

        # 调用LLM
        response = llm.complete(prompt)
        logger.debug("LLM原始响应: %s", response.text)

        try:
            # 提取匹配依据和JSON内容
            reasoning = "未提取到匹配依据"
            json_content = response.text

            # 通过正则表达式分离自然语言解释和JSON代码
            match = re.search(
                r'(.*?)(```json\s*?\n.*?\n```)(.*)',
                response.text,
                re.DOTALL
            )

            if match:
                # 提取匹配依据部分
                reasoning = match.group(1).strip()
                # 提取并清理JSON部分
                json_content = match.group(2)
                cleaned_response = re.sub(
                    r'^```json\n|```$',
                    '',
                    json_content,
                    flags=re.MULTILINE
                ).strip()

                # 记录匹配依据到日志（新增关键日志点）
                logger.info(
                    "LLM匹配依据分析：\n问题：'%s'\n匹配逻辑：%s",
                    question,
                    reasoning
                )
            else:
                cleaned_response = re.sub(
                    r'^```json\n|```$',
                    '',
                    response.text,
                    flags=re.MULTILINE
                ).strip()
                logger.warning("响应中未找到明确的JSON代码块")

            # 解析JSON（保持原有逻辑）
            matched = json.loads(cleaned_response)
            logger.debug("解析后的JSON内容: %s", matched)

            if not isinstance(matched, list):
                logger.warning("无效的匹配结果格式: %s", matched)
                return []

            # 验证匹配结果并构建valid_sections
            valid_sections = []
            for item in matched:
                if not all(map(str.isdigit, [item.get('part_number'),
                                             item.get('chapter_number'),
                                             item.get('section_number')])):
                    logger.warning("无效的章节编号格式: %s", item)
                    continue

                found = next(
                    (s for s in manual_structure
                     if s["part_number"] == item["part_number"] and
                     s["chapter_number"] == item["chapter_number"] and
                     s["section_number"] == item["section_number"]),
                    None
                )
                if found:
                    valid_sections.append({
                        **found,
                        "matched_title": item.get("title", "未提供标题"),
                        "matched_keywords": found.get("keywords", []),
                        "matched_abstract": found.get("abstract", ""),
                        "matching_reason": reasoning  # 将匹配依据也存入结果中
                    })
                    logger.debug("匹配成功 问题：'%s' ▶ 章节：%s | 关键词：'%s' | 摘要：'%s'",
                                 question,
                                 f"{found['part_number']}.{found['chapter_number']}.{found['section_number']}",
                                 found['keywords'],
                                 found['abstract'])
                else:
                    logger.warning("未找到匹配的章节: %s", item)

            # 去重处理（修复不可哈希问题）
            seen = set()
            unique_sections = []
            for section in valid_sections:
                identifier = (
                    section["part_number"],
                    section["chapter_number"],
                    section["section_number"]
                )
                if identifier not in seen:
                    seen.add(identifier)
                    unique_sections.append(section)
            valid_sections = unique_sections

            logger.info("语义匹配结果: %s",
                        [{"part": s["part_number"],
                          "chapter": s["chapter_number"],
                          "section": s["section_number"],
                          "标题": s["title"],
                          "匹配关键词": s.get("keywords", []),
                          "匹配摘要": s.get("abstract", ""),
                          "匹配依据": s.get("matching_reason", "")[:100] + "..." if len(
                              s.get("matching_reason", "")) > 100 else s.get("matching_reason", "")}
                         for s in valid_sections])

            return valid_sections

        except json.JSONDecodeError as e:
            logger.error("JSON解析错误: %s\n原始响应: %s", str(e), response.text)
            return []
        except Exception as e:
            logger.error("匹配过程中发生意外错误: %s", str(e))
            return []

    def _chat_finish(
            self,
            db_assistant_message: ChatMessage,
            db_user_message: ChatMessage,
            response_text: str,
            source_documents: List[dict],
            graph_data_source_ids: dict,
            annotation_silent: bool = False,
    ):
        if not annotation_silent:
            yield ChatEvent(
                event_type=ChatEventType.MESSAGE_ANNOTATIONS_PART,
                payload=ChatStreamMessagePayload(
                    state=ChatMessageSate.FINISHED,
                ),
            )

        post_verification_result_url = self._post_verification(
            self.user_question,
            response_text,
            self.db_chat_obj.id,
            db_assistant_message.id,
        )

        db_assistant_message.sources = source_documents
        db_assistant_message.graph_data = graph_data_source_ids
        db_assistant_message.content = response_text
        db_assistant_message.post_verification_result_url = post_verification_result_url
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)
        db_user_message.graph_data = graph_data_source_ids
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    def _chat(self) -> Generator[ChatEvent | str, None, None]:
        langfuse, trace_id, trace_url = self._get_langfuse_config()

        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url=trace_url,
                content=self.user_question,
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url=trace_url,
                content="",
            ),
        )

        _embed_model = get_default_embedding_model(self.db_session)
        _llm = self.chat_engine_config.get_llama_llm(self.db_session)
        _fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        _fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

        def _get_llamaindex_callback_manager_in_chat() -> CallbackManager:
            return self._get_llamaindex_callback_manager(
                langfuse=langfuse,
                trace_id=trace_id,
                llm=_llm,
                fast_llm=_fast_llm,
                embed_model=_embed_model,
            )

        # 1. Retrieve entities, relations, and chunks from the knowledge graph
        kg_config = self.chat_engine_config.knowledge_graph
        entities, relations, chunks, graph_data_source_ids, graph_knowledges_context = yield from self._search_kg(
            kg_config=kg_config,
            fast_dspy_lm=_fast_dspy_lm,
            embed_model=_embed_model,
            trace_url=trace_url,
            get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
        )

        # 2. Refine the user question using graph information and chat history
        # 2.1 Early stop if the user question does not have enough information, we need to ask a clarifying question
        early_stop, clarifying_question, refined_question = yield from self._refine_or_early_stop(
            get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
            fast_llm=_fast_llm,
            graph_knowledges_context=graph_knowledges_context,
        )
        if early_stop:
            # the clarifying question is the final response
            response_text = clarifying_question
            source_documents = []
        else:
            # 3. Retrieve the related chunks from the vector store
            # 4. Rerank after the retrieval
            # 5. Generate a response using the refined question and related chunks
            response, source_documents = yield from self._gen_answer_via_llama_index(
                get_llamaindex_callback_manager=_get_llamaindex_callback_manager_in_chat,
                refined_question=refined_question,
                graph_knowledges_context=graph_knowledges_context,
                llm=_llm,
                embed_model=_embed_model,
            )

            response_text = ""
            for word in response.response_gen:
                response_text += word
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )

            if not response_text:
                raise Exception("Got empty response from LLM")

        yield from self._chat_finish(
            db_assistant_message=db_assistant_message,
            db_user_message=db_user_message,
            response_text=response_text,
            source_documents=source_documents,
            graph_data_source_ids=graph_data_source_ids,
        )

    def _external_chat(self) -> Generator[ChatEvent | str, None, None]:
        # TODO: integration with langfuse.
        db_user_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.USER.value,
                trace_url="",
                content=self.user_question,
            ),
        )
        db_assistant_message = chat_repo.create_message(
            session=self.db_session,
            chat=self.db_chat_obj,
            chat_message=DBChatMessage(
                role=MessageRole.ASSISTANT.value,
                trace_url="",
                content="",
            ),
        )

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

        _embed_model = get_default_embedding_model(self.db_session)
        _fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)
        _fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)

        # retrieve entities, relations, and chunks from the knowledge graph
        # this retrieve progress is only for the clarifying question checking
        try:
            """
            kg_config = self.chat_engine_config.knowledge_graph
            _, _, _, graph_data_source_ids, graph_knowledges_context = yield from self._search_kg(
                kg_config=kg_config,
                fast_dspy_lm=_fast_dspy_lm,
                embed_model=_embed_model,
                trace_url="",
                get_llamaindex_callback_manager=lambda: self._get_llamaindex_callback_manager(
                    fast_llm=_fast_llm,
                    embed_model=_embed_model,
                ),
                annotation_silent=True,
            )
            """
            graph_data_source_ids = []
            graph_knowledges_context = ""

            logger.info("start to _refine_or_early_stop")
            early_stop, clarifying_question, goal = yield from self._refine_or_early_stop(
                get_llamaindex_callback_manager=lambda: self._get_llamaindex_callback_manager(
                    fast_llm=_fast_llm,
                    embed_model=_embed_model,
                ),
                fast_llm=_fast_llm,
                graph_knowledges_context=graph_knowledges_context,
                refined_question_prompt=self.chat_engine_config.llm.generate_goal_prompt,
                annotation_silent=True,
            )
            logger.info(f"start request stackvm with goal: {goal}")
            goal = goal.strip()
            if goal.startswith("Goal: "):
                goal = goal[len("Goal: "):].strip()

            if early_stop:
                # the clarifying question is the final response
                yield from self._chat_finish(
                    db_assistant_message=db_assistant_message,
                    db_user_message=db_user_message,
                    response_text=clarifying_question,
                    source_documents=[],
                    graph_data_source_ids=graph_data_source_ids,
                    annotation_silent=True,
                )
                return
        except Exception as e:
            logger.error(f"Failed to search kg or refine question: {e}")
            goal = self.user_question

        stream_chat_api_url = self.chat_engine_config.external_engine_config.stream_chat_api_url
        logger.debug(
            f"Chatting with external chat engine (api_url: {stream_chat_api_url}) to answer for user question: {self.user_question}")
        chat_params = {
            "goal": goal,
        }
        res = requests.post(stream_chat_api_url, json=chat_params, stream=True)

        # Notice: External type chat engine doesn't support non-streaming mode for now.
        stackvm_response_text = ""
        task_id = None
        for line in res.iter_lines():
            if not line:
                continue

            # Append to final response text.
            chunk = line.decode('utf-8')
            if chunk.startswith("0:"):
                word = json.loads(chunk[2:])
                stackvm_response_text += word
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )
            else:
                yield line + b'\n'

            try:
                if chunk.startswith("8:") and task_id is None:
                    states = json.loads(chunk[2:])
                    if len(states) > 0:
                        # accesss task by cd nhttp://endpoint/?task_id=$task_id
                        task_id = states[0].get("task_id")
            except Exception as e:
                logger.error(f"Failed to get task_id from chunk: {e}")

        """
        try:
            response_text = ""
            final_answer_gen = _fast_llm.stream(
                get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.condense_answer_prompt,
                    chat_history=self.chat_history,
                    question=self.user_question,
                    agent_answer=stackvm_response_text,
                )
            )
            for word in final_answer_gen:
                response_text += word
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )
        except Exception as e:
            for word in stackvm_response_text:
                yield ChatEvent(
                    event_type=ChatEventType.TEXT_PART,
                    payload=word,
                )
            logger.error(f"Failed to refine question: {e}")
        """
        response_text = stackvm_response_text
        base_url = stream_chat_api_url.replace('/api/stream_execute_vm', '')
        db_assistant_message.content = response_text
        db_assistant_message.trace_url = f"{base_url}?task_id={task_id}" if task_id else ""
        db_assistant_message.meta = {
            "task_id": task_id,
            "stackvm_response_text": stackvm_response_text,
            "goal": goal,
        }
        db_assistant_message.updated_at = datetime.now(UTC)
        db_assistant_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_assistant_message)
        db_user_message.trace_url = f"{base_url}?task_id={task_id}" if task_id else ""
        db_user_message.meta = {
            "task_id": task_id,
            "stackvm_response_text": stackvm_response_text,
            "goal": goal,
        }
        db_user_message.updated_at = datetime.now(UTC)
        db_user_message.finished_at = datetime.now(UTC)
        self.db_session.add(db_user_message)
        self.db_session.commit()

        yield ChatEvent(
            event_type=ChatEventType.DATA_PART,
            payload=ChatStreamDataPayload(
                chat=self.db_chat_obj,
                user_message=db_user_message,
                assistant_message=db_assistant_message,
            ),
        )

    def _parse_chat_messages(
            self, chat_messages: List[ChatMessage]
    ) -> tuple[str, List[ChatMessage]]:
        user_question = chat_messages[-1].content
        chat_history = chat_messages[:-1]
        return user_question, chat_history

    def _get_source_documents(self, response: StreamingResponse) -> List[dict]:
        source_nodes_ids = [s_n.node_id for s_n in response.source_nodes]
        stmt = (
            select(
                Chunk.id,
                Document.id,
                Document.name,
                Document.source_uri,
            )
            .outerjoin(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.id.in_(source_nodes_ids),
            )
        )
        source_chunks = self.db_session.execute(stmt).all()
        # Sort the source chunks based on the order of the source_nodes_ids, which are arranged according to their related scores.
        source_chunks = sorted(
            source_chunks, key=lambda x: source_nodes_ids.index(str(x[0]))
        )
        source_documents = []
        source_documents_ids = []
        for s in source_chunks:
            if s[1] not in source_documents_ids:
                source_documents_ids.append(s[1])
                source_documents.append(
                    {
                        "id": s[1],
                        "name": s[2],
                        "source_uri": s[3],
                    }
                )
        return source_documents

    def _post_verification(
            self, user_question: str, response_text: str, chat_id: UUID, message_id: int
    ) -> Optional[str]:
        # post verification to external service, will return the post verification result url
        post_verification_url = self.chat_engine_config.post_verification_url
        post_verification_token = self.chat_engine_config.post_verification_token

        if not post_verification_url:
            return

        external_request_id = f"{chat_id}_{message_id}"
        qa_content = f"User question: {user_question}\n\nAnswer:\n{response_text}"
        try:
            resp = requests.post(
                post_verification_url,
                json={
                    "external_request_id": external_request_id,
                    "qa_content": qa_content,
                },
                headers={
                    "Authorization": f"Bearer {post_verification_token}",
                }
                if post_verification_token
                else {},
                timeout=10,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            return urljoin(f"{post_verification_url}/", str(job_id))
        except Exception:
            logger.exception("Failed to post verification")


def get_prompt_by_jinja2_template(template_string: str, **kwargs) -> PromptTemplate:
    # use jinja2's template because it support complex render logic
    # for example:
    #       {% for e in entities %}
    #           {{ e.name }}
    #       {% endfor %}
    template = (
        jinja2.Template(template_string)
        .render(**kwargs)
        # llama-index will use f-string to format the template
        # so we need to escape the curly braces even if we do not use it
        .replace("{", "{{")
        .replace("}", "}}")
        # This is a workaround to bypass above escape,
        # llama-index will use f-string to format following variables,
        # maybe we can use regex to replace the variable name to make this more robust
        .replace("<<query_str>>", "{query_str}")
        .replace("<<context_str>>", "{context_str}")
        .replace("<<existing_answer>>", "{existing_answer}")
        .replace("<<context_msg>>", "{context_msg}")
    )
    return PromptTemplate(template=template)


def user_can_view_chat(chat: DBChat, user: Optional[User]) -> bool:
    # Anonymous or pulic chat can be accessed by anyone
    # Non-anonymous chat can be accessed by owner or superuser
    if not chat.user_id or chat.visibility == ChatVisibility.PUBLIC:
        return True
    return user is not None and (user.is_superuser or chat.user_id == user.id)


def user_can_edit_chat(chat: DBChat, user: Optional[User]) -> bool:
    if user is None:
        return False
    if user.is_superuser:
        return True
    return chat.user_id == user.id


def get_graph_data_from_langfuse(trace_url: str):
    start_time = time.time()
    langfuse_host = SiteSetting.langfuse_host
    langfuse_secret_key = SiteSetting.langfuse_secret_key
    langfuse_public_key = SiteSetting.langfuse_public_key
    enable_langfuse = langfuse_host and langfuse_secret_key and langfuse_public_key
    current_time = time.time()
    logger.debug(
        f"Graph Load - Fetch langfuse configs from site setting, time cost: {current_time - start_time}s"
    )
    logger.debug(
        f"Graph Load - trace_url: {trace_url}, enable_langfuse: {enable_langfuse}"
    )
    start_time = current_time
    if enable_langfuse and trace_url is not None and trace_url != "":
        langfuse_client = Langfuse(
            secret_key=langfuse_secret_key,
            public_key=langfuse_public_key,
            host=langfuse_host,
        )
        trace_id = trace_url.split("/trace/")[-1]
        ob_data = langfuse_client.fetch_observations(trace_id=trace_id)
        current_time = time.time()
        logger.debug(
            f"Graph Load - Fetch trace({trace_id}) from langfuse, time cost: {current_time - start_time}s"
        )
        start_time = current_time
        all_entities = []
        all_relationships = []

        for obd in ob_data.data:
            if obd.name == MyCBEventType.GRAPH_SEMANTIC_SEARCH:
                for _, sg in obd.output["queries"].items():
                    all_entities.extend(sg["entities"])
                    all_relationships.extend(sg["relationships"])

        unique_entities = {e["id"]: e for e in all_entities}.values()
        unique_relationships = {r["id"]: r for r in all_relationships}.values()

        logger.debug(
            f"Graph Load - Fetch trace({trace_id}) from langfuse, relationships: {len(unique_relationships)}, time cost: {time.time() - start_time}s"
        )

        return list(unique_entities), list(unique_relationships)
    else:
        return [], []


def get_chat_message_subgraph(
        session: Session, chat_message: DBChatMessage
) -> Tuple[List, List]:
    if chat_message.role != MessageRole.USER:
        return [], []

    # try to get subgraph from chat_message.graph_data
    try:
        if (
                chat_message.graph_data
                and "relationships" in chat_message.graph_data
                and len(chat_message.graph_data["relationships"]) > 0
        ):
            relationship_ids = chat_message.graph_data["relationships"]
            all_entities, all_relationships = editor.get_relationship_by_ids(
                session, relationship_ids
            )
            entities = [
                {
                    "id": e.id,
                    "name": e.name,
                    "description": e.description,
                    "meta": e.meta,
                    "entity_type": e.entity_type,
                }
                for e in all_entities
            ]
            relationships = [
                {
                    "id": r.id,
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "description": r.description,
                    "rag_description": f"{r.source_entity.name} -> {r.description} -> {r.target_entity.name}",
                    "meta": r.meta,
                    "weight": r.weight,
                    "last_modified_at": r.last_modified_at,
                }
                for r in all_relationships
            ]
            return entities, relationships
    except Exception as e:
        logger.error(f"Failed to get subgraph from chat_message.graph_data: {e}")

    # try to get subgraph from langfuse trace
    try:
        entities, relationships = get_graph_data_from_langfuse(chat_message.trace_url)
        if len(relationships) > 0:
            return list(entities), list(relationships)
    except Exception as e:
        logger.error(f"Failed to get subgraph from langfuse trace: {e}")

    chat: DBChat = chat_message.chat
    chat_engine_config = ChatEngineConfig.load_from_db(session, chat.engine.name)
    kg_config = chat_engine_config.knowledge_graph
    graph_store = TiDBGraphStore(
        dspy_lm=chat_engine_config.get_fast_dspy_lm(session),
        session=session,
        embed_model=get_default_embedding_model(session),
    )
    entities, relations, _ = graph_store.retrieve_with_weight(
        chat_message.content,
        [],
        depth=kg_config.depth,
        include_meta=kg_config.include_meta,
        with_degree=kg_config.with_degree,
        with_chunks=False,
    )
    return entities, relations


def check_rag_required_config(session: Session) -> tuple[bool]:
    # Check if llm, embedding model, and datasource are configured
    # If any of them is missing, the rag can not work
    has_default_llm = session.scalar(select(func.count(DBLLM.id))) > 0
    has_default_embedding_model = (
            session.scalar(select(func.count(DBEmbeddingModel.id))) > 0
    )
    has_datasource = session.scalar(select(func.count(DBDataSource.id))) > 0
    return has_default_llm, has_default_embedding_model, has_datasource


def check_rag_optional_config(session: Session) -> tuple[bool]:
    langfuse = bool(
        SiteSetting.langfuse_host
        and SiteSetting.langfuse_secret_key
        and SiteSetting.langfuse_public_key
    )
    default_reranker = session.scalar(select(func.count(DBRerankerModel.id))) > 0
    return langfuse, default_reranker


class LLMRecommendQuestions(BaseModel):
    """recommend questions respond model"""
    questions: List[str]


def get_chat_message_recommend_questions(
        db_session: Session,
        chat_message: DBChatMessage,
        engine_name: str = "default",
) -> List[str]:
    chat_engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
    _fast_llm = chat_engine_config.get_fast_llama_llm(db_session)

    statement = (
        select(RecommendQuestion.questions)
        .where(RecommendQuestion.chat_message_id == chat_message.id)
        .with_for_update()  # using write lock in case the same chat message trigger multiple requests
    )

    # questions = db_session.execute(statement).first()
    # if questions is not None:
    #     return questions
    result = db_session.execute(statement).first()
    if result is not None:
        questions = result[0]  # Extract the 'questions' field from the result tuple
        return questions


    recommend_questions = _fast_llm.structured_predict(
        output_cls=LLMRecommendQuestions,
        prompt=get_prompt_by_jinja2_template(
            chat_engine_config.llm.further_questions_prompt,
            chat_message_content=chat_message.content,
        ),
        # Ensure no tool-related parameters are set
        llm_kwargs={
            'tools': None,
            'tool_choice': None,
        }
    )

    db_session.add(RecommendQuestion(
        chat_message_id=chat_message.id,
        questions=recommend_questions.questions,
    ))
    db_session.commit()

    return recommend_questions.questions
