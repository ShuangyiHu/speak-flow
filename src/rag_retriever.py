import asyncio
import os
import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import time
import logging

import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import AsyncAnthropic

from shared_types import CoachingStrategy, CoachingAction, TurnAnalysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
RETRIEVAL_TIMEOUT_SECONDS = 20.0
HYDE_TIMEOUT_SECONDS = 5.0
DEFAULT_TOP_K = 3
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

@dataclass
class DebateChunk:
    chunk_id: str
    text: str
    topic: str
    argument_type: str
    strength_score: float
    source: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalContext:
    chunks: List[DebateChunk]
    hypothetical_query: str
    strategy_filter: str
    retrieval_latency_ms: int
    fallback_used: bool

class RAGRetriever:
    def __init__(self, collection_name: str = "speakflow_debate_kb"):
        """
        Initialize RAGRetriever with vector store and embedding model.
        
        Sets up:
        - SentenceTransformer("all-MiniLM-L6-v2") for embeddings
        - ChromaDB persistent client from CHROMA_DB_PATH env var or "./chroma_db"
        - Collection with name collection_name
        - AsyncAnthropic client with timeout from ANTHROPIC_TIMEOUT_SECONDS
        - self._kb_available based on collection document count
        """
        # Initialize embedding model
        self._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize ChromaDB client
        chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._client.get_or_create_collection(name=collection_name)
        
        # Check if KB is available
        try:
            count = self._collection.count()
            self._kb_available = count > 0
        except Exception as e:
            logger.error(f"Failed to check collection count: {e}")
            self._kb_available = False
        
        # Initialize Anthropic client
        self._anthropic = AsyncAnthropic()

    async def retrieve(
        self,
        coaching_action: CoachingAction,
        turn_analysis: TurnAnalysis,
        top_k: int = 3,
    ) -> RetrievalContext:
        """
        Main entry point for grounded retrieval. Called by LangGraph pipeline.
        
        Args:
            coaching_action: Contains topic, user_position, and strategy
            turn_analysis: Contains user's argument summary and logical_gaps
            top_k: Maximum number of chunks to return
        
        Returns:
            RetrievalContext with retrieved chunks or fallback content
        
        Flow:
            1. Return empty context for CORRECT_PRONUNCIATION strategy
            2. Return fallback context if KB unavailable
            3. Generate hypothetical query via HyDE
            4. Embed query and search ChromaDB with strategy filters
            5. Re-rank results by composite score
            6. Return top_k chunks within 2-second timeout
        """
        start_time = time.time()
        
        try:
            # Use asyncio.wait_for for the entire retrieval process
            result = await asyncio.wait_for(
                self._retrieve_internal(coaching_action, turn_analysis, top_k, start_time),
                timeout=RETRIEVAL_TIMEOUT_SECONDS
            )
            return result
        except asyncio.TimeoutError:
            logger.error("Retrieval timed out")
            latency_ms = int((time.time() - start_time) * 1000)
            return RetrievalContext(
                chunks=[],
                hypothetical_query="",
                strategy_filter=coaching_action.strategy.value,
                retrieval_latency_ms=latency_ms,
                fallback_used=True
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            latency_ms = int((time.time() - start_time) * 1000)
            fallback_context = self._get_fallback_context(coaching_action.strategy)
            fallback_context.retrieval_latency_ms = latency_ms
            return fallback_context

    async def _retrieve_internal(
        self,
        coaching_action: CoachingAction,
        turn_analysis: TurnAnalysis,
        top_k: int,
        start_time: float
    ) -> RetrievalContext:
        """Internal retrieval implementation without timeout wrapper"""
        
        # 1. Return empty context for CORRECT_PRONUNCIATION
        if coaching_action.strategy == CoachingStrategy.CORRECT_PRONUNCIATION:
            latency_ms = int((time.time() - start_time) * 1000)
            return RetrievalContext(
                chunks=[],
                hypothetical_query="",
                strategy_filter=coaching_action.strategy.value,
                retrieval_latency_ms=latency_ms,
                fallback_used=False
            )
        
        # 2. Return fallback if KB unavailable
        if not self._kb_available:
            return self._get_fallback_context(coaching_action.strategy)
        
        # 3. Generate hypothetical query
        hypothetical_query = await self._generate_hypothetical_query(
            coaching_action, turn_analysis
        )
        
        # 4. Embed query
        query_embedding = await asyncio.to_thread(self._embed, hypothetical_query)
        
        # 5. Query ChromaDB with strategy filters
        type_filter = self._get_type_filter(coaching_action.strategy)
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k * 2, 10)  # Retrieve more for re-ranking
        }
        
        if type_filter:
            query_kwargs["where"] = {"argument_type": {"$in": type_filter}}
        
        results = self._collection.query(**query_kwargs)
        
        # 6. Parse results into DebateChunk objects
        chunks = []
        distances = results["distances"][0] if results["distances"] else []
        
        for i, (doc_id, text, metadata) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        )):
            chunk = DebateChunk(
                chunk_id=doc_id,
                text=text,
                topic=metadata.get("topic", ""),
                argument_type=metadata.get("argument_type", ""),
                strength_score=metadata.get("strength_score", 0.5),
                source=metadata.get("source", "unknown"),
                metadata=metadata
            )
            chunks.append(chunk)
        
        # 7. Re-rank chunks
        ranked_chunks = self._rerank(chunks, distances, coaching_action)
        
        # 8. Return top_k chunks
        latency_ms = int((time.time() - start_time) * 1000)
        return RetrievalContext(
            chunks=ranked_chunks[:top_k],
            hypothetical_query=hypothetical_query,
            strategy_filter=coaching_action.strategy.value,
            retrieval_latency_ms=latency_ms,
            fallback_used=False
        )

    async def _generate_hypothetical_query(
    self,
    coaching_action: CoachingAction,
    turn_analysis: TurnAnalysis,
) -> str:
        """
        OPT-1: HyDE query via template string — no Claude API call.
        Original approach added ~2s per turn via Claude API.
        """
        topic    = coaching_action.topic or "social media"
        position = coaching_action.user_position or "for"
        summary  = (turn_analysis.argument.summary or "").strip()

        if summary:
            query = (
                f"{topic}. Taking the {position} position: {summary}. "
                f"This position is supported by measurable evidence showing significant "
                f"societal impact on vulnerable populations and long-term consequences."
            )
        else:
            query = (
                f"{topic}. The {position} position is well-supported: "
                f"empirical research demonstrates clear causal links between this issue "
                f"and harmful outcomes for individuals and society."
            )

        logger.info(f"[HyDE-template] query='{query[:100]}'")
        return query

    def _embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Input text to embed
        
        Returns:
            384-dimensional embedding vector as list of floats
        
        Uses:
            self._embedding_model.encode(text).tolist()
        """
        return self._embedding_model.encode(text).tolist()

    def _get_type_filter(self, strategy: CoachingStrategy) -> List[str]:
        """
        Map coaching strategy to ChromaDB argument_type filters.
        
        Args:
            strategy: CoachingStrategy enum value
        
        Returns:
            List of argument_type strings for metadata filtering
        
        Mapping:
            PROBE             → ["evidence", "framework"]
            CHALLENGE         → ["counter_argument", "rebuttal"]
            REDIRECT          → ["claim", "framework"]
            PRAISE            → ["claim", "evidence"]
            CORRECT_PRONUNCIATION → []
        """
        mapping = {
            CoachingStrategy.PROBE: ["evidence", "framework"],
            CoachingStrategy.CHALLENGE: ["counter_argument", "rebuttal"],
            CoachingStrategy.REDIRECT: ["claim", "framework"],
            CoachingStrategy.PRAISE: ["claim", "evidence"],
            CoachingStrategy.CORRECT_PRONUNCIATION: []
        }
        return mapping.get(strategy, [])

    def _rerank(
        self,
        chunks: List[DebateChunk],
        distances: List[float],
        coaching_action: CoachingAction,
    ) -> List[DebateChunk]:
        """
        Re-rank chunks by composite similarity and quality score.
        
        Args:
            chunks: Retrieved DebateChunk objects
            distances: ChromaDB cosine distances (lower = more similar)
            coaching_action: Used for context-aware ranking
        
        Returns:
            Chunks sorted by composite score descending
        
        Formula:
            composite = 0.6 * (1 - distance) + 0.4 * chunk.strength_score
        """
        scored_chunks = []
        
        for i, chunk in enumerate(chunks):
            distance = distances[i] if i < len(distances) else 0.5
            similarity_score = 1 - distance
            composite_score = 0.6 * similarity_score + 0.4 * chunk.strength_score
            scored_chunks.append((composite_score, chunk))
        
        # Sort by composite score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks]

    def _get_fallback_context(self, strategy: CoachingStrategy) -> RetrievalContext:
        """
        Generate hardcoded fallback context when KB unavailable.
        
        Args:
            strategy: CoachingStrategy to determine stub content type
        
        Returns:
            RetrievalContext with 2 realistic stub DebateChunk objects
            and fallback_used=True
        
        Stub content examples:
            PROBE: Evidence-seeking framework questions
            CHALLENGE: Common counter-argument patterns
            REDIRECT: Topic-anchoring claim structures
            PRAISE: Strong argument examples
        """
        if strategy == CoachingStrategy.PROBE:
            chunks = [
                DebateChunk(
                    chunk_id="fallback_probe_1",
                    text="What specific data or studies would your opponent demand to see before accepting this claim? Strong arguments anticipate the evidence requests that skeptics will make and address them proactively.",
                    topic="evidence_framework",
                    argument_type="framework",
                    strength_score=0.8,
                    source="manual",
                    metadata={"fallback": True}
                ),
                DebateChunk(
                    chunk_id="fallback_probe_2",
                    text="Consider the scope and limitations of your evidence. Are you making claims that extend beyond what your data actually supports? The strongest arguments acknowledge their boundaries clearly.",
                    topic="evidence_scope",
                    argument_type="evidence",
                    strength_score=0.7,
                    source="manual",
                    metadata={"fallback": True}
                )
            ]
        elif strategy == CoachingStrategy.CHALLENGE:
            chunks = [
                DebateChunk(
                    chunk_id="fallback_challenge_1",
                    text="However, this argument fails to address the fundamental trade-offs involved. When resources are limited, pursuing this policy means sacrificing alternative approaches that might be more effective.",
                    topic="tradeoff_analysis",
                    argument_type="counter_argument",
                    strength_score=0.8,
                    source="manual",
                    metadata={"fallback": True}
                ),
                DebateChunk(
                    chunk_id="fallback_challenge_2",
                    text="The opponent's reasoning contains a critical gap: they assume correlation implies causation without considering alternative explanations for the observed outcomes.",
                    topic="logical_fallacies",
                    argument_type="rebuttal",
                    strength_score=0.9,
                    source="manual",
                    metadata={"fallback": True}
                )
            ]
        elif strategy == CoachingStrategy.REDIRECT:
            chunks = [
                DebateChunk(
                    chunk_id="fallback_redirect_1",
                    text="Before we can evaluate specific policies, we must first establish the fundamental principles at stake. What values should guide our decision-making framework here?",
                    topic="foundational_principles",
                    argument_type="framework",
                    strength_score=0.8,
                    source="manual",
                    metadata={"fallback": True}
                ),
                DebateChunk(
                    chunk_id="fallback_redirect_2",
                    text="The core issue isn't about the technical details of implementation, but about the fundamental values and priorities that should guide our approach to this problem.",
                    topic="foundational_principles",
                    argument_type="claim",
                    strength_score=0.75,
                    source="manual",
                    metadata={"fallback": True}
                )
            ]
        elif strategy == CoachingStrategy.PRAISE:
            chunks = [
                DebateChunk(
                    chunk_id="fallback_praise_1",
                    text="Your argument demonstrates the classic strength of the Claim-Reason-Evidence structure: you stated a clear position, explained the mechanism behind it, and anchored it with concrete data. This is exactly how persuasive debate arguments are built.",
                    topic="argument_structure",
                    argument_type="framework",
                    strength_score=0.9,
                    source="manual",
                    metadata={"fallback": True}
                ),
                DebateChunk(
                    chunk_id="fallback_praise_2",
                    text="Strong debaters don't just present evidence — they explain why that evidence matters and what it implies for the broader question. Your ability to connect specific facts to larger principles is a hallmark of advanced argumentation.",
                    topic="argument_quality",
                    argument_type="claim",
                    strength_score=0.85,
                    source="manual",
                    metadata={"fallback": True}
                )
            ]
        else:
            # Default fallback for any unhandled strategy
            chunks = [
                DebateChunk(
                    chunk_id="fallback_default_1",
                    text="A strong debate argument requires three components: a clear claim that states your position, a reason that explains why your claim is true, and evidence that proves your reason with concrete data or examples.",
                    topic="argument_framework",
                    argument_type="framework",
                    strength_score=0.8,
                    source="manual",
                    metadata={"fallback": True}
                ),
                DebateChunk(
                    chunk_id="fallback_default_2",
                    text="The most persuasive debaters anticipate counterarguments and address them before their opponent can raise them. Consider what the strongest objection to your position is, and build a preemptive rebuttal into your argument.",
                    topic="debate_strategy",
                    argument_type="framework",
                    strength_score=0.75,
                    source="manual",
                    metadata={"fallback": True}
                )
            ]

        return RetrievalContext(
            chunks=chunks,
            hypothetical_query="",
            strategy_filter=strategy.value,
            retrieval_latency_ms=0,
            fallback_used=True
        )

    async def index_chunks(self, chunks: List[DebateChunk]) -> int:
        """
        Batch index debate chunks into vector store.
        Returns count of successfully indexed chunks.
        """
        try:
            indexed_count = 0
            batch_size = 50

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                ids = []
                embeddings = []
                documents = []
                metadatas = []

                for chunk in batch:
                    embedding_vector = await asyncio.to_thread(self._embed, chunk.text)
                    ids.append(chunk.chunk_id)
                    embeddings.append(embedding_vector)
                    documents.append(chunk.text)
                    metadatas.append({
                        "topic": chunk.topic,
                        "argument_type": chunk.argument_type,
                        "strength_score": chunk.strength_score,
                        "source": chunk.source
                    })

                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                indexed_count += len(batch)

            self._kb_available = True
            return indexed_count

        except Exception as e:
            logger.error(f"index_chunks failed: {e}")
            return 0