"""Qdrant utilities for message and archival memory storage."""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Tuple

from letta.constants import DEFAULT_EMBEDDING_CHUNK_SIZE
from letta.errors import LettaInvalidArgumentError
from letta.otel.tracing import trace_method
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import MessageRole, TagMatchMode
from letta.schemas.passage import Passage as PydanticPassage
from letta.settings import model_settings, settings

logger = logging.getLogger(__name__)


def should_use_qdrant() -> bool:
    """Check if Qdrant should be used as the vector database provider."""
    # We need OpenAI since we default to their embedding model
    return (
        settings.vector_db_provider == "qdrant"
        and bool(settings.qdrant_url)
        and bool(model_settings.openai_api_key)
    )


def should_use_qdrant_for_messages() -> bool:
    """Check if Qdrant should be used for messages."""
    return should_use_qdrant() and bool(settings.embed_all_messages)


def should_use_qdrant_for_tools() -> bool:
    """Check if Qdrant should be used for tools."""
    return should_use_qdrant() and bool(settings.embed_tools)


class QdrantClient:
    """Client for managing archival memory and messages with Qdrant vector database."""

    default_embedding_config = EmbeddingConfig(
        embedding_model="text-embedding-3-small",
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        embedding_dim=1536,
        embedding_chunk_size=DEFAULT_EMBEDDING_CHUNK_SIZE,
    )

    def __init__(self, url: str = None, api_key: str = None, prefer_grpc: bool = None):
        """Initialize Qdrant client."""
        from qdrant_client import AsyncQdrantClient

        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key
        self.prefer_grpc = prefer_grpc if prefer_grpc is not None else settings.qdrant_prefer_grpc

        from letta.services.agent_manager import AgentManager
        from letta.services.archive_manager import ArchiveManager

        self.archive_manager = ArchiveManager()
        self.agent_manager = AgentManager()

        if not self.url:
            raise ValueError("Qdrant URL not provided")

        # Create async client
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
        )

    @trace_method
    async def _generate_embeddings(self, texts: List[str], actor: "PydanticUser") -> List[List[float]]:
        """Generate embeddings using the default embedding configuration.

        Args:
            texts: List of texts to embed
            actor: User actor for embedding generation

        Returns:
            List of embedding vectors
        """
        from letta.llm_api.llm_client import LLMClient

        # filter out empty strings after stripping
        filtered_texts = [text for text in texts if text.strip()]

        # skip embedding if no valid texts
        if not filtered_texts:
            return []

        embedding_client = LLMClient.create(
            provider_type=self.default_embedding_config.embedding_endpoint_type,
            actor=actor,
        )
        embeddings = await embedding_client.request_embeddings(filtered_texts, self.default_embedding_config)
        return embeddings

    @trace_method
    async def _get_archive_namespace_name(self, archive_id: str) -> str:
        """Get namespace name for a specific archive."""
        return await self.archive_manager.get_or_set_vector_db_namespace_async(archive_id)

    @trace_method
    async def _get_message_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for messages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for messages
        """
        environment = settings.environment
        if environment:
            namespace_name = f"messages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"messages_{organization_id}"

        return namespace_name

    @trace_method
    async def _get_tool_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for tools (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for tools
        """
        environment = settings.environment
        if environment:
            namespace_name = f"tools_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"tools_{organization_id}"

        return namespace_name

    @trace_method
    async def _get_file_passages_namespace_name(self, organization_id: str) -> str:
        """Get namespace name for file passages (org-scoped).

        Args:
            organization_id: Organization ID for namespace generation

        Returns:
            The org-scoped namespace name for file passages
        """
        environment = settings.environment
        if environment:
            namespace_name = f"file_passages_{organization_id}_{environment.lower()}"
        else:
            namespace_name = f"file_passages_{organization_id}"

        return namespace_name

    def _build_qdrant_filter(self, turbopuffer_filter):
        """Convert Turbopuffer-style filter to Qdrant filter format.

        Args:
            turbopuffer_filter: Turbopuffer filter tuple or None

        Returns:
            Qdrant Filter object or None
        """
        from qdrant_client import models

        if turbopuffer_filter is None:
            return None

        # Handle tuple format: ("field", "operator", value)
        if isinstance(turbopuffer_filter, tuple):
            if len(turbopuffer_filter) == 2 and turbopuffer_filter[0] == "And":
                # ("And", [filters])
                sub_filters = [self._build_qdrant_filter(f) for f in turbopuffer_filter[1]]
                return models.Filter(must=[f for f in sub_filters if f is not None])

            field, operator, value = turbopuffer_filter

            if operator == "Eq":
                if value is None:
                    return models.Filter(
                        must=[
                            models.IsNullCondition(
                                is_null=models.PayloadField(key=field),
                            )
                        ]
                    )
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        )
                    ]
                )
            elif operator == "In":
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=value),
                        )
                    ]
                )
            elif operator == "Gte":
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            range=models.Range(gte=value),
                        )
                    ]
                )
            elif operator == "Lte":
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            range=models.Range(lte=value),
                        )
                    ]
                )
            elif operator == "ContainsAny":
                # For array fields
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            match=models.MatchAny(any=value),
                        )
                    ]
                )
            elif operator == "Contains":
                # For single value in array
                return models.Filter(
                    must=[
                        models.FieldCondition(
                            key=field,
                            match=models.MatchValue(value=value),
                        )
                    ]
                )

        return None

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Any],
        fts_results: List[Any],
        get_id_func: Callable[[Any], str],
        vector_weight: float,
        fts_weight: float,
        top_k: int,
    ) -> List[Tuple[Any, float, dict]]:
        """RRF implementation that works with any object type.

        RRF score = vector_weight * (1/(k + rank)) + fts_weight * (1/(k + rank))
        where k is a constant (typically 60) to avoid division by zero

        This is a pure rank-based fusion following the standard RRF algorithm.

        Args:
            vector_results: List of items from vector search (ordered by relevance)
            fts_results: List of items from FTS (ordered by relevance)
            get_id_func: Function to extract ID from an item
            vector_weight: Weight for vector search results
            fts_weight: Weight for FTS results
            top_k: Number of results to return

        Returns:
            List of (item, score, metadata) tuples sorted by RRF score
            metadata contains ranks from each result list
        """
        k = 60  # standard RRF constant from Cormack et al. (2009)

        # create rank mappings based on position in result lists
        # rank starts at 1, not 0
        vector_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(vector_results)}
        fts_ranks = {get_id_func(item): rank + 1 for rank, item in enumerate(fts_results)}

        # combine all unique items from both result sets
        all_items = {}
        for item in vector_results:
            all_items[get_id_func(item)] = item
        for item in fts_results:
            all_items[get_id_func(item)] = item

        # calculate RRF scores based purely on ranks
        rrf_scores = {}
        score_metadata = {}
        for item_id in all_items:
            # RRF formula: sum of 1/(k + rank) across result lists
            # If item not in a list, we don't add anything (equivalent to rank = infinity)
            vector_rrf_score = 0.0
            fts_rrf_score = 0.0

            if item_id in vector_ranks:
                vector_rrf_score = vector_weight / (k + vector_ranks[item_id])
            if item_id in fts_ranks:
                fts_rrf_score = fts_weight / (k + fts_ranks[item_id])

            combined_score = vector_rrf_score + fts_rrf_score

            rrf_scores[item_id] = combined_score
            score_metadata[item_id] = {
                "combined_score": combined_score,  # Final RRF score
                "vector_rank": vector_ranks.get(item_id),
                "fts_rank": fts_ranks.get(item_id),
            }

        # sort by RRF score and return with metadata
        sorted_results = sorted(
            [(all_items[iid], score, score_metadata[iid]) for iid, score in rrf_scores.items()], key=lambda x: x[1], reverse=True
        )

        return sorted_results[:top_k]

    async def _ensure_collection_exists(self, collection_name: str, vector_size: int = 1536):
        """Ensure a collection exists, create if it doesn't.

        Args:
            collection_name: Name of the collection
            vector_size: Size of the vector embeddings
        """
        from qdrant_client import models
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            # Check if collection exists
            await self.client.get_collection(collection_name=collection_name)
        except (UnexpectedResponse, Exception) as e:
            # Collection doesn't exist, create it
            logger.info(f"Creating Qdrant collection: {collection_name}")
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            # Create text index for full-text search
            await self.client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=20,
                ),
            )

    @trace_method
    async def insert_messages(
        self,
        messages: List["PydanticMessage"],
        organization_id: str,
        actor: "PydanticUser",
    ) -> bool:
        """Insert messages into Qdrant.

        Args:
            messages: List of messages to store
            organization_id: Organization ID for the messages
            actor: User actor for embedding generation

        Returns:
            True if successful
        """
        from qdrant_client import models

        if not messages:
            return True

        # Extract text and filter out empty content
        message_texts = []
        valid_messages = []
        for message in messages:
            # Get the text content from the message
            if hasattr(message, "text") and message.text:
                text = message.text
            elif hasattr(message, "content") and message.content:
                text = message.content
            else:
                continue

            if text.strip():
                message_texts.append(text)
                valid_messages.append(message)

        if not valid_messages:
            logger.warning("All messages had empty text content, skipping insertion")
            return True

        # Generate embeddings
        embeddings = await self._generate_embeddings(message_texts, actor)

        namespace_name = await self._get_message_namespace_name(organization_id)

        # Ensure collection exists
        await self._ensure_collection_exists(namespace_name)

        # Prepare points for Qdrant
        points = []
        for message, text, embedding in zip(valid_messages, message_texts, embeddings):
            payload = {
                "text": text,
                "role": message.role.value if hasattr(message.role, "value") else str(message.role),
                "agent_id": message.agent_id,
                "organization_id": organization_id,
                "created_at": message.created_at.isoformat() if message.created_at else datetime.now(timezone.utc).isoformat(),
            }

            # Add optional fields if present
            if hasattr(message, "project_id") and message.project_id:
                payload["project_id"] = message.project_id
            if hasattr(message, "template_id") and message.template_id:
                payload["template_id"] = message.template_id
            if hasattr(message, "conversation_id") and message.conversation_id:
                payload["conversation_id"] = message.conversation_id

            points.append(
                models.PointStruct(
                    id=message.id,
                    vector=embedding,
                    payload=payload,
                )
            )

        try:
            # Upsert points to Qdrant
            await self.client.upsert(
                collection_name=namespace_name,
                points=points,
            )
            logger.info(f"Successfully inserted {len(points)} messages into Qdrant collection {namespace_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert messages into Qdrant: {e}")
            raise

    @trace_method
    async def query_messages_by_org_id(
        self,
        organization_id: str,
        actor: "PydanticUser",
        query_text: Optional[str] = None,
        search_mode: str = "hybrid",  # "vector", "fts", "hybrid"
        top_k: int = 10,
        roles: Optional[List[MessageRole]] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        template_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        vector_weight: float = 0.5,
        fts_weight: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Tuple[dict, float, dict]]:
        """Query messages from Qdrant across an entire organization.

        Args:
            organization_id: Organization ID for namespace lookup (required)
            actor: User actor for embedding generation
            query_text: Text query for search
            search_mode: Search mode - "vector", "fts", or "hybrid"
            top_k: Number of results to return
            roles: Optional list of message roles to filter by
            agent_id: Optional agent ID to filter messages by
            project_id: Optional project ID to filter messages by
            template_id: Optional template ID to filter messages by
            conversation_id: Optional conversation ID to filter messages by
            vector_weight: Weight for vector search results in hybrid mode
            fts_weight: Weight for FTS results in hybrid mode
            start_date: Optional datetime to filter messages created after this date
            end_date: Optional datetime to filter messages created on or before this date

        Returns:
            List of (message_dict, score, metadata) tuples
        """
        from qdrant_client import models

        namespace_name = await self._get_message_namespace_name(organization_id)

        # Ensure collection exists
        try:
            await self.client.get_collection(collection_name=namespace_name)
        except Exception:
            # Collection doesn't exist, return empty results
            return []

        # Generate embedding for vector/hybrid search
        query_embedding = None
        if query_text and search_mode in ["vector", "hybrid"]:
            embeddings = await self._generate_embeddings([query_text], actor)
            query_embedding = embeddings[0] if embeddings else None

        # Build filters
        filter_conditions = []

        if roles:
            role_values = [r.value for r in roles]
            if len(role_values) == 1:
                filter_conditions.append(("role", "Eq", role_values[0]))
            else:
                filter_conditions.append(("role", "In", role_values))

        if agent_id:
            filter_conditions.append(("agent_id", "Eq", agent_id))

        if project_id:
            filter_conditions.append(("project_id", "Eq", project_id))

        if template_id:
            filter_conditions.append(("template_id", "Eq", template_id))

        if conversation_id == "default":
            filter_conditions.append(("conversation_id", "Eq", None))
        elif conversation_id is not None:
            filter_conditions.append(("conversation_id", "Eq", conversation_id))

        if start_date:
            if start_date.tzinfo is not None:
                start_date = start_date.astimezone(timezone.utc)
            filter_conditions.append(("created_at", "Gte", start_date.isoformat()))

        if end_date:
            if end_date.hour == 0 and end_date.minute == 0 and end_date.second == 0 and end_date.microsecond == 0:
                from datetime import timedelta

                end_date = end_date + timedelta(days=1) - timedelta(microseconds=1)
            if end_date.tzinfo is not None:
                end_date = end_date.astimezone(timezone.utc)
            filter_conditions.append(("created_at", "Lte", end_date.isoformat()))

        # Combine filters
        final_filter = None
        if len(filter_conditions) == 1:
            final_filter = self._build_qdrant_filter(filter_conditions[0])
        elif len(filter_conditions) > 1:
            final_filter = self._build_qdrant_filter(("And", filter_conditions))

        try:
            if search_mode == "vector":
                # Vector search
                if not query_embedding:
                    return []

                results = await self.client.query_points(
                    collection_name=namespace_name,
                    query=query_embedding,
                    query_filter=final_filter,
                    limit=top_k,
                    with_payload=True,
                )

                # Process results
                output = []
                for idx, point in enumerate(results.points):
                    message_dict = {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "role": point.payload.get("role"),
                        "agent_id": point.payload.get("agent_id"),
                        "created_at": point.payload.get("created_at"),
                    }
                    metadata = {
                        "combined_score": point.score,
                        "vector_rank": idx + 1,
                    }
                    output.append((message_dict, point.score, metadata))

                return output

            elif search_mode == "fts":
                # Full-text search
                if not query_text:
                    return []

                # Build text search filter
                text_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="text",
                            match=models.MatchText(text=query_text),
                        )
                    ]
                )

                # Combine with other filters
                if final_filter:
                    combined_conditions = final_filter.must + text_filter.must
                    text_filter = models.Filter(must=combined_conditions)

                results = await self.client.query_points(
                    collection_name=namespace_name,
                    query_filter=text_filter,
                    limit=top_k,
                    with_payload=True,
                )

                # Process results
                output = []
                for idx, point in enumerate(results.points):
                    message_dict = {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "role": point.payload.get("role"),
                        "agent_id": point.payload.get("agent_id"),
                        "created_at": point.payload.get("created_at"),
                    }
                    metadata = {
                        "combined_score": 1.0 / (idx + 1),  # Simple rank-based score
                        "fts_rank": idx + 1,
                    }
                    output.append((message_dict, metadata["combined_score"], metadata))

                return output

            elif search_mode == "hybrid":
                # Hybrid search - perform both searches and combine with RRF
                if not query_text or not query_embedding:
                    return []

                # Vector search
                vector_results = await self.client.query_points(
                    collection_name=namespace_name,
                    query=query_embedding,
                    query_filter=final_filter,
                    limit=top_k * 2,  # Get more results for fusion
                    with_payload=True,
                )

                # Full-text search
                text_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="text",
                            match=models.MatchText(text=query_text),
                        )
                    ]
                )
                if final_filter:
                    combined_conditions = final_filter.must + text_filter.must
                    text_filter = models.Filter(must=combined_conditions)

                fts_results = await self.client.query_points(
                    collection_name=namespace_name,
                    query_filter=text_filter,
                    limit=top_k * 2,
                    with_payload=True,
                )

                # Convert to message dicts
                vector_messages = []
                for point in vector_results.points:
                    message_dict = {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "role": point.payload.get("role"),
                        "agent_id": point.payload.get("agent_id"),
                        "created_at": point.payload.get("created_at"),
                    }
                    vector_messages.append(message_dict)

                fts_messages = []
                for point in fts_results.points:
                    message_dict = {
                        "id": point.id,
                        "text": point.payload.get("text", ""),
                        "role": point.payload.get("role"),
                        "agent_id": point.payload.get("agent_id"),
                        "created_at": point.payload.get("created_at"),
                    }
                    fts_messages.append(message_dict)

                # Apply RRF
                results_with_metadata = self._reciprocal_rank_fusion(
                    vector_results=vector_messages,
                    fts_results=fts_messages,
                    get_id_func=lambda msg_dict: msg_dict["id"],
                    vector_weight=vector_weight,
                    fts_weight=fts_weight,
                    top_k=top_k,
                )

                return results_with_metadata

        except Exception as e:
            logger.error(f"Failed to query messages from Qdrant: {e}")
            raise

        return []

    @trace_method
    async def delete_messages(self, agent_id: str, organization_id: str, message_ids: List[str]) -> bool:
        """Delete multiple messages from Qdrant.

        Args:
            agent_id: Agent ID (for logging)
            organization_id: Organization ID for namespace lookup
            message_ids: List of message IDs to delete

        Returns:
            True if successful
        """
        if not message_ids:
            return True

        namespace_name = await self._get_message_namespace_name(organization_id)

        try:
            await self.client.delete(
                collection_name=namespace_name,
                points_selector=message_ids,
            )
            logger.info(f"Successfully deleted {len(message_ids)} messages from Qdrant for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete messages from Qdrant: {e}")
            raise

    @trace_method
    async def delete_all_messages(self, agent_id: str, organization_id: str) -> bool:
        """Delete all messages for an agent from Qdrant.

        Args:
            agent_id: Agent ID to filter by
            organization_id: Organization ID for namespace lookup

        Returns:
            True if successful
        """
        from qdrant_client import models

        namespace_name = await self._get_message_namespace_name(organization_id)

        try:
            # Delete by filter
            await self.client.delete(
                collection_name=namespace_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="agent_id",
                                match=models.MatchValue(value=agent_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"Successfully deleted all messages for agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all messages from Qdrant: {e}")
            raise
