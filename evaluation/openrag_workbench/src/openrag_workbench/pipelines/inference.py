"""OpenRAG inference pipeline implementation for ragbench."""

import asyncio
import logging
import traceback
from typing import Any

from openrag_sdk import ContentEvent, DoneEvent, OpenRAGClient, SourcesEvent
from openrag_sdk.models import SettingsUpdateOptions
from pydantic import BaseModel, Field
from ragworkbench.api.inference import InferenceParams, InferencePipeline
from ragworkbench.api.inference_result import InferenceResult, Trajectory
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.boards.board_registry import inference_pipeline
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry

from openrag_workbench.pipelines.ingest import OpenRAGIngestArtifact

logger = logging.getLogger(__name__)


class OpenRagAnswer(BaseModel):
    """Structured answer returned from OpenRAG streaming inference."""

    answer: str
    context_ids: list[str]
    trajectory: Trajectory | None = None
    partial_answer: bool = False


class GenerativeModelParams(BaseModel):
    """Parameters for generative model configuration."""

    provider_id: str = Field(
        description="LLM provider identifier (e.g., 'anthropic', 'openai')"
    )
    model_id: str = Field(
        description="Model identifier (e.g., 'claude-3-5-sonnet-20241022')"
    )


class OpenRAGInferenceParams(InferenceParams):
    """Parameters for OpenRAG inference pipeline."""

    generative_model: GenerativeModelParams = Field(
        description="Generative model configuration"
    )
    timeout: float = Field(
        default=300.0,
        description="HTTP request timeout in seconds (default: 5 minutes)",
    )


@inference_pipeline(name="openrag", params_class=OpenRAGInferenceParams)
class OpenRAGInference(InferencePipeline):
    """Inference pipeline for OpenRAG backend."""

    def __init__(
        self,
        params: OpenRAGInferenceParams,
        cache_dir: str | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ) -> None:
        """
        Initialize OpenRAG inference pipeline.

        Args:
            params: OpenRAG inference parameters
            cache_dir: Optional directory for caching generation results
            cache_mode: Cache operation mode (on/off/refresh)
        """
        super().__init__(params, cache_dir=cache_dir, cache_mode=cache_mode)
        self.params: OpenRAGInferenceParams = params
        self._ingest_artifact: OpenRAGIngestArtifact | None = None

    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        """
        Set the ingest artifacts from the ingestion pipeline.

        Args:
            ingest_artifacts: List of ingest artifacts (expects single OpenRAGIngestArtifact)
        """
        if len(ingest_artifacts) != 1:
            raise ValueError(
                f"Expected exactly 1 ingest artifact, got {len(ingest_artifacts)}"
            )

        artifact = ingest_artifacts[0]
        if not isinstance(artifact, OpenRAGIngestArtifact):
            raise TypeError(
                f"Expected OpenRAGIngestArtifact, got {type(artifact).__name__}"
            )

        self._ingest_artifact = artifact
        logger.info(
            f"Set ingest artifact with index: {artifact.index_name}"
        )

        # Configure generative model settings using async SDK client
        asyncio.run(self._configure_generative_model_async(artifact))

    def _get_additional_cache_params(self) -> dict[str, Any] | None:
        """
        Include index_name in the cache key to differentiate between different indices.

        Returns:
            Dictionary with index_name if artifact is set, None otherwise
        """
        if self._ingest_artifact is not None:
            return {"index_name": self._ingest_artifact.index_name}
        return None
    @staticmethod
    def _is_invalid_answer(answer: str) -> bool:
        """
        Check if an answer contains error patterns that indicate an invalid response.

        Args:
            answer: The answer text to validate

        Returns:
            True if the answer is invalid, False otherwise
        """
        if not answer or not answer.strip():
            return True

        # Specific error patterns that indicate invalid answers
        invalid_patterns = [
            "Timeout updating tool list",
            "litellm.InternalServerError",
        ]

        # Check if answer contains these error patterns
        for pattern in invalid_patterns:
            if pattern in answer:
                return True

        return False


    async def _configure_generative_model_async(
        self, artifact: OpenRAGIngestArtifact
    ) -> None:
        """Configure the generative model settings using SDK client.

        Args:
            artifact: The ingest artifact containing configuration
        """
        logger.info("Configuring generative model settings")

        # Use SDK client for LLM configuration and index_name (URL from environment)
        async with OpenRAGClient(timeout=self.params.timeout) as sdk_client:
            settings_dict = {
                "llm_provider": self.params.generative_model.provider_id,
                "llm_model": self.params.generative_model.model_id,
                "index_name": artifact.index_name,
            }
            
            # Add tracking API key if provided (for cost tracking)
            if self.params.tracking_api_key:
                settings_dict["openai_api_key"] = self.params.tracking_api_key
                logger.info("Set tracking API key for cost tracking")
            
            logger.info(f"Updating settings with: {settings_dict}")
            settings_options = SettingsUpdateOptions(**settings_dict)
            await sdk_client.settings.update(settings_options)

            # Verify settings were applied correctly
            logger.info("Verifying settings were applied correctly...")
            current_settings = await sdk_client.settings.get()
            
            # Check each setting that was sent
            # All settings are in the knowledge section, except index_name and openai_api_key which aren't returned
            mismatches = []
            for key, expected_value in settings_dict.items():
                if key in ("index_name", "openai_api_key"):
                    # index_name, openai_api_key are not returned by the settings endpoint, skip verification
                    continue
                
                # All other fields are in the knowledge section
                actual_value = getattr(current_settings.agent, key, None)
                if actual_value != expected_value:
                    mismatches.append(
                        f"{key}: expected={expected_value}, actual={actual_value}"
                    )
            
            if mismatches:
                error_msg = f"Settings verification failed. Mismatches: {', '.join(mismatches)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Settings verification successful - all values match.")
            
            # Wait for settings to propagate through the system
            logger.info("Waiting 90 seconds for settings to propagate...")
            await asyncio.sleep(90)
            logger.info("Settings propagation wait complete.")

    def process_no_cache(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """
        Process a single benchmark entry without using cache.

        Args:
            benchmark_entry: The benchmark entry to process

        Returns:
            InferenceResult with answer, retrieved context, and trajectory
        """
        if self._ingest_artifact is None:
            raise RuntimeError(
                "Ingest artifacts not set. Call set_ingest_artifacts() first."
            )

        question = benchmark_entry.question
        logger.debug(f"Processing question: {question}")

        # Perform inference using async SDK client
        openrag_answer = asyncio.run(self._infer_single_async(question))

        if openrag_answer.partial_answer:
            logger.warning(
                "Returning inference result with partial answer due to interrupted stream"
            )

        # Build inference result
        return InferenceResult(
            answer=openrag_answer.answer,
            context_ids=openrag_answer.context_ids,
            trajectory=openrag_answer.trajectory,
            **benchmark_entry.model_dump(),
        )

    @staticmethod
    async def _stream_chat_response(
        sdk_client: OpenRAGClient, question: str
    ) -> OpenRagAnswer:
        """
        Stream chat response from SDK client and parse events.

        Args:
            sdk_client: The OpenRAG SDK client
            question: The question to ask

        Returns:
            Structured answer with answer text, context ids, trajectory, and partial flag
        """
        # Stream the chat response
        answer_parts = []
        context_ids: list[str] = []
        trajectory: list[dict[str, Any]] = []
        is_partial_answer = False

        # Use async context manager to properly initialize the stream
        async with sdk_client.chat.stream(message=question) as stream:
            try:
                async for event in stream:
                    # Log each event received
                    logger.debug(f"Received event: {type(event).__name__} - {event}")

                    # Handle different event types from the stream
                    if isinstance(event, ContentEvent):
                        # Content event contains the answer text (delta field)
                        answer_parts.append(event.delta)
                    elif isinstance(event, SourcesEvent):
                        # Sources event contains retrieved documents
                        # Each SourcesEvent represents a query's results
                        current_sources = [
                            {"filename": source.filename, "text": source.text}
                            for source in event.sources
                        ]
                        context_ids.extend(
                            source.filename
                            for source in event.sources
                            if source.filename
                        )
                        # Add this query's results to the list
                        # Use the query field from the SourcesEvent
                        trajectory.append(
                            {
                                "query": event.query,
                                "results": current_sources,
                            }
                        )
                    elif isinstance(event, DoneEvent):
                        # Done event signals completion
                        logger.debug("Stream completed")
            except Exception as exc:
                partial_answer_text = "".join(answer_parts)
                if partial_answer_text or trajectory:
                    is_partial_answer = True
                    logger.warning(
                        "Stream interrupted; returning partial response. "
                        f"error={exc}, partial_answer_length={len(partial_answer_text)}, "
                        f"trajectory_count={len(trajectory)}, "
                        f"partial_answer=\n{partial_answer_text}\n"
                    )
                else:
                    raise

        answer = "".join(answer_parts)
        return OpenRagAnswer(
            answer=answer,
            context_ids=context_ids,
            trajectory=trajectory,
            partial_answer=is_partial_answer,
        )

    async def _infer_single_async(
        self, question: str
    ) -> OpenRagAnswer:
        """
        Perform inference for a single question using SDK client.

        Args:
            question: The question to answer

        Returns:
            Structured answer with answer text, context ids, trajectory, and partial flag
        """
        if self._ingest_artifact is None:
            raise RuntimeError(
                "Ingest artifact not set. Call set_ingest_artifacts() first."
            )

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"Inference attempt {attempt}/{max_retries} for question: {question}"
                )

                # Use SDK client for chat/inference (URL from environment)
                async with OpenRAGClient(timeout=self.params.timeout) as sdk_client:
                    openrag_answer = await self._stream_chat_response(
                        sdk_client, question
                    )

                    # Check for common invalid answer patterns, only on non-partial answers. 
                    # Partial answers may be empty and that is ok.
                    if not openrag_answer.partial_answer and self._is_invalid_answer(openrag_answer.answer):
                        raise ValueError(
                            f"Invalid answer detected: '{openrag_answer.answer[:100]}'.... "
                            "Answer appears to be an error message or non-response."
                        )

                    if openrag_answer.partial_answer and attempt < max_retries:
                        raise RuntimeError(
                            f"Partial answer received in attemot {attempt}, before final retry attempt; retrying"
                        )

                    return openrag_answer

            except Exception as e:
                logger.error(
                    f"Inference attempt {attempt}/{max_retries} failed: {e}\n{traceback.format_exc()}"
                )
                if attempt < max_retries:
                    # Wait before retry (exponential backoff)
                    await asyncio.sleep(2**attempt)

        error_msg = f"Cannot run inference on question '{question}' after {max_retries} attempts. See log for details."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
