"""
RAG Pipeline for document-based Q&A with Multi-Agent Architecture.

This is the main orchestrator that coordinates:
- Triage Agent: Document identification + language detection
- Query Agent: RAG response generation
- ChromaDB: Vector search with document filtering

IMPORTANT: Run ingest.py first to populate the vector database.
"""

from typing import Optional

import logfire

from .agents import create_triage_agent, create_query_agent
from .agents.triage import build_triage_prompt
from .agents.query import build_query_prompt
from .config import Valves, get_default_valves, CHROMA_DIR, CHROMA_HOST, LLM_FALLBACK_MODEL
from .context import check_clarification_context, extract_conversation_context
from .database import VectorStore
from .llm import run_agent_with_fallback
from .models import Language, ProductInfo


class Pipeline:
    """
    RAG Pipeline for document-based Q&A.

    Thin orchestrator following Single Responsibility Principle.
    All domain logic is delegated to specialized modules.
    """

    def __init__(self):
        """Initialize the pipeline."""
        self.name = "Bot Tecnico - Documentos"
        self.valves = get_default_valves()

        # Lazy-initialized components
        self._vector_store: Optional[VectorStore] = None
        self._triage_agent = None
        self._available_products: list[str] = []
        self._products: dict[str, ProductInfo] = {}
        self._initialized = False

    # Re-export Valves for Open WebUI discovery
    Valves = Valves

    def _ensure_initialized(self) -> None:
        """Lazy initialization of components."""
        if self._initialized:
            return

        self._configure_logging()
        self._validate_config()
        self._init_vector_store()
        self._init_agents()

        self._initialized = True
        logfire.info(
            f"Pipeline initialized with {len(self._available_products)} documents: "
            f"{self._available_products}"
        )

    def _configure_logging(self) -> None:
        """Configure Logfire for observability."""
        if self.valves.LOGFIRE_TOKEN:
            logfire.configure(
                token=self.valves.LOGFIRE_TOKEN,
                send_to_logfire=True
            )
            try:
                logfire.instrument_requests()
            except Exception:
                pass

    def _validate_config(self) -> None:
        """Validate required configuration."""
        # Only check CHROMA_DIR in legacy local mode (no CHROMA_HOST)
        if not CHROMA_HOST and not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB directory not found at {CHROMA_DIR}. "
                "Please run 'python ingest.py' first."
            )

        if not self.valves.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is required. Set it in environment or .env file."
            )

    def _init_vector_store(self) -> None:
        """Initialize vector store and discover documents."""
        self._vector_store = VectorStore()
        self._available_products, self._products = self._vector_store.discover_products()

        if self._vector_store.document_count == 0:
            logfire.warn("ChromaDB collection is empty. Run ingest.py to add documents.")

    def _init_agents(self) -> None:
        """Initialize the triage agent."""
        documents_formatted = self._format_documents()
        self._triage_agent = create_triage_agent(documents_formatted)

    def _format_documents(self) -> str:
        """Format documents for prompts."""
        if not self._available_products:
            return "No documents available"

        lines = []
        for key in self._available_products:
            product = self._products.get(key)
            desc = product.description if product else key
            lines.append(f"- {key}: {desc}")
        return "\n".join(lines)

    def _format_document_list_markdown(self) -> str:
        """Format documents as markdown list for user display."""
        return "\n".join([
            f"- **{key}**: {self._products[key].description}"
            for key in self._available_products
        ])

    def pipe(
        self,
        user_message: str,
        model_id: str = None,
        messages: list = None,
        body: dict = None,
    ) -> str:
        """
        Process a user message through the multi-agent RAG pipeline.

        Flow:
        1. Check if this is a follow-up to a clarification question
        2. Triage Agent: Identify document + detect language
        3. If ambiguous: Return clarification question
        4. Query Agent: Search filtered document + generate response
        """
        with logfire.span("Pipeline.pipe"):
            logfire.info(f"Received query: {user_message[:100]}...")

            try:
                self._ensure_initialized()

                # Step 1: Check clarification context
                effective_query = self._process_clarification(user_message, messages)

                # Step 2: Extract conversation context
                context_hint = self._get_context_hint(messages)

                # Step 3: Triage - Document ID + Language Detection
                triage = self._run_triage(effective_query, context_hint)

                # Step 4: Handle ambiguity
                if triage.confidence == "ambiguous" and triage.clarification_question:
                    return self._build_clarification_response(triage.clarification_question)

                # Step 5: Search context
                product_filter = triage.identified_product if triage.confidence != "low" else None
                if triage.confidence == "low":
                    logfire.info("Low confidence, searching all documents")

                retrieved_context = self._search_context(
                    triage.reformulated_query,
                    product_filter,
                )

                if not retrieved_context:
                    return self._no_results_message(triage.detected_language)

                # Step 6: Generate response
                return self._generate_response(
                    user_message,
                    retrieved_context,
                    triage.detected_language,
                )

            except FileNotFoundError as e:
                error_msg = f"Error de configuracion: {str(e)}"
                logfire.error(error_msg)
                return f"Error: {error_msg}"

            except Exception as e:
                error_msg = f"Error procesando la consulta: {str(e)}"
                logfire.error(error_msg, exc_info=True)
                return f"Error: {error_msg}"

    def _process_clarification(self, user_message: str, messages: list) -> str:
        """Process clarification context and return effective query."""
        clarification_ctx = check_clarification_context(messages)

        if clarification_ctx:
            effective_query = (
                f"{clarification_ctx['original_query']}. "
                f"El documento es: {user_message}"
            )
            logfire.info(f"Merged clarification response: {effective_query}")
            return effective_query

        return user_message

    def _get_context_hint(self, messages: list) -> str:
        """Extract context hint from conversation history."""
        conv_context = extract_conversation_context(messages, self._available_products)

        if conv_context.previously_identified_product:
            logfire.info(f"Using previous document context: {conv_context.previously_identified_product}")
            return f"\nPreviously discussed document: {conv_context.previously_identified_product}"

        return ""

    def _run_triage(self, query: str, context_hint: str):
        """Run the triage agent and return result."""
        with logfire.span("Triage Agent"):
            prompt = build_triage_prompt(
                query=query,
                products_formatted=self._format_documents(),
                context_hint=context_hint,
            )

            result = run_agent_with_fallback(
                self._triage_agent,
                prompt,
                fallback_model=LLM_FALLBACK_MODEL,
            )
            triage = result.output

            logfire.info(
                f"Triage: document={triage.identified_product}, "
                f"confidence={triage.confidence}, "
                f"language={triage.detected_language}"
            )

            return triage

    def _build_clarification_response(self, question: str) -> str:
        """Build clarification response with document list."""
        logfire.info("Document ambiguous, asking for clarification")
        return f"{question}\n\nDocumentos disponibles:\n{self._format_document_list_markdown()}"

    def _search_context(self, query: str, product_filter: Optional[str]) -> str:
        """Search for relevant context in vector store."""
        with logfire.span("Context Retrieval"):
            return self._vector_store.search(
                query=query,
                top_k=self.valves.TOP_K_RESULTS,
                similarity_threshold=self.valves.SIMILARITY_THRESHOLD,
                product_filter=product_filter,
            )

    def _no_results_message(self, language: Language) -> str:
        """Return appropriate no-results message."""
        messages = {
            Language.SPANISH: "No encontré información relevante en la documentación disponible para tu consulta.",
            Language.ENGLISH: "I couldn't find relevant information in the available documentation for your query.",
        }
        return messages.get(language, messages[Language.ENGLISH])

    def _generate_response(
        self,
        user_question: str,
        context: str,
        language: Language,
    ) -> str:
        """Generate response using the query agent."""
        with logfire.span("Query Agent"):
            query_agent = create_query_agent(language)
            prompt = build_query_prompt(user_question, context)

            result = run_agent_with_fallback(
                query_agent,
                prompt,
                fallback_model=LLM_FALLBACK_MODEL,
            )

            logfire.info("Response generated successfully")
            return result.output

    async def on_startup(self):
        """Called when the pipeline is loaded."""
        try:
            self._ensure_initialized()
            logfire.info(f"Pipeline '{self.name}' ready with {len(self._available_products)} documents")
        except Exception as e:
            print(f"Pipeline startup error: {e}")

    async def on_shutdown(self):
        """Called when the pipeline is unloaded."""
        logfire.info(f"Pipeline '{self.name}' shutting down...")
        self._initialized = False
        self._vector_store = None
        self._triage_agent = None
        self._available_products = []
        self._products = {}
