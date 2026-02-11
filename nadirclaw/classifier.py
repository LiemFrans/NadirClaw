"""
Binary complexity classifier using sentence embedding prototypes.

Classifies prompts as simple or complex by comparing their embeddings
to pre-computed centroids of seed examples for each class.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed prototypes
# ---------------------------------------------------------------------------

SIMPLE_PROTOTYPES: List[str] = [
    # Basic factual Q&A
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "How many continents are there?",
    "What is the chemical symbol for gold?",
    "What language do they speak in Brazil?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in our solar system?",
    "What is the speed of light?",
    "How many sides does a hexagon have?",

    # Simple math
    "What is 25 times 4?",
    "Calculate 150 divided by 3",
    "What is 17 plus 38?",
    "What is 10% of 250?",
    "Convert 5 kilometers to miles",

    # Definitions
    "Define photosynthesis",
    "What does GDP stand for?",
    "What is an algorithm?",
    "Define the word 'serendipity'",
    "What is a metaphor?",

    # Translations
    "How do you say hello in Japanese?",
    "Translate 'thank you' to Spanish",
    "What is 'goodbye' in French?",

    # Yes/no questions
    "Is Python a compiled language?",
    "Does the Earth revolve around the Sun?",
    "Is 7 a prime number?",
    "Can penguins fly?",
    "Is HTML a programming language?",

    # Simple formatting / single-step tasks
    "Convert this text to uppercase: hello world",
    "List the days of the week",
    "Sort these numbers: 5, 2, 8, 1, 9",
    "What is today's date?",
    "Reverse the string 'hello'",

    # Greetings and small talk
    "Hello, how are you?",
    "Good morning!",
    "Tell me a joke",
    "What's your name?",

    # Lookup requests
    "What is the boiling point of water?",
    "What timezone is New York in?",
    "What is the population of Tokyo?",
    "How tall is Mount Everest?",
    "What is the atomic number of carbon?",

    # Simple instructions
    "Summarize this in one sentence",
    "Give me a synonym for 'happy'",
    "What is the opposite of 'cold'?",
    "Name three primary colors",
    "What comes after Tuesday?",

    # Agent-style simple tasks
    "Read the file config.yaml",
    "Show me the contents of README.md",
    "What's in the .env file?",
    "List all files in the src directory",
    "Find all Python files in this project",

    # Simple shell commands
    "Run npm install",
    "Check the git status",
    "What branch am I on?",
    "Show me the last 5 git commits",
    "Run the tests",

    # Single web lookups
    "Search for the latest version of React",
    "What is the current price of Bitcoin?",
    "Look up the npm package express",
    "Fetch the docs for FastAPI Query parameters",

    # Simple code questions
    "What does this function do?",
    "What's the syntax for a Python list comprehension?",
    "How do I create a new branch in git?",
    "What port is the server running on?",
    "Show me the imports in main.py",

    # Quick edits
    "Change the port from 3000 to 8080",
    "Add a newline at the end of this file",
    "Rename the variable x to count",
    "Fix this typo: 'recieve' should be 'receive'",
    "Remove the unused import on line 3",

    # Simple status / info
    "What version of Node is installed?",
    "How much disk space is available?",
    "Is the database running?",
    "What's my IP address?",
    "Show the environment variables",
]

COMPLEX_PROTOTYPES: List[str] = [
    # System / architecture design
    "Design a microservices architecture for a real-time multiplayer game with matchmaking, leaderboards, and anti-cheat systems",
    "Architect a distributed event-sourcing system for a financial trading platform that handles millions of transactions per second",
    "Design a scalable notification system that supports push, email, SMS, and in-app channels with user preferences and rate limiting",
    "Create a system design for a URL shortener that handles 100 million daily requests with analytics",

    # Multi-step reasoning
    "Explain the trade-offs between consistency and availability in distributed systems, using real-world examples from companies like Netflix and Amazon",
    "Compare and contrast the CAP theorem implications for SQL vs NoSQL databases in a microservices context, then recommend an approach for an e-commerce platform",
    "Analyze the economic impacts of remote work adoption on urban real estate markets, transportation infrastructure, and local businesses",
    "Walk me through the reasoning behind choosing between a monolithic and microservices architecture for a startup's MVP",

    # Code implementation
    "Implement a thread-safe LRU cache in Python with TTL support, size limits, and cache statistics tracking",
    "Write a complete REST API with authentication, rate limiting, pagination, and comprehensive error handling using FastAPI",
    "Build a React component library with a design system that includes theming, accessibility, responsive design, and documentation",
    "Implement a B-tree data structure with insert, delete, search, and range query operations, including rebalancing",
    "Create a Python decorator that implements retry logic with exponential backoff, jitter, circuit breaking, and logging",

    # Debugging / optimization
    "Debug this memory leak in a Node.js application that occurs during WebSocket connections under high concurrency, and explain the GC implications",
    "Optimize this SQL query that takes 30 seconds on a table with 50 million rows, considering indexing strategies, query plan analysis, and partitioning",
    "Profile and optimize a Python data pipeline that processes 10GB CSV files, currently taking 4 hours to complete",

    # Research synthesis
    "Compare the latest transformer architectures (GPT-4, Claude, Gemini, Llama) in terms of training methodology, context handling, and real-world performance benchmarks",
    "Synthesize the current research on quantum error correction and explain the implications for practical quantum computing timelines",
    "Analyze the trade-offs between different consensus algorithms (Paxos, Raft, PBFT) for blockchain applications",

    # Creative writing
    "Write a short story exploring the ethical implications of AI consciousness, incorporating themes from both Eastern and Western philosophy",
    "Create a detailed fictional world with its own political system, economy, religion, and history spanning 1000 years",
    "Write a technical blog post explaining category theory concepts using cooking metaphors, targeted at software engineers",

    # Mathematical proofs and analysis
    "Prove that the halting problem is undecidable using a diagonalization argument, and explain the practical implications for software verification",
    "Derive the backpropagation algorithm from first principles and explain how automatic differentiation relates to it",
    "Prove the correctness of Dijkstra's algorithm and analyze its time complexity under different priority queue implementations",

    # Multi-part constrained questions
    "Design a database schema for a social media platform, then write migration scripts, explain indexing strategy, outline the caching layer, and describe the data archival process",
    "Create a complete CI/CD pipeline configuration including build, test, security scanning, staging deployment, canary releases, and rollback procedures",
    "Build a comprehensive monitoring and alerting system covering application metrics, infrastructure health, business KPIs, and incident response automation",

    # Security auditing
    "Perform a security audit of this authentication flow, identifying potential vulnerabilities including OWASP Top 10 risks, and recommend mitigations with code examples",
    "Design a zero-trust security architecture for a multi-cloud environment with on-premises legacy systems",
    "Analyze the security implications of implementing OAuth 2.0 with PKCE for a mobile application, including threat modeling",

    # Strategic planning
    "Develop a technical migration strategy to move a monolithic PHP application to a cloud-native architecture, including timeline, risk assessment, and team structure",
    "Create a comprehensive data strategy for a healthcare company transitioning to AI-driven diagnostics, addressing privacy, compliance, and technical infrastructure",
    "Design a disaster recovery plan for a financial services company with RPO of 15 minutes and RTO of 1 hour across multiple regions",

    # Advanced analysis
    "Analyze the performance characteristics of different garbage collection algorithms and recommend the optimal GC configuration for a latency-sensitive trading system",
    "Compare container orchestration approaches (Kubernetes, Nomad, ECS) for a company running 500 microservices across three cloud providers",
    "Evaluate the trade-offs of different state management solutions in a large-scale React application with offline-first requirements",

    # Complex code review / refactoring
    "Refactor this 2000-line God class into a proper domain-driven design with bounded contexts, aggregates, and event handlers",
    "Review this distributed transaction implementation for correctness, identify race conditions, and propose a saga pattern alternative",
    "Analyze this machine learning pipeline for data leakage, feature engineering issues, and suggest improvements to the model evaluation strategy",

    # Cross-domain synthesis
    "Explain how principles from control theory apply to designing auto-scaling systems, with specific implementation recommendations",
    "Connect concepts from game theory to microservices communication patterns and explain how to prevent cascade failures",
    "Apply information theory concepts to optimize data compression in a real-time video streaming architecture",

    # Complex debugging scenarios
    "Diagnose why this Kubernetes cluster experiences intermittent pod evictions during peak traffic, analyzing resource limits, node pressure, and scheduler behavior",
    "Investigate a race condition in this concurrent Go program that only manifests under specific timing conditions with high CPU load",
    "Troubleshoot why this distributed cache shows inconsistent reads after network partitions despite using quorum-based replication",

    # Agent-style complex tasks
    "Refactor the authentication system from JWT to session-based auth, updating all middleware, routes, and tests across the codebase",
    "Migrate this Express.js app to use TypeScript, converting all files, adding type definitions, and updating the build pipeline",
    "Split this monolithic app.py into a proper package structure with separate modules for routes, models, services, and utils",

    # Multi-step debugging with tool use
    "The deploy is failing in CI — check the GitHub Actions logs, find the error, trace it back to the source file, and fix it",
    "Users are reporting 500 errors on the /api/checkout endpoint — check the logs, reproduce the issue, identify the root cause, and implement a fix",
    "The app is consuming 4GB of memory in production — profile the application, identify the memory leak, and fix it",

    # Complex code generation with context
    "Read the existing database schema, then implement a new feature for user notifications with migrations, models, API endpoints, and WebSocket support",
    "Look at how authentication works in this codebase, then add OAuth2 login with Google and GitHub providers, including callback handlers and token refresh",
    "Analyze the current test suite, identify gaps in coverage, and write integration tests for all untested API endpoints",

    # Multi-tool orchestration
    "Set up a complete development environment: clone the repo, install dependencies, configure the database, run migrations, seed test data, and verify everything works",
    "Create a new microservice that integrates with the existing API gateway — scaffold the project, implement the service, add Docker config, update the gateway routes, and write deployment manifests",
    "Audit all dependencies for security vulnerabilities, update the vulnerable ones, verify no breaking changes, and update the lockfile",

    # Architecture decisions requiring analysis
    "Analyze the current database query patterns, identify the N+1 queries, and refactor the ORM layer to use eager loading with proper join strategies",
    "Review the error handling across the entire API, standardize the error response format, add proper error codes, and implement a global exception handler",
    "Evaluate whether we should migrate from REST to GraphQL for this project, analyze the current endpoints, and create a migration plan with schema design",

    # Complex browser / web tasks
    "Scrape the documentation site, extract all API endpoints and their parameters, generate TypeScript types, and create an API client library",
    "Monitor this web application for performance regressions — set up Lighthouse CI, configure budgets, and create alerts for Core Web Vitals degradation",
]


class BinaryComplexityClassifier:
    """
    Classifies prompts as simple or complex using semantic prototype centroids.

    At init time, embeds all seed prototypes and computes a normalised centroid
    for each class.  At inference time, embeds the prompt (~10 ms on warm
    encoder), computes cosine similarity to both centroids, and returns a
    binary decision with a confidence score.
    """

    def __init__(
        self,
        allowed_providers: Optional[List[str]] = None,
        allowed_models: Optional[List[str]] = None,
    ):
        self.allowed_providers = allowed_providers or []
        self.allowed_models = allowed_models or []

        # Load model performance data for ranking
        self.performance_data = self._load_performance_data()

        # Load encoder and compute centroids
        from nadirclaw.encoder import get_shared_encoder_sync

        self.encoder = get_shared_encoder_sync()
        self._simple_centroid, self._complex_centroid = self._compute_centroids()

        logger.info(
            "BinaryComplexityClassifier ready — %d simple / %d complex prototypes",
            len(SIMPLE_PROTOTYPES),
            len(COMPLEX_PROTOTYPES),
        )

    # ------------------------------------------------------------------
    # Startup: pre-compute centroids
    # ------------------------------------------------------------------

    def _compute_centroids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Embed all prototypes and return L2-normalised centroids."""
        simple_embs = self.encoder.encode(SIMPLE_PROTOTYPES, show_progress_bar=False)
        complex_embs = self.encoder.encode(COMPLEX_PROTOTYPES, show_progress_bar=False)

        simple_centroid = simple_embs.mean(axis=0)
        complex_centroid = complex_embs.mean(axis=0)

        # Normalise so dot product == cosine similarity
        simple_centroid = simple_centroid / np.linalg.norm(simple_centroid)
        complex_centroid = complex_centroid / np.linalg.norm(complex_centroid)

        return simple_centroid, complex_centroid

    # ------------------------------------------------------------------
    # Core classification
    # ------------------------------------------------------------------

    def classify(self, prompt: str) -> Tuple[bool, float]:
        """
        Classify a prompt as simple or complex.

        Borderline cases (confidence < threshold) are biased toward complex —
        it is cheaper to over-serve a simple prompt than to under-serve a
        complex one.

        Returns:
            (is_complex, confidence) where confidence is in [0, 1].
            confidence near 0 means borderline; near 1 means very clear.
        """
        from nadirclaw.settings import settings

        threshold = settings.CONFIDENCE_THRESHOLD

        emb = self.encoder.encode([prompt], show_progress_bar=False)[0]
        emb = emb / np.linalg.norm(emb)

        sim_simple = float(np.dot(emb, self._simple_centroid))
        sim_complex = float(np.dot(emb, self._complex_centroid))

        confidence = abs(sim_complex - sim_simple)

        if confidence < threshold:
            # Borderline -> default to complex (safe bias)
            is_complex = True
        else:
            is_complex = sim_complex > sim_simple

        return is_complex, confidence

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyze(self, text: str, **kwargs) -> Dict[str, Any]:
        """Async analyse — conforms to the analyzer interface."""
        return self._analyze_sync(text)

    def _analyze_sync(self, text: str) -> Dict[str, Any]:
        start = time.time()
        is_complex, confidence = self.classify(text)

        recommended_model, recommended_provider = self._select_model(is_complex, confidence)
        ranked_models = self._build_ranked_models(is_complex, confidence)

        complexity_score = self._confidence_to_score(is_complex, confidence)
        tier = 3 if is_complex else 1
        tier_name = "complex" if is_complex else "simple"

        latency_ms = int((time.time() - start) * 1000)

        return {
            "recommended_model": recommended_model,
            "recommended_provider": recommended_provider,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "complexity_tier": tier,
            "complexity_name": tier_name,
            "tier": tier,
            "tier_name": tier_name,
            "reasoning": (
                f"Binary classifier: {'complex' if is_complex else 'simple'} "
                f"(confidence={confidence:.3f})"
            ),
            "ranked_models": ranked_models,
            "analyzer_latency_ms": latency_ms,
            "analyzer_type": "binary",
            "selection_method": "binary_classifier",
            "model_type": "binary_classifier",
        }

    # ------------------------------------------------------------------
    # Model selection helpers
    # ------------------------------------------------------------------

    def _select_model(self, is_complex: bool, confidence: float) -> Tuple[str, str]:
        """Pick the best model based on classification.

        If explicit tier models are configured (NADIRCLAW_SIMPLE_MODEL /
        NADIRCLAW_COMPLEX_MODEL), use those directly. Otherwise fall back to
        ranking candidates from models.json.
        """
        from nadirclaw.settings import settings

        # Explicit tier mapping takes priority
        model = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL
        provider = model.split("/")[0] if "/" in model else "api"
        # LiteLLM auto-detects provider from model name, so "api" is fine
        return model, provider

    def _build_ranked_models(self, is_complex: bool, confidence: float) -> List[Dict[str, Any]]:
        """Build a ranked model list."""
        candidates = self._get_candidate_models()
        if not candidates:
            return []

        if is_complex:
            candidates.sort(key=lambda m: m["quality_index"], reverse=True)
        else:
            candidates.sort(key=lambda m: m["cost"])

        ranked = []
        for c in candidates[:10]:
            ranked.append({
                "model_name": c["api_id"],
                "provider": c["provider"],
                "confidence": confidence,
                "reasoning": f"Binary classifier: {'complex->quality' if is_complex else 'simple->cost'} priority",
                "cost_per_million_tokens": c["cost"],
                "quality_index": c["quality_index"],
                "api_id": c["api_id"],
                "performance_name": c["model_name"],
                "suitability_score": c["quality_index"] if is_complex else max(0, 100 - c["cost"] * 10),
            })
        return ranked

    def _get_candidate_models(self) -> List[Dict[str, Any]]:
        """Return a flat list of candidate models filtered by allowed_providers/models."""
        if not self.performance_data:
            return []

        candidates = []
        for model in self.performance_data:
            api_id = model.get("api_id", "")
            model_name = model.get("model", "")
            provider = model.get("api_provider", "").lower()
            route = (model.get("route", "") or "").lower()

            # Provider filter
            if self.allowed_providers:
                allowed_lower = [p.lower() for p in self.allowed_providers]
                if provider not in allowed_lower and route not in allowed_lower:
                    continue

            # Model filter
            if self.allowed_models:
                if not any(a in (model_name, api_id) for a in self.allowed_models):
                    continue

            quality_index = float(model.get("quality_index", 50))
            cost = float(model.get("blended_cost_usd1m", 1.0))

            candidates.append({
                "api_id": api_id,
                "model_name": model_name,
                "provider": route or provider,
                "quality_index": quality_index,
                "cost": cost,
            })

        # Add any allowed_models that weren't found in performance data
        # (e.g. custom Ollama models). Treat them as cheap/low-quality so the
        # router sends simple prompts their way by default.
        known_ids = {c["api_id"] for c in candidates}
        for m in (self.allowed_models or []):
            if m not in known_ids:
                provider = m.split("/")[0] if "/" in m else "unknown"
                candidates.append({
                    "api_id": m,
                    "model_name": m,
                    "provider": provider,
                    "quality_index": 30.0,   # assume modest quality
                    "cost": 0.0,             # assume free (local)
                })

        return candidates

    @staticmethod
    def _confidence_to_score(is_complex: bool, confidence: float) -> float:
        """Map binary decision + confidence to a 0-1 complexity score."""
        if is_complex:
            return 0.5 + min(confidence * 5, 0.5)  # 0.5 - 1.0
        else:
            return 0.5 - min(confidence * 5, 0.5)  # 0.0 - 0.5

    def _load_performance_data(self) -> List[Dict]:
        """Load models.json from the package directory."""
        try:
            path = os.path.join(os.path.dirname(__file__), "models.json")
            with open(path) as f:
                data = json.load(f)
            return data.get("models", [])
        except Exception as e:
            logger.error("Error loading models.json: %s", e)
            return []


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------
_singleton: Optional[BinaryComplexityClassifier] = None


def get_binary_classifier(
    allowed_providers: Optional[List[str]] = None,
    allowed_models: Optional[List[str]] = None,
) -> BinaryComplexityClassifier:
    """Return a new classifier instance with the given filters.

    The heavy part (encoder + centroids) is shared via get_shared_encoder_sync().
    """
    return BinaryComplexityClassifier(
        allowed_providers=allowed_providers,
        allowed_models=allowed_models,
    )


def warmup() -> None:
    """Pre-warm the encoder and compute centroids once at startup."""
    global _singleton
    logger.info("Warming up BinaryComplexityClassifier ...")
    _singleton = BinaryComplexityClassifier()
    logger.info("BinaryComplexityClassifier warmup complete")
