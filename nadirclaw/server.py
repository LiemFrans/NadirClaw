"""
NadirClaw — Lightweight LLM router server.

Routes simple prompts to cheap/local models and complex prompts to premium models.
OpenAI-compatible API at /v1/chat/completions.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from nadirclaw.auth import UserSession, validate_local_auth
from nadirclaw.settings import settings

logger = logging.getLogger("nadirclaw")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NadirClaw",
    version="0.2.0",
    description="Open-source LLM router — simple prompts to free models, complex to premium",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = False


class ClassifyRequest(BaseModel):
    prompt: str
    system_message: Optional[str] = ""


class ClassifyBatchRequest(BaseModel):
    prompts: List[str]


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log_request(entry: Dict[str, Any]) -> None:
    """Append a JSON line to the request log and print to console."""
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(request_log, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    tier = entry.get("tier", "?")
    model = entry.get("selected_model", "?")
    conf = entry.get("confidence", 0)
    score = entry.get("complexity_score", 0)
    prompt_preview = entry.get("prompt", "")[:80]
    latency = entry.get("classifier_latency_ms", "?")
    total = entry.get("total_latency_ms", "?")
    logger.info(
        "%-8s model=%-35s conf=%.3f score=%.2f lat=%sms total=%sms  \"%s\"",
        tier, model, conf, score, latency, total, prompt_preview,
    )


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    request_log = log_dir / "requests.jsonl"

    logger.info("=" * 60)
    logger.info("NadirClaw starting...")
    logger.info("Log file: %s", request_log.resolve())
    logger.info("=" * 60)

    # Warm up the binary classifier
    try:
        from nadirclaw.classifier import warmup
        logger.info("Warming up binary classifier...")
        warmup()
        logger.info("Binary classifier ready")
    except Exception as e:
        logger.error("Failed to warm binary classifier: %s", e)
        raise

    # Show config
    try:
        import litellm
        litellm.set_verbose = False
        logger.info("Simple model:  %s", settings.SIMPLE_MODEL)
        logger.info("Complex model: %s", settings.COMPLEX_MODEL)
        if settings.has_explicit_tiers:
            logger.info("Tier config:   explicit (env vars)")
        else:
            logger.info("Tier config:   derived from NADIRCLAW_MODELS")
        logger.info("Ollama base:   %s", settings.OLLAMA_API_BASE)
        token = settings.AUTH_TOKEN
        if token:
            logger.info("Auth:          %s***", token[:6] if len(token) >= 6 else token)
        else:
            logger.info("Auth:          disabled (local-only)")
        # Log credential status
        from nadirclaw.credentials import detect_provider, get_credential_source

        for model in settings.tier_models:
            provider = detect_provider(model)
            if provider and provider != "ollama":
                source = get_credential_source(provider)
                if source:
                    logger.info("Credential:    %s → %s", provider, source)
                else:
                    logger.warning("Credential:    %s → NOT CONFIGURED", provider)

    except Exception as e:
        logger.warning("LiteLLM setup issue: %s", e)

    logger.info("Ready! Listening for requests...")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Smart routing internals
# ---------------------------------------------------------------------------

async def _smart_route_analysis(
    prompt: str, system_message: str, user: UserSession
) -> tuple:
    """Run classifier, return (selected_model, analysis_dict). No LLM call."""
    from nadirclaw.classifier import get_binary_classifier

    analyzer = get_binary_classifier()
    result = await analyzer.analyze(text=prompt, system_message=system_message)

    is_complex = result.get("tier_name") == "complex"
    selected = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL

    analysis = {
        "strategy": "smart-routing",
        "analyzer": result.get("analyzer_type", "binary"),
        "selected_model": selected,
        "complexity_score": result.get("complexity_score"),
        "tier": result.get("tier_name"),
        "confidence": result.get("confidence"),
        "reasoning": result.get("reasoning"),
        "classifier_latency_ms": result.get("analyzer_latency_ms"),
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
        "ranked_models": [
            {"model": m.get("model_name"), "score": m.get("suitability_score")}
            for m in result.get("ranked_models", [])[:5]
        ],
    }

    return selected, analysis


async def _smart_route_full(
    messages: List[ChatMessage], user: UserSession
) -> tuple:
    """Smart route for full completions."""
    user_msgs = [m.content for m in messages if m.role == "user"]
    prompt = user_msgs[-1] if user_msgs else ""
    system_msg = next((m.content for m in messages if m.role == "system"), "")
    return await _smart_route_analysis(prompt, system_msg, user)


# ---------------------------------------------------------------------------
# /v1/classify — dry-run classification (no LLM call)
# ---------------------------------------------------------------------------

@app.post("/v1/classify")
async def classify_prompt(
    request: ClassifyRequest,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Classify a prompt without calling any LLM."""
    _, analysis = await _smart_route_analysis(
        request.prompt, request.system_message or "", current_user
    )

    _log_request({
        "type": "classify",
        "prompt": request.prompt,
        **analysis,
    })

    return {
        "prompt": request.prompt,
        "classification": analysis,
    }


@app.post("/v1/classify/batch")
async def classify_batch(
    request: ClassifyBatchRequest,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """Classify multiple prompts at once."""
    results = []
    for prompt in request.prompts:
        _, analysis = await _smart_route_analysis(prompt, "", current_user)
        results.append({
            "prompt": prompt,
            "selected_model": analysis.get("selected_model"),
            "tier": analysis.get("tier"),
            "confidence": analysis.get("confidence"),
            "complexity_score": analysis.get("complexity_score"),
        })
        _log_request({"type": "classify_batch", "prompt": prompt, **analysis})

    simple_count = sum(1 for r in results if r["tier"] == "simple")
    complex_count = sum(1 for r in results if r["tier"] == "complex")

    return {
        "total": len(results),
        "simple": simple_count,
        "complex": complex_count,
        "results": results,
    }


# ---------------------------------------------------------------------------
# /v1/chat/completions — full completion with routing
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    start_time = time.time()
    request_id = str(uuid.uuid4())

    try:
        # Extract prompt for logging
        user_msgs = [m.content for m in request.messages if m.role == "user"]
        prompt_text = user_msgs[-1] if user_msgs else ""

        # Route
        if request.model and request.model != "auto":
            selected_model = request.model
            analysis_info = {
                "strategy": "direct",
                "selected_model": selected_model,
                "tier": "direct",
                "confidence": 1.0,
                "complexity_score": 0,
            }
        else:
            selected_model, analysis_info = await _smart_route_full(
                request.messages, current_user
            )

        # Call the model via LiteLLM
        import litellm

        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        call_kwargs = {"model": selected_model, "messages": messages}
        if request.temperature is not None:
            call_kwargs["temperature"] = request.temperature
        if request.max_tokens is not None:
            call_kwargs["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            call_kwargs["top_p"] = request.top_p

        # Resolve provider credential
        from nadirclaw.credentials import detect_provider, get_credential

        provider = detect_provider(selected_model)
        if provider and provider != "ollama":
            api_key = get_credential(provider)
            if api_key:
                call_kwargs["api_key"] = api_key

        logger.debug("Calling LiteLLM: model=%s", selected_model)
        response = await litellm.acompletion(**call_kwargs)

        elapsed_ms = int((time.time() - start_time) * 1000)

        _log_request({
            "type": "completion",
            "request_id": request_id,
            "prompt": prompt_text,
            "selected_model": selected_model,
            "tier": analysis_info.get("tier"),
            "confidence": analysis_info.get("confidence"),
            "complexity_score": analysis_info.get("complexity_score"),
            "classifier_latency_ms": analysis_info.get("classifier_latency_ms"),
            "total_latency_ms": elapsed_ms,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "response_preview": (response.choices[0].message.content or "")[:100],
        })

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.choices[0].message.content,
                    },
                    "finish_reason": response.choices[0].finish_reason or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            "nadirclaw_metadata": {
                "request_id": request_id,
                "response_time_ms": elapsed_ms,
                "routing": analysis_info,
            },
        }

    except Exception as e:
        logger.error("Completion error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# /v1/logs — view request logs
# ---------------------------------------------------------------------------

@app.get("/v1/logs")
async def view_logs(
    limit: int = 20,
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    """View recent request logs."""
    request_log = settings.LOG_DIR / "requests.jsonl"
    if not request_log.exists():
        return {"logs": [], "total": 0}

    lines = request_log.read_text().strip().split("\n")
    recent = lines[-limit:] if len(lines) > limit else lines
    logs = []
    for line in reversed(recent):
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    return {"logs": logs, "total": len(lines), "showing": len(logs)}


# ---------------------------------------------------------------------------
# /v1/models & /health
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models(
    current_user: UserSession = Depends(validate_local_auth),
) -> Dict[str, Any]:
    models = settings.tier_models
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": int(time.time()),
                "owned_by": m.split("/")[0] if "/" in m else "api",
            }
            for m in models
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "0.2.0",
        "simple_model": settings.SIMPLE_MODEL,
        "complex_model": settings.COMPLEX_MODEL,
    }


@app.get("/")
async def root():
    return {
        "name": "NadirClaw",
        "version": "0.2.0",
        "description": "Open-source LLM router",
        "status": "ok",
    }
