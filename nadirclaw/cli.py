"""NadirClaw CLI — serve, classify, onboard, and status commands."""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.2.0", prog_name="nadirclaw")
def main():
    """NadirClaw — Open-source LLM router."""
    pass


@main.command()
@click.option("--port", default=None, type=int, help="Port to listen on (default: 8000)")
@click.option("--simple-model", default=None, help="Model for simple prompts")
@click.option("--complex-model", default=None, help="Model for complex prompts")
@click.option("--models", default=None, help="Comma-separated model list (legacy)")
@click.option("--token", default=None, help="Auth token")
@click.option("--verbose", is_flag=True, help="Enable verbose logging")
def serve(port, simple_model, complex_model, models, token, verbose):
    """Start the NadirClaw router server."""
    import logging

    from dotenv import load_dotenv

    load_dotenv()

    # Override env vars from CLI flags
    if port:
        os.environ["NADIRCLAW_PORT"] = str(port)
    if simple_model:
        os.environ["NADIRCLAW_SIMPLE_MODEL"] = simple_model
    if complex_model:
        os.environ["NADIRCLAW_COMPLEX_MODEL"] = complex_model
    if models:
        os.environ["NADIRCLAW_MODELS"] = models
    if token:
        os.environ["NADIRCLAW_AUTH_TOKEN"] = token

    log_level = "debug" if verbose else "info"
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import uvicorn

    from nadirclaw.settings import settings

    actual_port = port or settings.PORT
    click.echo(f"Starting NadirClaw on port {actual_port}...")
    click.echo(f"  Simple model:  {settings.SIMPLE_MODEL}")
    click.echo(f"  Complex model: {settings.COMPLEX_MODEL}")
    uvicorn.run(
        "nadirclaw.server:app",
        host="0.0.0.0",
        port=actual_port,
        log_level=log_level,
    )


@main.command()
@click.argument("prompt")
def classify(prompt):
    """Classify a prompt as simple or complex (no server needed)."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    from nadirclaw.classifier import BinaryComplexityClassifier
    from nadirclaw.settings import settings

    classifier = BinaryComplexityClassifier()
    is_complex, confidence = classifier.classify(prompt)

    tier = "complex" if is_complex else "simple"
    score = classifier._confidence_to_score(is_complex, confidence)

    # Pick model from explicit tier config
    model = settings.COMPLEX_MODEL if is_complex else settings.SIMPLE_MODEL

    click.echo(f"Tier:       {tier}")
    click.echo(f"Confidence: {confidence:.4f}")
    click.echo(f"Score:      {score:.4f}")
    click.echo(f"Model:      {model}")


@main.command()
def status():
    """Check if NadirClaw server is running and show config."""
    import urllib.request

    from nadirclaw.settings import settings

    click.echo("NadirClaw Status")
    click.echo("-" * 40)
    click.echo(f"Simple model:  {settings.SIMPLE_MODEL}")
    click.echo(f"Complex model: {settings.COMPLEX_MODEL}")
    if settings.has_explicit_tiers:
        click.echo("Tier config:   explicit (env vars)")
    else:
        click.echo("Tier config:   derived from NADIRCLAW_MODELS")
    click.echo(f"Port:          {settings.PORT}")
    click.echo(f"Threshold:     {settings.CONFIDENCE_THRESHOLD}")
    click.echo(f"Log dir:       {settings.LOG_DIR}")
    token = settings.AUTH_TOKEN
    click.echo(f"Token:         {token[:6]}***" if len(token) >= 6 else f"Token:         {token}")

    # Check if server is running
    try:
        url = f"http://localhost:{settings.PORT}/health"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            click.echo(f"\nServer:        RUNNING ({data.get('status', '?')})")
    except Exception:
        click.echo("\nServer:        NOT RUNNING")


@main.command(name="build-centroids")
def build_centroids():
    """Regenerate centroid .npy files from prototype prompts."""
    import logging

    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from nadirclaw.encoder import get_shared_encoder_sync
    from nadirclaw.prototypes import COMPLEX_PROTOTYPES, SIMPLE_PROTOTYPES

    click.echo("Loading encoder...")
    encoder = get_shared_encoder_sync()

    click.echo(f"Encoding {len(SIMPLE_PROTOTYPES)} simple prototypes...")
    simple_embs = encoder.encode(SIMPLE_PROTOTYPES, show_progress_bar=False)
    simple_centroid = simple_embs.mean(axis=0)
    simple_centroid = simple_centroid / np.linalg.norm(simple_centroid)

    click.echo(f"Encoding {len(COMPLEX_PROTOTYPES)} complex prototypes...")
    complex_embs = encoder.encode(COMPLEX_PROTOTYPES, show_progress_bar=False)
    complex_centroid = complex_embs.mean(axis=0)
    complex_centroid = complex_centroid / np.linalg.norm(complex_centroid)

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    simple_path = os.path.join(pkg_dir, "simple_centroid.npy")
    complex_path = os.path.join(pkg_dir, "complex_centroid.npy")

    np.save(simple_path, simple_centroid.astype(np.float32))
    np.save(complex_path, complex_centroid.astype(np.float32))

    click.echo(f"\nSaved: {simple_path}")
    click.echo(f"Saved: {complex_path}")
    click.echo(f"Centroid dimension: {simple_centroid.shape[0]}")


@main.group()
def openclaw():
    """OpenClaw integration commands."""
    pass


@openclaw.command()
def onboard():
    """Auto-configure OpenClaw to use NadirClaw as a provider."""
    from nadirclaw.settings import settings

    openclaw_dir = Path.home() / ".openclaw"
    config_path = openclaw_dir / "openclaw.json"

    # Read existing config or start fresh
    existing = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = json.load(f)
            # Create backup
            backup_path = config_path.with_suffix(
                f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
            )
            shutil.copy2(config_path, backup_path)
            click.echo(f"Backed up existing config to {backup_path}")
        except Exception as e:
            click.echo(f"Warning: could not read existing config: {e}")

    # Build the NadirClaw provider config
    nadirclaw_provider = {
        "baseUrl": f"http://localhost:{settings.PORT}/v1",
        "apiKey": "${NADIRCLAW_AUTH_TOKEN}",
        "api": "openai-completions",
        "models": [
            {
                "id": "auto",
                "reasoning": True,
                "input": ["text"],
                "contextWindow": 200000,
                "maxTokens": 64000,
            }
        ],
    }

    # Merge into existing config
    if "models" not in existing:
        existing["models"] = {}
    if "mode" not in existing["models"]:
        existing["models"]["mode"] = "merge"
    if "providers" not in existing["models"]:
        existing["models"]["providers"] = {}

    existing["models"]["providers"]["nadirclaw"] = nadirclaw_provider

    # Set default agent model
    if "agents" not in existing:
        existing["agents"] = {}
    if "defaults" not in existing["agents"]:
        existing["agents"]["defaults"] = {}
    if "model" not in existing["agents"]["defaults"]:
        existing["agents"]["defaults"]["model"] = {}

    existing["agents"]["defaults"]["model"]["primary"] = "nadirclaw/auto"

    # Write config
    openclaw_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(existing, f, indent=2)

    click.echo(f"\nWrote OpenClaw config to {config_path}")
    click.echo("\nNadirClaw provider added with model 'nadirclaw/auto'")
    click.echo("Default agent model set to 'nadirclaw/auto'")
    click.echo("\nNext steps:")
    click.echo("  1. Start NadirClaw:  nadirclaw serve")
    click.echo("  2. Verify:           openclaw doctor")


@main.group()
def codex():
    """OpenAI Codex integration commands."""
    pass


@codex.command()
def onboard():
    """Auto-configure Codex to use NadirClaw as a provider."""
    from nadirclaw.settings import settings

    codex_dir = Path.home() / ".codex"
    config_path = codex_dir / "config.toml"

    # Backup existing config if present
    if config_path.exists():
        backup_path = config_path.with_suffix(
            f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.toml"
        )
        shutil.copy2(config_path, backup_path)
        click.echo(f"Backed up existing config to {backup_path}")

    config_content = f"""\
model_provider = "nadirclaw"

[model_providers.nadirclaw]
base_url = "http://localhost:{settings.PORT}/v1"
env_key = "NADIRCLAW_AUTH_TOKEN"
"""

    codex_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config_content)

    click.echo(f"\nWrote Codex config to {config_path}")
    click.echo("\nNadirClaw configured as Codex model provider.")
    click.echo(f"  Base URL: http://localhost:{settings.PORT}/v1")
    click.echo("  Auth:     $NADIRCLAW_AUTH_TOKEN")
    click.echo("\nNext steps:")
    click.echo("  1. Start NadirClaw:  nadirclaw serve")
    click.echo("  2. Run Codex:        codex")


if __name__ == "__main__":
    main()
