"""Guards on the packaging, not the code.

A container that ships without the corpus comes up healthy, serves the SPA, and
abstains on every question - it looks like a retrieval quality problem rather
than an empty index, which is why it survived until someone deployed it. These
assertions are cheap and would have caught it before the image was built.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
RENDER_YAML = ROOT / "render.yaml"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_image_ships_the_corpus(dockerfile: str):
    """Without this the deployed index is empty and every answer is a refusal."""
    assert "data/corpus/" in dockerfile, (
        "Dockerfile must COPY data/corpus/ - a container cannot answer anything without it"
    )


def test_corpus_dir_points_at_the_baked_copy_not_the_volume(dockerfile: str):
    """``VOLUME ["/data"]`` means a mounted volume shadows anything baked there.

    The sample corpus is immutable image content and belongs under /app; the
    volume is for uploads and the index.
    """
    assert "RAG_CORPUS_DIR=/app/data/corpus" in dockerfile
    assert "RAG_CORPUS_DIR=/data/corpus" not in dockerfile


def test_corpus_is_not_excluded_from_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    offending = [
        line
        for line in ignored
        if line.strip() and not line.startswith("#") and "corpus" in line
    ]
    assert not offending, f".dockerignore would exclude the corpus: {offending}"


def test_the_corpus_actually_has_documents():
    files = list((ROOT / "data" / "corpus").iterdir())
    assert files, "data/corpus is empty, so any image built from it is useless"


@pytest.mark.skipif(not RENDER_YAML.exists(), reason="no render blueprint")
def test_render_blueprint_keeps_the_public_demo_keyless():
    """A public URL carrying an API key is a quota anyone can spend."""
    blueprint = yaml.safe_load(RENDER_YAML.read_text(encoding="utf-8"))
    service = blueprint["services"][0]
    env = {item["key"]: item for item in service["envVars"]}

    assert env["RAG_LLM_API_KEY"].get("value") == "", "public demo must not ship a key"
    assert env["RAG_AUTH_ALLOW_REGISTRATION"]["value"] == "false"
    # A literal secret in the blueprint would be committed to the repo.
    assert "value" not in env["RAG_AUTH_SECRET"]
    assert env["RAG_AUTH_SECRET"]["generateValue"] is True
    assert service["healthCheckPath"] == "/live"
