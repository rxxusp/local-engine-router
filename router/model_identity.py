"""Canonical model identity for local model ids.

Local model ids are wildly inconsistent: the same underlying model appears as
``qwen2.5-7b-instruct-q4_k_m.gguf`` (llama.cpp), ``Qwen/Qwen2.5-7B-Instruct-AWQ``
(vLLM), ``qwen2.5:7b`` (Ollama), ``mlx-community/Qwen2.5-7B-Instruct-4bit``
(MLX), or a community fine-tune like ``huihui_ai/qwen2.5-abliterated:7b``.
Benchmark intelligence and calibration must key on *what the model is*, not on
the local spelling, so this module parses an id into a :class:`ModelIdentity`:

  * ``family``      — the base model family slug (``qwen2.5``, ``llama3.1``, …)
  * ``size_b``      — parameter count in billions, when the id carries one
  * ``variant``     — ``instruct`` / ``base`` / ``code`` / ``vision`` / ``""``
  * ``quant``       — quantization token (``q4_k_m``, ``awq``, ``fp8``, …)
  * ``fmt``         — packaging format (``gguf``, ``awq``, ``exl2``, ``mlx``, …)
  * ``fine_tune``   — community fine-tune name when recognized (``hermes``, …)
  * ``abliterated`` — abliterated/uncensored variant flag
  * ``canonical``   — stable slug (``family-size-variant[+fine_tune]``) used as
                      the benchmark-cache key
  * ``base_canonical`` — the canonical id of the *base* model a fine-tune or
                      abliterated variant derives from (inheritance target)
  * ``confidence``  — how sure the parse is (1.0 family+size, down to 0.35 for
                      an unrecognized name)

Everything here is pure and deterministic: no I/O, no network, no clock.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Token tables
# --------------------------------------------------------------------------- #

# GGUF-style quant tokens: q4_k_m, q8_0, q3_k_l, iq2_xxs, q5_1, ...
_GGUF_QUANT_RE = re.compile(r"^i?q\d(?:_[a-z0-9]+)*$")

# Non-GGUF quant / precision tokens.
_QUANT_TOKENS = frozenset(
    {
        "fp16", "f16", "bf16", "fp8", "f32", "fp32", "fp4",
        "int4", "int8", "4bit", "8bit", "2bit", "3bit", "6bit",
        "awq", "gptq", "exl2", "exl3", "nf4", "bnb", "marlin",
        "w4a16", "w8a8", "w8a16", "mxfp4",
    }
)

# Packaging-format tokens (may appear as a suffix: "...-GGUF", "...-MLX-4bit").
_FORMAT_TOKENS = {
    "gguf": "gguf",
    "awq": "awq",
    "gptq": "gptq",
    "exl2": "exl2",
    "exl3": "exl3",
    "mlx": "mlx",
    "safetensors": "safetensors",
    "onnx": "onnx",
}

# Chat/instruct-style variant tokens.
_INSTRUCT_TOKENS = frozenset({"instruct", "it", "chat", "chatml"})
_BASE_TOKENS = frozenset({"base", "text", "pt"})
_CODE_TOKENS = frozenset({"coder", "code", "codestral", "starcoder", "codellama"})
_VISION_TOKENS = frozenset({"vl", "vision", "llava", "multimodal", "v"})

# Tokens that mark an abliterated / uncensored community variant.
_ABLITERATED_TOKENS = frozenset({"abliterated", "abliterate", "uncensored"})

# Known community fine-tune names. Matching any of these marks the id as a
# fine-tune of its detected base family (benchmark scores are then inherited
# from the base with reduced confidence rather than matched exactly).
_FINE_TUNE_TOKENS = frozenset(
    {
        "hermes", "openhermes", "dolphin", "wizardlm", "wizard", "vicuna",
        "alpaca", "orca", "airoboros", "magnum", "smaug", "tulu", "zephyr",
        "starling", "openchat", "capybara", "bagel", "mythomax", "goliath",
        "beagle", "neuralchat", "nemotron", "openorca", "samantha", "silicon",
        "einstein", "calme", "athene",
    }
)

# Hugging-Face orgs that publish repacks/fine-tunes rather than base models.
_COMMUNITY_ORGS = frozenset(
    {
        "thebloke", "bartowski", "mlx-community", "unsloth", "lmstudio-community",
        "nousresearch", "huihui-ai", "huihui_ai", "cognitivecomputations",
        "mradermacher", "qwp", "teknium", "failspy", "arcee-ai", "gguf",
    }
)

# Embedding-model families (used by the smart picker to keep embedding models
# out of chat candidate sets and vice versa).
_EMBEDDING_FAMILY_RE = re.compile(
    r"(embed|bge|e5|gte|minilm|arctic-embed|mxbai|jina|nomic|paraphrase|sentence)"
)

# Vision-capable families (a coarse signal; explicit metadata wins).
_VISION_FAMILY_RE = re.compile(r"(llava|vl\b|-vl|vision|moondream|minicpm-v|pixtral)")

# Family detection: ordered (regex, slug) pairs matched against the normalized
# name (lowercase, '_' -> '-', org prefix removed). First match wins, so more
# specific patterns (with versions) must precede generic ones.
_FAMILY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p), slug)
    for p, slug in [
        # Qwen
        (r"qwen[-.]?3", "qwen3"),
        (r"qwen[-.]?2[-.]5|qwen2\.5", "qwen2.5"),
        (r"qwen[-.]?2", "qwen2"),
        (r"qwq", "qwq"),
        (r"qwen", "qwen"),
        # Llama (meta- prefix optional; "3.1"/"3-1"/"31" spellings)
        (r"(?:meta-)?llama[-.]?3[-.]3", "llama3.3"),
        (r"(?:meta-)?llama[-.]?3[-.]2", "llama3.2"),
        (r"(?:meta-)?llama[-.]?3[-.]1", "llama3.1"),
        (r"(?:meta-)?llama[-.]?3", "llama3"),
        (r"(?:meta-)?llama[-.]?2", "llama2"),
        (r"codellama", "codellama"),
        (r"tinyllama", "tinyllama"),
        # DeepSeek
        (r"deepseek[-.]?r1", "deepseek-r1"),
        (r"deepseek[-.]?v3", "deepseek-v3"),
        (r"deepseek[-.]?v2", "deepseek-v2"),
        (r"deepseek[-.]?coder", "deepseek-coder"),
        (r"deepseek", "deepseek"),
        # Mistral family
        (r"mixtral", "mixtral"),
        (r"mistral[-.]?small", "mistral-small"),
        (r"mistral[-.]?large", "mistral-large"),
        (r"mistral[-.]?nemo", "mistral-nemo"),
        (r"ministral", "ministral"),
        (r"magistral", "magistral"),
        (r"devstral", "devstral"),
        (r"codestral", "codestral"),
        (r"mistral", "mistral"),
        # Google
        (r"gemma[-.]?3n", "gemma3n"),
        (r"gemma[-.]?3", "gemma3"),
        (r"gemma[-.]?2", "gemma2"),
        (r"gemma", "gemma"),
        # Microsoft
        (r"phi[-.]?4", "phi4"),
        (r"phi[-.]?3[-.]5", "phi3.5"),
        (r"phi[-.]?3", "phi3"),
        (r"phi[-.]?2", "phi2"),
        # OpenAI open-weights
        (r"gpt[-.]?oss", "gpt-oss"),
        # Others
        (r"glm[-.]?4", "glm4"),
        (r"granite", "granite"),
        (r"command[-.]?r[-.]?plus", "command-r-plus"),
        (r"command[-.]?r", "command-r"),
        (r"smollm[-.]?3|smollm3", "smollm3"),
        (r"smollm[-.]?2|smollm2", "smollm2"),
        (r"smollm", "smollm"),
        (r"starcoder[-.]?2", "starcoder2"),
        (r"starcoder", "starcoder"),
        (r"olmo[-.]?2", "olmo2"),
        (r"olmo", "olmo"),
        (r"internlm[-.]?2", "internlm2"),
        (r"internlm", "internlm"),
        (r"minicpm", "minicpm"),
        (r"falcon[-.]?3", "falcon3"),
        (r"falcon", "falcon"),
        (r"yi[-.]?1[-.]5|yi-1\.5", "yi1.5"),
        (r"\byi\b|^yi-", "yi"),
        (r"kimi", "kimi"),
        (r"seed[-.]?oss", "seed-oss"),
        (r"ernie", "ernie"),
        (r"exaone", "exaone"),
        (r"aya", "aya"),
        (r"solar", "solar"),
        (r"dbrx", "dbrx"),
        (r"hunyuan", "hunyuan"),
        (r"pixtral", "pixtral"),
        (r"moondream", "moondream"),
        (r"llava", "llava"),
        # Embedding families
        (r"nomic[-.]?embed", "nomic-embed"),
        (r"mxbai[-.]?embed", "mxbai-embed"),
        (r"snowflake[-.]?arctic[-.]?embed", "arctic-embed"),
        (r"bge[-.]?m3", "bge-m3"),
        (r"bge", "bge"),
        (r"gte", "gte"),
        (r"all[-.]?minilm|minilm", "minilm"),
        (r"embeddinggemma", "embeddinggemma"),
    ]
)

# Size token: "7b", "0.5b", "70b", "135m", "8x7b" (MoE: total = n*m).
_SIZE_RE = re.compile(r"^(\d+(?:\.\d+)?)([bm])$")
_MOE_SIZE_RE = re.compile(r"^(\d+)x(\d+(?:\.\d+)?)([bm])$")
_VERSION_RE = re.compile(r"^v\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class ModelIdentity:
    """Parsed identity of a local model id. See the module docstring."""

    raw: str
    canonical: str
    family: str | None = None
    size_b: float | None = None
    variant: str = ""  # "instruct" | "base" | "code" | "vision" | ""
    quant: str | None = None
    fmt: str | None = None
    version: str | None = None
    fine_tune: str | None = None
    abliterated: bool = False
    base_canonical: str | None = None
    confidence: float = 1.0
    tokens: tuple[str, ...] = field(default_factory=tuple, repr=False)

    @property
    def is_embedding(self) -> bool:
        return bool(_EMBEDDING_FAMILY_RE.search(self.canonical))

    @property
    def is_vision(self) -> bool:
        return bool(_VISION_FAMILY_RE.search(self.canonical)) or self.variant == "vision"


def _strip_org(model_id: str) -> tuple[str | None, str]:
    """Split ``org/name`` -> (org, name); tolerate deeper paths (last part wins)."""
    if "/" not in model_id:
        return None, model_id
    parts = [p for p in model_id.split("/") if p]
    if not parts:
        return None, ""
    if len(parts) == 1:
        return None, parts[0]
    return parts[0].lower(), parts[-1]


def _normalize(name: str) -> str:
    """Lowercase and normalize separators (``_`` -> ``-``); keep dots (versions)."""
    return name.lower().replace("_", "-").strip("-")


def _tokenize(name: str) -> list[str]:
    """Split a normalized name into tokens on ``-`` and ``:``.

    GGUF quants like ``q4_k_m`` were normalized to ``q4-k-m``; re-join runs of
    quant-ish fragments so the quant survives tokenization as one token.
    """
    rough = [t for t in re.split(r"[-:]+", name) if t]
    out: list[str] = []
    i = 0
    while i < len(rough):
        tok = rough[i]
        # Re-join GGUF quant fragments: q4 k m -> q4_k_m ; iq2 xxs -> iq2_xxs
        if re.fullmatch(r"i?q\d+", tok):
            frag = [tok]
            j = i + 1
            while j < len(rough) and re.fullmatch(r"[01klsm]|xxs|xs|s|m|l|nl", rough[j]):
                frag.append(rough[j])
                j += 1
            out.append("_".join(frag))
            i = j
            continue
        out.append(tok)
        i += 1
    return out


def _detect_family(name: str) -> str | None:
    for pat, slug in _FAMILY_PATTERNS:
        if pat.search(name):
            return slug
    return None


def _parse_size(tokens: list[str]) -> float | None:
    for tok in tokens:
        m = _MOE_SIZE_RE.match(tok)
        if m:
            n, per, unit = int(m.group(1)), float(m.group(2)), m.group(3)
            total = n * per
            return total if unit == "b" else total / 1000.0
        m = _SIZE_RE.match(tok)
        if m:
            val, unit = float(m.group(1)), m.group(2)
            # Bare "1b".."999b" is a size; avoid false positives like "v2b".
            return val if unit == "b" else val / 1000.0
    return None


def _parse_quant_and_format(tokens: list[str]) -> tuple[str | None, str | None]:
    quant: str | None = None
    fmt: str | None = None
    for tok in tokens:
        if tok in _FORMAT_TOKENS and fmt is None:
            fmt = _FORMAT_TOKENS[tok]
        if quant is None:
            if _GGUF_QUANT_RE.match(tok):
                quant = tok
            elif tok in _QUANT_TOKENS:
                quant = tok
    # A GGUF quant implies the GGUF format even without an explicit token.
    if fmt is None and quant is not None and _GGUF_QUANT_RE.match(quant):
        fmt = "gguf"
    # awq/gptq/exl2 double as both quant and packaging format.
    if fmt is None and quant in ("awq", "gptq", "exl2", "exl3"):
        fmt = quant
    return quant, fmt


def _parse_variant(tokens: list[str], name: str) -> str:
    toks = set(tokens)
    if toks & _VISION_TOKENS - {"v"} or _VISION_FAMILY_RE.search(name):
        return "vision"
    if toks & _CODE_TOKENS:
        return "code"
    if toks & _INSTRUCT_TOKENS:
        return "instruct"
    if toks & _BASE_TOKENS:
        return "base"
    return ""


def quant_quality_penalty(quant: str | None) -> float:
    """Quality-score penalty (0..~0.15) attributed to quantization loss.

    Rough, monotone heuristic: heavier quants lose more. Unknown/absent quant
    is treated as unquantized (no penalty)."""
    if not quant:
        return 0.0
    q = quant.lower()
    if q.startswith(("iq1", "iq2", "q2")):
        return 0.12
    if q.startswith(("iq3", "q3", "3bit")):
        return 0.08
    if q.startswith(("q4", "iq4", "int4", "4bit", "nf4", "w4")) or q in (
        "awq", "gptq", "exl2", "exl3", "fp4", "mxfp4",
    ):
        return 0.04
    if q.startswith(("q5", "5bit")):
        return 0.02
    if q.startswith(("q6", "q8", "int8", "8bit", "fp8", "w8", "6bit")):
        return 0.01
    return 0.0  # fp16/bf16/f32 — effectively lossless


def identify(model_id: str) -> ModelIdentity:
    """Parse *model_id* into a :class:`ModelIdentity`. Never raises."""
    raw = model_id or ""
    org, name = _strip_org(raw.strip())
    # Strip file extensions before tokenizing (".gguf" both marks the format
    # and would otherwise glue itself to the quant token).
    fmt_from_ext: str | None = None
    lowered = name.lower()
    for ext, f in ((".gguf", "gguf"), (".safetensors", "safetensors"), (".onnx", "onnx")):
        if lowered.endswith(ext):
            name = name[: -len(ext)]
            fmt_from_ext = f
            break

    norm = _normalize(name)
    tokens = _tokenize(norm)

    family = _detect_family(norm)
    size_b = _parse_size(tokens)
    quant, fmt = _parse_quant_and_format(tokens)
    if fmt is None:
        fmt = fmt_from_ext
    # An mlx-community repack is MLX format even without a token.
    if fmt is None and org == "mlx-community":
        fmt = "mlx"
    variant = _parse_variant(tokens, norm)
    version = next((t for t in tokens if _VERSION_RE.match(t)), None)

    abliterated = any(t in _ABLITERATED_TOKENS for t in tokens)
    fine_tune = next((t for t in tokens if t in _FINE_TUNE_TOKENS), None)
    # Fine-tunes are chat-tuned by definition; treat an unmarked one as instruct.
    if fine_tune and not variant:
        variant = "instruct"

    # Canonical slug: family-size-variant, then fine-tune / abliterated
    # markers. Falls back to the cleaned name when the family is unknown.
    def _slug(parts: list[str]) -> str:
        return "-".join(p for p in parts if p)

    size_token = None
    if size_b is not None:
        size_token = (
            f"{size_b:g}b" if size_b >= 1 else f"{int(round(size_b * 1000))}m"
        )

    if family:
        base_parts = [family, size_token or "", variant]
        base_canonical = _slug(base_parts)
        parts = list(base_parts)
        if fine_tune:
            parts.append(fine_tune)
        if abliterated:
            parts.append("abliterated")
        if version and not fine_tune:
            parts.append(version)
        canonical = _slug(parts)
        confidence = 1.0 if size_b is not None else 0.75
        if fine_tune or abliterated:
            confidence *= 0.9
    else:
        # Unknown family: canonical = name minus quant/format/noise tokens.
        noise = set(_FORMAT_TOKENS) | _QUANT_TOKENS | {"latest"}
        kept = [
            t for t in tokens
            if t not in noise and not _GGUF_QUANT_RE.match(t)
        ]
        canonical = _slug(kept) or norm or raw.lower()
        base_canonical = None
        confidence = 0.35

    # A fine-tune / abliterated variant inherits from its base; a plain model
    # is its own base.
    if family and not (fine_tune or abliterated):
        base_canonical = canonical if not version else _slug(
            [family, size_token or "", variant]
        )

    return ModelIdentity(
        raw=raw,
        canonical=canonical,
        family=family,
        size_b=size_b,
        variant=variant,
        quant=quant,
        fmt=fmt,
        version=version,
        fine_tune=fine_tune,
        abliterated=abliterated,
        base_canonical=base_canonical,
        confidence=confidence,
        tokens=tuple(tokens),
    )
