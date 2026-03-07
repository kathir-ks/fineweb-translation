"""
Shared utility functions for the fineweb-translation pipeline.

Contains text preprocessing (sentence splitting, string cleaning),
the SIGALRM-based timeout decorator (Linux only), and the
preprocess_and_tokenize function used by all tokenization scripts.
"""

import re
import signal
import logging
from unicodedata import normalize

from IndicTransTokenizer import IndicTransTokenizer, IndicProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeout decorator (requires SIGALRM — Linux / macOS only)
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Function call timed out")

signal.signal(signal.SIGALRM, _timeout_handler)


def timeout(seconds):
    """Decorator that raises TimeoutError if the wrapped function
    takes longer than *seconds* to execute.

    .. note:: Uses ``signal.SIGALRM`` — only works on Unix systems.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Text cleaning & sentence splitting
# ---------------------------------------------------------------------------

def clean_string(s: str) -> str:
    """Strip leading/trailing symbols, bullet points, and whitespace."""
    # Remove common junk characters at boundaries
    stripped = s.strip("@#$^&*-_+=[]{}|\\<>/\n")
    stripped = stripped.strip()

    # Strip bullet-point prefixes (•, ○, *, -, numbered lists)
    pattern = r'^\s*(\•|\○|\*|\-|[0-9]+\.)\s*'
    stripped = re.sub(pattern, '', stripped)
    return stripped.strip()


def split_with_delimiter(
    text: str,
    delimiter_pattern: str = (
        r'(?<!\d)\.(?!\d)'   # period not between digits
        r'|(?<!\w)\.(?!\w)'  # period not between word chars
        r'|[?!।|॥؟۔\n](?:\n+)?'  # common sentence-ending punctuation
    ),
) -> list[str]:
    """Split *text* on sentence-ending delimiters, keeping the delimiter
    attached to the preceding segment."""
    lines = re.split(f'({delimiter_pattern})', text)
    if len(lines) % 2 == 0:
        out = [lines[i] + lines[i + 1] for i in range(0, len(lines), 2)]
    else:
        out = [lines[i] + lines[i + 1] for i in range(0, len(lines) - 1, 2)]
        out.append(lines[-1])
    return out


def split_into_sentences(text: str, method: str = "regex") -> list[str]:
    """Normalize *text* and split it into cleaned sentences."""
    split_methods = {
        "regex": split_with_delimiter,
    }
    text = normalize('NFKC', text).lower()
    sents = [
        clean_string(sent.text if not isinstance(sent, str) else sent)
        for sent in split_methods[method](text)
        if len(sent)
    ]
    return [sent for sent in sents if len(sent)]


# ---------------------------------------------------------------------------
# Tokenization helper
# ---------------------------------------------------------------------------

@timeout(1)
def preprocess_and_tokenize(
    tokenizer: IndicTransTokenizer,
    ip: IndicProcessor,
    batch: list[str],
    src_lang: str,
    tgt_lang: str,
) -> dict:
    """Preprocess and tokenize a batch of sentences.

    Returns a dict with keys ``"batch"`` (token ids & attention masks as
    plain lists) and ``"placeholder_entity_maps"``.
    """
    batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
    batch = tokenizer(
        batch,
        padding="longest",
        truncation=True,
        max_length=256,
        src=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    batch = {key: value.tolist() for key, value in batch.items()}
    placeholder_entity_maps = ip.get_placeholder_entity_maps(clear_ple_maps=True)
    return {"batch": batch, "placeholder_entity_maps": placeholder_entity_maps}
