"""Prompt loader with YAML frontmatter stripping and {variable} substitution."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str, **kwargs: object) -> str:
    """Load a prompt file by name and substitute {variable} placeholders.

    Args:
        name: Filename without extension, e.g. ``"scout_characterize_v1"``.
        **kwargs: Substitution variables for ``str.format_map``.

    Returns:
        The prompt body with YAML frontmatter stripped and placeholders filled.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        KeyError: If a placeholder in the template has no matching kwarg.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    body = _strip_frontmatter(text)
    if kwargs:
        body = body.format_map(kwargs)
    return body


def _strip_frontmatter(text: str) -> str:
    """Remove YAML frontmatter delimited by leading ``---`` ... ``---``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return text
