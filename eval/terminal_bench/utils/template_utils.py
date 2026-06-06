from __future__ import annotations

from pathlib import Path


def render_prompt_template(template_path: Path, instruction: str) -> str:
    return template_path.read_text().replace("{{ instruction }}", instruction)
