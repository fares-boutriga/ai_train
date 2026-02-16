from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env_file(env_file: Optional[str]) -> Optional[Path]:
    """Load environment variables from a dotenv file if provided and present."""
    if not env_file:
        return None

    path = Path(env_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Env file not found: {path}. Create it from the matching .example file."
        )

    load_dotenv(dotenv_path=path, override=False)
    return path

