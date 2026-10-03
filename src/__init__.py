"""Paquete del pipeline. Pone scripts/ (oficial, sin modificar) en sys.path
una sola vez para que cualquier modulo pueda hacer `import citations` /
`import common` tal como lo hace scripts/evaluate.py."""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
