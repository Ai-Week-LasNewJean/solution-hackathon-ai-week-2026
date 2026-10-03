"""Validacion de forma contra schema/submission.schema.json, mas validate()
de scripts/evaluate.py -- el segundo es literalmente lo que corre el jurado.

Los "then" del schema exigen las llaves por formato (respuesta_correcta,
etc.) sin condicionarlas a `abstencion`, aunque la descripcion de esas
propiedades dice "con abstencion true se admite null" -- el enum de
respuesta_correcta ni siquiera incluye null. evaluate.validate() (la
verificacion real del jurado) SI exime esas llaves cuando abstencion es
true. Para no generar falsos positivos sobre entregas correctas, aqui el
`allOf` por formato solo se aplica a filas que no se abstuvieron; el resto
del schema (llaves base, tipos) se valida siempre.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

import src  # noqa: F401
from common import read_jsonl
from evaluate import validate as evaluate_validate

from src import config


def check(submission_path: Path, expected_ids: set[int] | None = None) -> list[str]:
    """Devuelve la lista de errores (vacia si la entrega es valida). Corre
    tanto el schema JSON formal como validate() de evaluate.py."""
    schema = json.loads((config.SCHEMA / "submission.schema.json").read_text(encoding="utf-8"))
    schema_sin_abstencion = {k: v for k, v in schema.items() if k != "allOf"}
    rows = read_jsonl(submission_path)

    errores: list[str] = []
    for i, row in enumerate(rows):
        effective_schema = schema_sin_abstencion if row.get("abstencion") else schema
        try:
            jsonschema.validate(row, effective_schema)
        except jsonschema.ValidationError as exc:
            errores.append(f"linea {i + 1} (id={row.get('id')}): {exc.message}")

    if expected_ids is not None:
        errores += evaluate_validate(rows, expected_ids)
    return errores


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--submission", type=Path, required=True)
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    args = ap.parse_args()

    if args.split == "sample":
        expected = {r["id"] for r in read_jsonl(config.DATA / "sample_50.jsonl")}
    else:
        expected = {r["id"] for r in read_jsonl(config.DATA / "test_992.jsonl")}

    errores = check(args.submission, expected)
    if errores:
        print(f"{len(errores)} error(es):", file=sys.stderr)
        for e in errores[:40]:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("schema_check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
