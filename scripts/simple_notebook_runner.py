"""Minimal notebook executor for environments without nbconvert/nbclient."""

from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _split_last_expr(source: str) -> tuple[str, ast.Expression | None]:
    tree = ast.parse(source)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        body = tree.body[:-1]
        last_expr = ast.Expression(tree.body[-1].value)
        ast.fix_missing_locations(last_expr)
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        return compile(module, "<notebook-cell>", "exec"), compile(last_expr, "<notebook-cell>", "eval")
    return compile(tree, "<notebook-cell>", "exec"), None


def _display_data(value: Any) -> dict[str, Any]:
    data = {"text/plain": repr(value)}
    if hasattr(value, "_repr_html_"):
        html = value._repr_html_()
        if html:
            data["text/html"] = html
    return {
        "output_type": "execute_result",
        "execution_count": None,
        "metadata": {},
        "data": data,
    }


def execute_notebook(path: str | Path) -> None:
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {}
    execution_count = 0

    def display(value: Any) -> None:
        current_outputs.append(_display_data(value))

    namespace["display"] = display

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue

        execution_count += 1
        source = "".join(cell.get("source", []))
        current_outputs: list[dict[str, Any]] = []
        stdout = io.StringIO()

        try:
            exec_code, eval_code = _split_last_expr(source)
            with contextlib.redirect_stdout(stdout):
                exec(exec_code, namespace)
                if eval_code is not None:
                    value = eval(eval_code, namespace)
                    if value is not None:
                        current_outputs.append(_display_data(value))
        except Exception:
            current_outputs.append(
                {
                    "output_type": "error",
                    "ename": sys.exc_info()[0].__name__,
                    "evalue": str(sys.exc_info()[1]),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
            cell["execution_count"] = execution_count
            cell["outputs"] = current_outputs
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
            raise

        text = stdout.getvalue()
        if text:
            current_outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": text})

        cell["execution_count"] = execution_count
        for output in current_outputs:
            if output.get("output_type") == "execute_result":
                output["execution_count"] = execution_count
        cell["outputs"] = current_outputs

    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.simple_notebook_runner NOTEBOOK.ipynb")
    execute_notebook(sys.argv[1])
