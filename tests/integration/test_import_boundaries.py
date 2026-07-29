import ast
from pathlib import Path

FORBIDDEN = {
    "primary_ml_cka.domain": ("torch", "transformers", "vllm", "primary_ml_cka.experiment"),
    "primary_ml_cka.models": ("primary_ml_cka.experiment",),
    "primary_ml_cka.attack.optimization": (
        "primary_ml_cka.models.targets.generation",
        "primary_ml_cka.models.backends.vllm_generation",
        "primary_ml_cka.models.backends.target_transformers_generation",
    ),
    "primary_ml_cka.attack": (
        "primary_ml_cka.models.targets",
        "primary_ml_cka.models.backends.target_transformers_generation",
        "primary_ml_cka.models.backends.vllm_generation",
    ),
    "primary_ml_cka.evaluation": ("primary_ml_cka.attack.engine",),
}


def imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return tuple(found)


def test_layer_import_boundaries() -> None:
    root = Path("src/primary_ml_cka")
    violations = []
    for path in root.rglob("*.py"):
        module = ".".join(path.with_suffix("").parts[1:])
        for layer, forbidden in FORBIDDEN.items():
            if module == layer or module.startswith(layer + "."):
                for imported in imports(path):
                    if any(
                        imported == item or imported.startswith(item + ".") for item in forbidden
                    ):
                        violations.append(f"{module} imports {imported}")
    assert not violations, "\n".join(violations)
