"""Architecture guardrails for the deliberately small mini package."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "robometanorm"
TEST_ROOT = PROJECT_ROOT / "tests"

EXPECTED_PRODUCTION_FILES = {
    "__init__.py",
    "__main__.py",
    "adapters/__init__.py",
    "adapters/filesystem.py",
    "cli/__init__.py",
    "cli/main.py",
    "evidence.py",
    "models.py",
    "pipeline.py",
    "standard.py",
    "vlm.py",
    "writer.py",
}
EXPECTED_TEST_FILES = {
    "__init__.py",
    "mini_fixtures.py",
    "integration/__init__.py",
    "integration/test_cli.py",
    "unit/__init__.py",
    "unit/test_architecture.py",
    "unit/test_discovery.py",
    "unit/test_evidence.py",
    "unit/test_models.py",
    "unit/test_pipeline.py",
    "unit/test_standard.py",
    "unit/test_vlm.py",
    "unit/test_writer.py",
}
LEGACY_MODULE_PARTS = (
    "application",
    "camera",
    "domain",
    "machine",
    "writers",
    "robot_identity",
    "episode_sampling",
)


class MiniArchitectureTest(unittest.TestCase):
    def test_production_python_tree_is_exact(self) -> None:
        actual = {
            path.relative_to(PACKAGE_ROOT).as_posix()
            for path in PACKAGE_ROOT.rglob("*.py")
        }
        self.assertEqual(actual, EXPECTED_PRODUCTION_FILES)

    def test_test_python_tree_is_exact(self) -> None:
        actual = {
            path.relative_to(TEST_ROOT).as_posix()
            for path in TEST_ROOT.rglob("*.py")
        }
        self.assertEqual(actual, EXPECTED_TEST_FILES)

    def test_legacy_directories_and_root_modules_are_physically_absent(self) -> None:
        legacy_paths = (
            PACKAGE_ROOT / "application",
            PACKAGE_ROOT / "camera",
            PACKAGE_ROOT / "domain",
            PACKAGE_ROOT / "machine",
            PACKAGE_ROOT / "writers",
            PACKAGE_ROOT / "robot_identity.py",
            PACKAGE_ROOT / "episode_sampling.py",
        )
        for path in legacy_paths:
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertFalse(path.exists())

    def test_production_source_contains_no_forbidden_hardcoding_or_modalities(self) -> None:
        forbidden = (
            "airbot",
            "agilex",
            "galaxea",
            "galbot",
            "aloha",
            "franka",
            "unitree",
            "image_top_left",
            "urdf",
            "tactile",
            "audio",
            "触觉",
            "声音",
            "音频",
        )
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8").casefold()
            for token in forbidden:
                with self.subTest(path=path.name, token=token):
                    self.assertNotIn(token.casefold(), source)

    def test_confidence_default_occurs_once_and_parser_reuses_the_constant(self) -> None:
        production_sources = {
            path: path.read_text(encoding="utf-8")
            for path in PACKAGE_ROOT.rglob("*.py")
        }
        occurrences = sum(source.count("0.85") for source in production_sources.values())
        self.assertEqual(occurrences, 1)

        cli_path = PACKAGE_ROOT / "cli" / "main.py"
        cli_tree = ast.parse(production_sources[cli_path], filename=str(cli_path))
        assignments = [
            node
            for node in cli_tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "DEFAULT_CONFIDENCE_THRESHOLD"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].value.value, 0.85)

        threshold_arguments = [
            node
            for node in ast.walk(cli_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--confidence-threshold"
        ]
        self.assertEqual(len(threshold_arguments), 1)
        keywords = {keyword.arg: keyword.value for keyword in threshold_arguments[0].keywords}
        self.assertIsInstance(keywords.get("default"), ast.Name)
        self.assertEqual(keywords["default"].id, "DEFAULT_CONFIDENCE_THRESHOLD")
        self.assertTrue(
            any(
                isinstance(node, ast.Name)
                and node.id == "DEFAULT_CONFIDENCE_THRESHOLD"
                for node in ast.walk(keywords["help"])
            )
        )

    def test_production_and_tests_do_not_import_legacy_packages(self) -> None:
        for root in (PACKAGE_ROOT, TEST_ROOT):
            for path in sorted(root.rglob("*.py")):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                for part in LEGACY_MODULE_PARTS:
                    self.assertNotIn("robometanorm" + "." + part, source)
                imported_modules: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported_modules.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        module = "." * node.level + (node.module or "")
                        imported_modules.append(module)
                        imported_modules.extend(
                            module + ("." if node.module else "") + alias.name
                            for alias in node.names
                        )
                for module in imported_modules:
                    with self.subTest(path=path.name, module=module):
                        absolute_prefix = "robometanorm" + "."
                        relative_module = module.lstrip(".").split(".", 1)[0]
                        self.assertFalse(
                            any(
                                module == absolute_prefix + part
                                or module.startswith(absolute_prefix + part + ".")
                                or (module.startswith(".") and relative_module == part)
                                for part in LEGACY_MODULE_PARTS
                            )
                        )

    def test_pyproject_has_one_dependency_and_exact_console_script(self) -> None:
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        dependency_block = re.search(
            r"(?ms)^dependencies\s*=\s*\[(.*?)^\]", pyproject
        )
        self.assertIsNotNone(dependency_block)
        dependencies = re.findall(r'"([^"]+)"', dependency_block.group(1))
        self.assertEqual(dependencies, ["pyarrow>=14.0"])
        lowered = pyproject.casefold()
        self.assertNotIn("numpy", lowered)
        self.assertNotIn("opencv", lowered)
        self.assertNotIn("[project.optional-dependencies]", lowered)

        script_block = re.search(
            r"(?ms)^\[project\.scripts\]\s*\n(.*?)(?=^\[|\Z)", pyproject
        )
        self.assertIsNotNone(script_block)
        script_lines = [
            line.strip()
            for line in script_block.group(1).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            script_lines,
            ['robometanorm = "robometanorm.cli.main:main"'],
        )


if __name__ == "__main__":
    unittest.main()
