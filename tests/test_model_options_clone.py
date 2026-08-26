import ast
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "spectrum.py", ROOT / "nodes.py")


def _deepcopy_list_dict(value, memo=None):
    """Test double for ComfyUI's container-only deepcopy contract."""
    if memo is None:
        memo = {}

    value_id = id(value)
    if value_id in memo:
        return memo[value_id]

    if isinstance(value, dict):
        result = {
            _deepcopy_list_dict(key, memo): _deepcopy_list_dict(item, memo)
            for key, item in value.items()
        }
    elif isinstance(value, list):
        result = [_deepcopy_list_dict(item, memo) for item in value]
    else:
        result = value

    memo[value_id] = result
    return result


def _load_clone_helper(source_path):
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_clone_model_options"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "comfy": types.SimpleNamespace(
            utils=types.SimpleNamespace(deepcopy_list_dict=_deepcopy_list_dict)
        )
    }
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace["_clone_model_options"]


class ModelOptionsCloneTest(unittest.TestCase):
    def test_helpers_copy_only_dict_and_list_containers(self):
        tensor_state = object()

        for source_path in SOURCES:
            with self.subTest(source=source_path.name):
                original = {
                    "transformer_options": {
                        "states": [tensor_state],
                    }
                }
                model = types.SimpleNamespace(model_options=original)

                _load_clone_helper(source_path)(model)

                self.assertIsNot(model.model_options, original)
                self.assertIsNot(
                    model.model_options["transformer_options"],
                    original["transformer_options"],
                )
                self.assertIsNot(
                    model.model_options["transformer_options"]["states"],
                    original["transformer_options"]["states"],
                )
                self.assertIs(
                    model.model_options["transformer_options"]["states"][0],
                    tensor_state,
                )

                model.model_options["transformer_options"]["states"].append(object())
                self.assertEqual(
                    original["transformer_options"]["states"],
                    [tensor_state],
                )


if __name__ == "__main__":
    unittest.main()
