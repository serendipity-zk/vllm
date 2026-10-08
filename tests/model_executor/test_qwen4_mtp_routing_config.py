# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression of the pure draft/config and capture-binding helpers."""
import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, environment):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), environment)
    return environment


@dataclass
class Aux:
    enable_return_routed_experts: bool = False

    @property
    def enabled(self):
        return self.enable_return_routed_experts


@dataclass
class Config:
    model_config: object
    speculative_config: object
    aux_output_config: Aux = field(default_factory=Aux)
    quant_config: object = None


class DraftRoutingConfigTests(unittest.TestCase):
    def test_draft_has_fresh_disabled_connector_and_restored_quant(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                quant = SimpleNamespace(ignored_layers=["mtp.layers.0.ffn"])
                env = functions("vllm/models/qwen4_exp/nvidia/mtp.py", {
                    "_remap_ignored_layers", "_remap_quantized_layers",
                    "_make_draft_vllm_config",
                }, {
                    "VllmConfig": Config, "AuxOutputConfig": Aux, "replace": replace,
                    "get_draft_quant_config": lambda _: quant,
                    "configure_quant_config": lambda *_: None,
                    "Qwen4ExpMTP": object, "re": __import__("re"),
                })
                draft_model = object()
                target = Config(object(), SimpleNamespace(draft_model_config=draft_model), Aux(enabled))
                draft = env["_make_draft_vllm_config"](target, 48)
                self.assertIs(draft.model_config, draft_model)
                self.assertFalse(draft.aux_output_config.enabled)
                self.assertIsNot(draft.aux_output_config, target.aux_output_config)
                self.assertEqual(target.aux_output_config.enabled, enabled)
                self.assertIs(draft.quant_config, quant)
                self.assertEqual(quant.ignored_layers, ["mtp.layers.48.ffn"])

    def test_ordinary_native_and_other_models_layer_counts(self):
        env = functions("vllm/model_executor/layers/fused_moe/routed_experts_capturer.py", {
            "num_target_layers", "num_capture_layers",
        }, {"VllmConfig": Config})
        for model_type, method, fields, expected in (
            ("qwen4_exp_text", None, {"mtp_num_hidden_layers": 1}, 48),
            ("qwen4_exp_text", "mtp", {"mtp_num_hidden_layers": 1}, 49),
            ("qwen4_exp", "mtp", {"mtp_num_hidden_layers": 1, "num_nextn_predict_layers": 1}, 49),
            ("deepseek", "mtp", {"num_nextn_predict_layers": 2, "mtp_num_hidden_layers": 1}, 50),
            ("qwen4_exp_text", "eagle", {"mtp_num_hidden_layers": 1}, 48),
        ):
            with self.subTest(model_type=model_type, method=method):
                model = SimpleNamespace(get_total_num_hidden_layers=lambda: 48,
                    hf_text_config=SimpleNamespace(model_type=model_type, **fields))
                c = Config(model, None if method is None else SimpleNamespace(method=method))
                self.assertEqual(env["num_capture_layers"](c), expected)

    def test_target_capturer_is_bound_to_actual_draft_model(self):
        path = ROOT / "vllm/v1/worker/gpu/model_runner.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GPUModelRunner")
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_bind_drafter_routed_experts_capture")
        calls = []
        env = {"RoutedExpertsCapturer": object, "AutoRegressiveSpeculator": SimpleNamespace,
               "bind_routed_experts_capturer": lambda model, capturer: calls.append((model, capturer))}
        exec(compile(ast.Module([node], type_ignores=[]), str(path), "exec"), env)
        draft_model, target_capturer = object(), object()
        speculator = SimpleNamespace(model=draft_model)
        env[node.name](SimpleNamespace(speculator=speculator), target_capturer)
        self.assertEqual(calls, [(draft_model, target_capturer)])
        self.assertIs(speculator.routed_experts_capturer, target_capturer)


if __name__ == "__main__":
    unittest.main()
