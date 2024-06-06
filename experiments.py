import itertools
import json
import os
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

from export_onnx import export_transformer_to_onnx
from src.config import ALL_MODELS, LLAMA_1_7B, W4A16, W4A8, W8A8, LLMConfig

models = [LLAMA_1_7B]  ##ALL_MODELS
quants = [W8A8, W4A16, W4A8]
accelerators = ["tpu_like"]  # , "tpu_big_sram"]

mapping_path = "inputs/mapping/default.yaml"

layers_to_plot = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]


def accelerator_path(accelerator: str):
    return f"inputs/hardware/{accelerator}.yaml"


def get_post_simulation_factor(cfg: LLMConfig, layer: str):
    """The model is simulated with reduced parameters i.e. only one layer. This function returns the factor with which
    the results for the given layer have to be multiplied in order to come to the result for the full model"""
    if "_proj" in layer:
        # K, Q, V and output projection
        return 4 * cfg.num_head * cfg.num_layer
    elif "mul_" in layer:
        return cfg.num_head * cfg.num_layer
    elif "feedforward_" in layer:
        return cfg.num_layer
    else:
        return 1


def generalize_layer_name(layer: str):
    if "key_proj" in layer:
        return "linear projection"
    elif "mul_qk_t" in layer:
        return "mul K*Q^T"
    elif "mul_logits" in layer:
        return "mul attn*V"
    elif "feedforward_expand" in layer:
        return "MLP layer 1"
    elif "feedforward_contract" in layer:
        return "MLP layer 2"
    else:
        return layer


if __name__ == "__main__":
    for model, accelerator, quant in itertools.product(models, accelerators, quants):

        experiment_name = f"{model.name}_{quant.name}_{accelerator}"
        dump_path = f"outputs/{experiment_name}"
        onnx_path = f"outputs/onnx/{model.name}_{quant.name}.onnx"
        pickle_filename = f"{dump_path}/cmes.pickle"

        if not os.path.exists(onnx_path):
            export_transformer_to_onnx(model.to_simulatable_config(), quant, path=onnx_path)

        api.get_hardware_performance_zigzag(
            workload=onnx_path,
            accelerator=accelerator_path(accelerator),
            mapping=mapping_path,
            opt="EDP",
            dump_folder=dump_path,
            pickle_filename=pickle_filename,
            nb_spatial_mappings_generated=1,
        )

        with open(pickle_filename, "rb") as fp:
            cmes = pickle.load(fp)

        # Plots for single layers
        cmes_to_plot = [next(filter(lambda x: name in x.layer.name, cmes)) for name in layers_to_plot]
        bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{dump_path}/all_layers_single.png")
        bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path=f"{dump_path}/interesting_layers_single.png")

        # Compute full results
        complete_result_cmes = [cme * get_post_simulation_factor(model, cme.layer.name) for cme in cmes_to_plot]
        bar_plot_cost_model_evaluations_breakdown(
            complete_result_cmes, save_path=f"{dump_path}/interesting_layers_full.png"
        )
        result_dump_dict = {
            generalize_layer_name(cme.layer.name): cme.__simplejsonrepr__() for cme in complete_result_cmes
        }
        with open(f"{dump_path}/full_model_result.json", "w") as f:
            json.dump(result_dump_dict, f, indent=4)

        # Save which layers are plotted
        with open(f"{dump_path}/info.txt", "w") as f:
            f.write("Layers shown in plot interesting_layers_single:\n")
            for idx, cme in enumerate(cmes_to_plot):
                f.write(f"\t{idx}: {cme.layer.name}\n")
            f.write("Components shown in plot interesting_layers_full:\n")
            for idx, cme in enumerate(cmes_to_plot):
                f.write(f"\t{idx}: {generalize_layer_name(cme.layer.name)}\n")
            f.write("Components shown in plot all_layers_single:\n")
            for idx, cme in enumerate(cmes):
                f.write(f"\t{idx}: {cme.layer.name}\n")
