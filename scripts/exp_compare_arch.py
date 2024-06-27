"""
Make plots to compare different architectures on the same models
"""

import itertools
import os
import sys
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

sys.path.append(os.getcwd())
from src.export_onnx import export_transformer_to_onnx
from src.config import ALL_MODELS, BATCH_SIZE, LLAMA_2_7B, W4A16, W4A8, W8A8
from src.util import (
    CME_T,
    accelerator_path,
    generalize_layer_name,
    get_cmes_full_model,
    get_cmes_to_plot,
)
from src.plots import (
    plot_energy_and_latency_minimal,
    plot_energy_compare_minimal,
    plot_energy_clean,
    plot_energy_minimal,
    plot_latency_clean,
    plot_latency_compare,
    plot_latency_compare_minimal,
)

models = [LLAMA_2_7B]
quant = W4A16
accelerators = ["generic_array_32b", "generic_array_edge_32b"]
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/exp_compare_arch"


def run_experiment():
    for model, accelerator, do_prefill in itertools.product(models, accelerators, [True, False]):

        experiment_name = (
            f"{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}_{accelerator}"
        )
        dump_path = f"{out_path}/{experiment_name}"
        onnx_path = f"outputs/onnx/{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}.onnx"
        pickle_filename = f"{dump_path}/cmes.pickle"

        print(f"--- Running {experiment_name} ---")

        if not os.path.exists(onnx_path):
            export_transformer_to_onnx(model.to_simulatable_config(), quant, path=onnx_path, prefill=do_prefill)

        api.get_hardware_performance_zigzag(
            workload=onnx_path,
            accelerator=accelerator_path(accelerator),
            mapping=mapping_path,
            opt="EDP",
            dump_folder=dump_path,
            pickle_filename=pickle_filename,
            nb_spatial_mappings_generated=3,
        )

        with open(pickle_filename, "rb") as fp:
            cmes = pickle.load(fp)

        # Plots for single layers
        cmes_to_plot = get_cmes_to_plot(cmes)
        bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{dump_path}/all_layers_single.png")
        bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path=f"{dump_path}/interesting_layers_single.png")

        # Compute generalized results for full LLM
        complete_result_cmes = get_cmes_full_model(cmes_to_plot, model, prefill=do_prefill)
        bar_plot_cost_model_evaluations_breakdown(
            complete_result_cmes, save_path=f"{dump_path}/interesting_layers_full.png"
        )

        plot_energy_clean(complete_result_cmes, f"{out_path}/{experiment_name}/energy.png")
        plot_latency_clean(complete_result_cmes, f"{out_path}/{experiment_name}/latency.png")


if __name__ == "__main__":
    # run_experiment()

    # For each model: combine archs:
    for model in models:

        cmes_per_arch: list[list[CME_T]] = []

        for arch, do_prefill in itertools.product(accelerators, [True, False]):
            experiment_name = f"{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}_{arch}"
            dump_path = f"{out_path}/{experiment_name}"
            pickle_filename = f"{dump_path}/cmes.pickle"
            with open(pickle_filename, "rb") as fp:
                cmes: list[CME_T] = pickle.load(fp)

            cmes = get_cmes_to_plot(cmes)
            cmes = get_cmes_full_model(cmes, model, prefill=do_prefill)
            cmes_per_arch.append((cmes))

        groups = ["Cloud\nprefill", "Cloud\ndecode", "Edge\nprefill", "Edge\ndecode"]

        plot_energy_and_latency_minimal(
            cmes_per_arch,
            groups=groups,
            title=f"{model.name} ({quant.name})",
            filename=f"{out_path}/compare_energy_and_latency_{model.name}.png",
        )
