"""
Make a single plot for 1 model on 1 architecture
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
from src.config import ALL_MODELS, BATCH_SIZE, LLAMA_2_7B, W1A32, W1A8, W32A32, W4A16, W4A8, W8A8
from src.util import (
    CME_T,
    accelerator_path,
    generalize_layer_name,
    get_cmes_full_model,
    get_cmes_to_plot,
)
from src.plots import (
    plot_energy_and_latency,
    plot_energy_compare,
    plot_energy_compare_minimal,
    plot_energy_clean,
    plot_energy_minimal,
    plot_latency_clean,
    plot_latency_compare,
)

model = LLAMA_2_7B
quants = [W1A32, W4A16, W32A32]
accelerator = "generic_array_32b"
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/exp_quant"


def run_experiment():
    for quant, do_prefill in itertools.product(quants, [True, False]):

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

    for quant in quants:
        cmes_per_group: list[list[CME_T]] = []

        for do_prefill in [True, False]:
            experiment_name = (
                f"{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}_{accelerator}"
            )
            dump_path = f"{out_path}/{experiment_name}"
            pickle_filename = f"{dump_path}/cmes.pickle"
            with open(pickle_filename, "rb") as fp:
                cmes: list[CME_T] = pickle.load(fp)

            cmes = get_cmes_to_plot(cmes)
            cmes = get_cmes_full_model(cmes, model, prefill=do_prefill)
            cmes_per_group.append((cmes))

        plot_energy_and_latency(
            cmes_per_group,
            supergroups=["Prefill", "Decode"],
            title=f"{model.name} ({quant.name})",
            filename=f"{out_path}/energy_and_latency_{quant.name}_{model.name}.png",
        )
