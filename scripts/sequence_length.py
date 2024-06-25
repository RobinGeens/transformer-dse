"""
Show what happens if the sequence length becomes very small / large.
"""

import itertools
import os
import sys
import pickle
from zigzag import api
from zigzag.opt.loma.LomaEngine import NoValidLoopOrderingFoundException
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

sys.path.append(os.getcwd())

from src.export_onnx import export_transformer_to_onnx
from src.config import ALL_MODELS, BATCH_SIZE, PAPER_MODELS, W4A16, W8A8, LLMConfig
from src.util import (
    CME_T,
    accelerator_path,
    generalize_layer_name,
    get_cmes_full_model,
    get_cmes_to_plot,
)
from src.plots import (
    plot_energy_compare_minimal,
    plot_energy_clean,
    plot_energy_minimal,
    plot_latency_clean,
)

quant = W8A8
accelerators = ["generic_array_8b", "generic_array_edge_8b"]
mapping_path = "inputs/mapping/weight_unrolled_256.yaml"
out_path = "outputs/sequence_length"

scenarios = [
    (128, 128),
    (16_384, 64),
    (128, 16_384),
]

simulation_models: list[LLMConfig] = []
for model, scenario in itertools.product(PAPER_MODELS, scenarios):
    prefill_len, decode_len = scenario
    model_sim = model.to_simulatable_config()
    model_sim.prefill_size = prefill_len
    model_sim.decode_idx = prefill_len + decode_len // 2
    model_sim.decode_simulation_multiplier = decode_len
    simulation_models.append(model_sim)


def run_experiment():
    for model_sim, accelerator, do_prefill in itertools.product(simulation_models, accelerators, [True, False]):

        identifier = (
            f"{model_sim.parameterized_name}_{quant.name}_"
            f"{f'prefill={model_sim.prefill_size}' if do_prefill else f'decode={model_sim.decode_idx}'}"
        )
        experiment_name = f"{identifier}_{accelerator}"
        dump_path = f"{out_path}/{experiment_name}"
        onnx_path = f"outputs/onnx/{identifier}.onnx"
        pickle_filename = f"{dump_path}/cmes.pickle"

        print(f"--- Running {experiment_name} ---")

        try:
            if not os.path.exists(onnx_path):
                export_transformer_to_onnx(model_sim, quant, path=onnx_path, prefill=do_prefill)

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
            bar_plot_cost_model_evaluations_breakdown(
                cmes_to_plot, save_path=f"{dump_path}/interesting_layers_single.png"
            )

            # Compute generalized results for full LLM
            original_config = next(filter(lambda x: x.name == model_sim.name, PAPER_MODELS))
            complete_result_cmes = get_cmes_full_model(cmes_to_plot, original_config, prefill=do_prefill)
            bar_plot_cost_model_evaluations_breakdown(
                complete_result_cmes, save_path=f"{dump_path}/interesting_layers_full.png"
            )

            plot_energy_clean(complete_result_cmes, f"{out_path}/{experiment_name}/energy.png")
            plot_latency_clean(complete_result_cmes, f"{out_path}/{experiment_name}/latency.png")

        except NoValidLoopOrderingFoundException:
            print(f"Failed {experiment_name}")


if __name__ == "__main__":
    # run_experiment()

    # For each model: combine archs:
    for model_sim in simulation_models:

        cmes_per_arch: list[list[CME_T]] = []

        for accelerator, do_prefill in itertools.product(accelerators, [True, False]):
            identifier = (
                f"{model_sim.parameterized_name}_{quant.name}_"
                f"{f'prefill={model_sim.prefill_size}' if do_prefill else f'decode={model_sim.decode_idx}'}"
            )
            experiment_name = f"{identifier}_{accelerator}"
            dump_path = f"{out_path}/{experiment_name}"
            pickle_filename = f"{dump_path}/cmes.pickle"

            with open(pickle_filename, "rb") as fp:
                cmes: list[CME_T] = pickle.load(fp)

            cmes = get_cmes_to_plot(cmes)
            original_config = next(filter(lambda x: x.name == model_sim.name, PAPER_MODELS))
            cmes = get_cmes_full_model(cmes, original_config, prefill=do_prefill)
            cmes_per_arch.append((cmes))

        plot_energy_compare_minimal(
            cmes_per_arch,
            groups=["Cloud prefill", "Cloud decode", "Edge prefill", "Edge decode"],
            title=model_sim.name,
            filename=f"{out_path}/compare_{model_sim.name}_({model_sim.prefill_size}_{model_sim.decode_idx}).png",
        )
