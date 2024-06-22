import itertools
import os
import pickle
from zigzag import api
from zigzag.opt.loma.LomaEngine import NoValidLoopOrderingFoundException
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

from scripts.plot_anda import cmes_to_array
from src.export_onnx import export_transformer_to_onnx
from src.config import ALL_MODELS, BATCH_SIZE, W4A16, W8A8
from src.util import (
    CME_T,
    accelerator_path,
    generalize_layer_name,
    get_cmes_full_model,
    get_cmes_to_plot,
)
from src.plots import (
    plot_energy_compare_archs,
    plot_energy_zigzag_clean,
    plot_energy_small,
    plot_latency_zigzag_clean,
)

models = ALL_MODELS
quant = W8A8
accelerators = ["generic_array_8b", "generic_array_edge_8b"]
mapping_path = "inputs/mapping/output_unrolled_256.yaml"
do_prefill = True

out_path = "outputs/compare_arch"


def run_experiment():
    for model, accelerator in itertools.product(models, accelerators):

        experiment_name = (
            f"{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}_{accelerator}"
        )
        dump_path = f"{out_path}/{experiment_name}"
        onnx_path = f"outputs/onnx/{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}.onnx"
        pickle_filename = f"{dump_path}/cmes.pickle"

        print(f"--- Running {experiment_name} ---")

        try:
            if not os.path.exists(onnx_path):
                export_transformer_to_onnx(model.to_simulatable_config(), quant, path=onnx_path)

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
            complete_result_cmes = get_cmes_full_model(cmes_to_plot, model)
            bar_plot_cost_model_evaluations_breakdown(
                complete_result_cmes, save_path=f"{dump_path}/interesting_layers_full.png"
            )

            plot_energy_zigzag_clean(complete_result_cmes, f"{out_path}/{experiment_name}/energy.png")
            plot_latency_zigzag_clean(complete_result_cmes, f"{out_path}/{experiment_name}/latency.png")

            with open(f"{dump_path}/info.txt", "w") as f:
                f.write("Layers shown in plot interesting_layers_single:\n")
                for idx, cme in enumerate(cmes_to_plot):
                    f.write(f"\t{idx}: {cme.layer.name}\n")
                f.write(
                    "\tNote: the linear projection shows a single projection (e.g. key) for ALL heads. The MatMuls "
                    "(attention and logits) are shown for a SINGLE head.\n"
                )
                f.write("Components shown in plot interesting_layers_full:\n")
                for idx, cme in enumerate(cmes_to_plot):
                    f.write(f"\t{idx}: {generalize_layer_name(cme.layer.name)}\n")
                f.write("Components shown in plot all_layers_single:\n")
                for idx, cme in enumerate(cmes):
                    f.write(f"\t{idx}: {cme.layer.name}\n")

        except NoValidLoopOrderingFoundException:
            print(f"Failed {experiment_name}")


if __name__ == "__main__":
    # run_experiment()

    # For each model: combine archs:
    for model in models:

        cmes_per_arch: list[list[CME_T]] = []

        for arch in accelerators:
            experiment_name = f"{model.parameterized_name}_{quant.name}_{'prefill' if do_prefill else 'decode'}_{arch}"
            dump_path = f"{out_path}/{experiment_name}"
            pickle_filename = f"{dump_path}/cmes.pickle"
            with open(pickle_filename, "rb") as fp:
                cmes: list[CME_T] = pickle.load(fp)

            cmes = get_cmes_to_plot(cmes)
            cmes_per_arch.append((cmes))

        plot_energy_compare_archs(
            cmes_per_arch[0], cmes_per_arch[1], title=model.name, filename=f"{out_path}/compare_{model.name}.png"
        )
