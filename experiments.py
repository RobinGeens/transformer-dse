import itertools
import json
import os
import pickle
from zigzag import api
from zigzag.opt.loma.LomaEngine import NoValidLoopOrderingFoundException
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

from export_onnx import export_transformer_to_onnx
from src.config import ALL_MODELS, BATCH_SIZE, W4A16, W8A8
from src.util import accelerator_path, generalize_layer_name, get_cmes_full_model, get_cmes_to_plot

models = ALL_MODELS
quants = [W8A8, W4A16]
accelerators = ["generic_array", "generic_array_edge"]
batch_sizes = [BATCH_SIZE]

mapping_path = "inputs/mapping/output_st_256.yaml"


if __name__ == "__main__":
    for model, accelerator, quant, batch_size in itertools.product(models, accelerators, quants, batch_sizes):

        # Overwrite batch size for the experiment
        model.batch_size = batch_size

        experiment_name = f"{model.parameterized_name}_{quant.name}_{accelerator}"
        dump_path = f"outputs/experiments/{experiment_name}"
        onnx_path = f"outputs/onnx/{model.parameterized_name}_{quant.name}.onnx"
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
            # result_dump_dict = {
            #     generalize_layer_name(cme.layer.name): cme.__simplejsonrepr__() for cme in complete_result_cmes
            # }
            # with open(f"{dump_path}/full_model_result.json", "w") as f:
            #     json.dump(result_dump_dict, f, indent=4)

            # Save which layers are plotted
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
