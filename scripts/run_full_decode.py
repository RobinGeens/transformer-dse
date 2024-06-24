"""
Simulate the decoding phase by evaluating each token separately in the sequence from L/2 to L.
"""

import json
import sys
import imageio
import os
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)

sys.path.append(os.getcwd())
from src.export_onnx import export_transformer_to_onnx
from src.config import GPT3_175B, LLAMA_1_7B, OPT_125M, W16A32, W4A16, W4A8, W8A8
from src.util import get_cmes_full_model, get_cmes_to_plot

model = LLAMA_1_7B
quant = W8A8
# workload_path = "inputs/workload/matmul.yaml"
accelerator_path = "inputs/hardware/generic_array.yaml"
mapping_path = "inputs/mapping/weight_st_256.yaml"
pickle_filename = "outputs/TPU-cmes.pickle"


def run_experiment():
    for decode_idx in range(model.seq_len // 2 + 1, model.seq_len):
        # Overwrite decode_idx
        model.decode_idx = decode_idx

        output_dir = f"outputs/full_decode/{model.name}_{quant.name}_decode={decode_idx}"
        workload_path = f"outputs/full_decode/onnx/{model.name}_{quant.name}_decode={decode_idx}.onnx"

        try:
            if not os.path.exists(workload_path):
                export_transformer_to_onnx(model.to_simulatable_config(), quant, path=workload_path, prefill=False)

            _, _, cmes = api.get_hardware_performance_zigzag(
                workload=workload_path,
                accelerator=accelerator_path,
                mapping=mapping_path,
                opt="energy",
                dump_folder=output_dir,
                pickle_filename=pickle_filename,
                nb_spatial_mappings_generated=1,
            )

            with open(pickle_filename, "rb") as fp:
                cmes = pickle.load(fp)

            cmes_to_plot = get_cmes_to_plot(cmes)

            bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{output_dir}/all_layers.png")
            bar_plot_cost_model_evaluations_breakdown(
                cmes_to_plot, save_path=f"{output_dir}/interesting_layers_single.png"
            )

            # Compute generalized results for full LLM
            complete_result_cmes = get_cmes_full_model(cmes_to_plot, model)
            bar_plot_cost_model_evaluations_breakdown(
                complete_result_cmes, save_path=f"{output_dir}/interesting_layers_full.png"
            )

        except:
            continue


def make_gif():

    image_list = [
        f"outputs/full_decode/{model.name}_{quant.name}_decode={idx}/interesting_layers_full.png"
        for idx in range(1025, 2048)
    ]

    output_gif = "outputs/decoding.gif"

    images = [imageio.imread(file) for file in image_list]
    gif_duration = 10  # in seconds
    time_per_frame = gif_duration / len(image_list)
    # duration is the time between frames in seconds
    imageio.mimsave(output_gif, images, duration=time_per_frame)


if __name__ == "__main__":
    # run_experiment()

    result_list = [
        f"outputs/full_decode/{model.name}_{quant.name}_decode={idx}/overall_simple.json" for idx in range(1025, 2048)
    ]

    energy_list = [json.load(open(file))["energy"] for file in result_list]

    total_energy = sum(energy_list)
    total_energy_approx = energy_list[len(energy_list) // 2] * len(energy_list)

    print(f"Total energy {total_energy:.6e} approximated by {total_energy_approx:.6e}")
    print(f"Relative error: {(abs(total_energy - total_energy_approx) / total_energy)}")
