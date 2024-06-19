import os
from itertools import product
import pickle
import matplotlib.pyplot as plt
import numpy as np

# import seaborn
from zigzag import api

from src.config import GPT3_175B, LLAMA_1_7B, W4A16
from export_onnx import export_transformer_to_onnx
from src.util import CME_T, LAYERS_TO_PLOT, accelerator_path, get_cmes_full_model, get_cmes_to_plot

# Plot info
groups = ["Linear projection", "Q*K^T", "S*V", "MLP 1", "MLP 2"]
bars = ["MAC", "weight (INT4)", "act (FP16)", "output (FP32)"]
bars_alt = ["MAC", "act1 (FP16)", "act2 (FP16)", "output (FP32)"]
bars_alt_idx = [1, 2]  # Use alt bar names for bars at idx 1 and 2
sections = ["MAC", "RF", "SRAM", "DRAM"]
colors = ["#feb29b", "#bed2c6", "#ffd19d", "#ed8687"]
# colors = seaborn.color_palette("Set2", len(sections))


# Experiment info
models = [LLAMA_1_7B, GPT3_175B]
quant = W4A16
accelerators = ["generic_array", "generic_array_edge"]
mapping_path = "inputs/mapping/output_st_256.yaml"


def cme_to_array_per_operand(cme: CME_T):
    # This will give the same data as in the json output dumps
    data = cme.__jsonrepr__()["outputs"]["energy"]

    result = np.zeros((len(bars), len(sections)))
    result[0] = [data["operational_energy"]] + (len(sections) - 1) * [0]
    for idx, op in enumerate(["W", "I", "O"]):
        energy_per_level = [
            np.sum(list(energy_per_dir.values()))
            for energy_per_dir in data["memory_energy_breakdown_per_level_per_operand"][op]
        ]
        result[idx + 1] = [0] + energy_per_level
    return result


def cmes_to_array(cmes: list[CME_T]):
    assert len(cmes) == len(groups)
    return np.array([cme_to_array_per_operand(cme) for cme in cmes])


def make_plot(cmes: list[CME_T], filename: str, title: str):
    data = cmes_to_array(cmes)

    _, ax = plt.subplots(figsize=(12, 6))
    plt.rc("font", family="DejaVu Serif")
    plt.style.use("ggplot")
    bar_width = 0.6
    bar_spacing = 0.2
    group_spacing = 1.2

    indices = np.arange(len(groups)) * (len(bars) * bar_width + bar_spacing + group_spacing)

    for i, _ in enumerate(bars):
        bottom = np.zeros(len(groups))
        for j, section in enumerate(sections):
            positions = indices + i * (bar_width + bar_spacing)
            heights = data[:, i, j]
            ax.bar(
                positions,
                heights,
                bar_width,
                bottom=bottom,
                label=f"{section}" if i == 0 else "",
                color=colors[j],
                edgecolor="black",
            )
            bottom += heights

    # Add group names
    for i, group in enumerate(groups):
        position = indices[i] + (len(bars) * (bar_width + bar_spacing)) * 0.4
        # plt.text(position, -1e12, group, ha="center", va="top", fontsize=16, rotation=0)
        ax.annotate(
            group,
            xy=(position, 0),  # Reference in coordinate system
            xycoords="data",  # Use coordinate system of data points
            xytext=(0, -6.5),  # Offset from reference
            textcoords="offset fontsize",  # Offset value is relative to fontsize
            ha="center",
            va="top",
            weight="normal" if i in bars_alt_idx else "bold",
            fontsize=14,
            rotation=0,
        )

    # Set operand names
    xtick_labels: list[str] = []
    for idx, _ in enumerate(groups):
        xtick_labels += bars_alt if idx in bars_alt_idx else bars
    xticks_positions = [
        indices[i] + j * (bar_width + bar_spacing) + bar_width / 2 for i in range(len(groups)) for j in range(len(bars))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xtick_labels, fontsize=14, ha="right")

    # Add labels and title
    # ax.set_xlabel("Module", fontsize=16)
    ax.set_ylabel("Energy [pJ]", fontsize=16)
    ax.set_title(f"Energy distribution of {title}", fontsize=16)

    ax.legend(loc="upper center", ncol=4, fontsize=14)  # bbox_to_anchor=(0.5, 1.12),

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"outputs/anda/{filename}.png")


if __name__ == "__main__":
    for model, accelerator in product(models, accelerators):
        experiment_name = f"{model.parameterized_name}_{accelerator}"
        dump_path = f"outputs/anda/{experiment_name}"
        onnx_path = f"outputs/onnx/{model.parameterized_name}_{quant.name}.onnx"
        pickle_filename = f"{dump_path}/cmes.pickle"

        if not os.path.exists(onnx_path):
            export_transformer_to_onnx(model.to_simulatable_config(), quant, path=onnx_path)

        # api.get_hardware_performance_zigzag(
        #     workload=onnx_path,
        #     accelerator=accelerator_path(accelerator),
        #     mapping=mapping_path,
        #     opt="energy",
        #     dump_folder=dump_path,
        #     pickle_filename=pickle_filename,
        #     nb_spatial_mappings_generated=3,
        # )

        with open(pickle_filename, "rb") as fp:
            cmes = pickle.load(fp)

        cmes_to_plot = get_cmes_to_plot(cmes)
        complete_result_cmes = get_cmes_full_model(cmes_to_plot, model)

        match accelerator:
            case "generic_array":
                accel_name = "Cloud architecture"
            case "generic_array_edge":
                accel_name = "Edge architecture"
            case _:
                raise ValueError
        make_plot(complete_result_cmes, filename=experiment_name, title=f"{model.name} on {accel_name}")
