from typing import Any, TypeVar
import matplotlib.pyplot as plt
import numpy as np
import seaborn


from src.config import LLMConfig

CME_T = TypeVar("CME_T", Any, Any)  # CME type not available here
LAYERS_TO_PLOT = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]
# LAYERS_TO_PLOT_GROUPED = [("key_proj",), ("mul_qk_t", "mul_logits"), ("feedforward_expand", "feedforward_contract")]


def accelerator_path(accelerator: str):
    return f"inputs/hardware/{accelerator}.yaml"


def generalize_layer_name(layer: str):
    """Give the layer name a prettier format, and generalize single layers to full LLM. e.g. key projection -> all
    linear projections"""
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


def get_cmes_to_plot(cmes: list[CME_T]):
    """Return CMEs in order of `LAYERS_TO_PLOT"""
    result: list[CME_T] = []
    for name in LAYERS_TO_PLOT:
        cme = next(filter(lambda x: name in x.layer.name, cmes), None)
        if cme is not None:
            result.append(cme)
    return result


def get_cmes_full_model(cmes: list[CME_T], model: LLMConfig):
    """Generalize the zigzag results (for single layers) to a full LLM"""
    return [cme * model.get_post_simulation_factor(cme.layer.name) for cme in cmes]


def clean_zigzag_plot(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    groups = ["Linear projection", "Attention", "FFN"]
    bars = ["MAC", "RF", "SRAM", "DRAM"]
    sections = ["MAC", "weight", "act", "output"]
    colors = seaborn.color_palette("pastel", len(sections))

    def cme_to_array_per_mem(cme: CME_T):
        """Energy per memory, per operand"""
        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = np.zeros((len(bars), len(sections)))
        result[0] = [data["operational_energy"]] + (len(sections) - 1) * [0]
        for mem_level, _ in enumerate(bars[1:]):
            energy_per_op = [data["memory_energy_breakdown_per_level"][op][mem_level] for op in operands]
            result[mem_level + 1] = [0] + energy_per_op
        return result

    def cmes_to_array_grouped(cmes: list[CME_T]):
        """Here, we group the `LAYERS_TO_PLOT` to `groups`"""
        assert len(cmes) == len(LAYERS_TO_PLOT)
        f = lambda x: cme_to_array_per_mem(x)
        return np.array([f(cmes[0]), f(cmes[1]) + f(cmes[2]), f(cmes[3]) + f(cmes[4])])

    data = cmes_to_array_grouped(cmes)

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
            weight="normal",
            fontsize=14,
            rotation=0,
        )

    # Set operand names
    xtick_labels: list[str] = []
    for idx, _ in enumerate(groups):
        xtick_labels += bars
    xticks_positions = [
        indices[i] + j * (bar_width + bar_spacing) + bar_width / 2 for i in range(len(groups)) for j in range(len(bars))
    ]
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(xtick_labels, fontsize=14, ha="right")

    # Add labels and title
    # ax.set_xlabel("Module", fontsize=16)
    ax.set_ylabel("Energy [pJ]", fontsize=16)
    # ax.set_title(f"Energy distribution of {'placeholder'}", fontsize=16)

    ax.legend(loc="upper center", ncol=4, fontsize=14)  # bbox_to_anchor=(0.5, 1.12),

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
