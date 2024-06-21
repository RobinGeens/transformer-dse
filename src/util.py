from typing import Any, TypeVar
import matplotlib.pyplot as plt
import numpy as np
import seaborn


from src.config import LLMConfig

CME_T = TypeVar("CME_T", Any, Any)  # CME type not available here
LAYERS_TO_PLOT = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]


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


def clean_zigzag_plot_energy(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    groups = ["Linear projection", "Attention", "FFN"]
    bars = ["MAC", "RF", "SRAM", "DRAM"]
    sections = ["MAC", "weight", "act", "output"]

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
    p = Plotter(groups, bars, sections)
    p.legend_cols = 4
    p.ylabel = "Energy (pJ)"
    p.plot(data, filename)


def clean_zigzag_plot_latency(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    groups = ["Linear projection", "Attention", "FFN"]
    bars = [""]
    sections = ["Ideal computation", "Spatial stall", "Temporal stall", "Data loading", "Data off-loading"]

    def cme_to_array(cme: CME_T):
        """Latency per category.
        Shape = (len(bars), len(sections))"""
        # Hard-copied from zigzag `plot_cme`
        result = np.array(
            [
                [
                    cme.ideal_cycle,  # Ideal computation
                    cme.ideal_temporal_cycle - cme.ideal_cycle,  # Spatial stall
                    cme.latency_total0 - cme.ideal_temporal_cycle,  # Temporal stall
                    cme.latency_total1 - cme.latency_total0,  # Data loading
                    cme.latency_total2 - cme.latency_total1,  # Data off-loading
                ]
            ]
        )
        # la_tot[idx] = cme.latency_total2
        return result

    def cmes_to_array_grouped(cmes: list[CME_T]):
        """Here, we group the `LAYERS_TO_PLOT` to `groups`
        Shape = (len(groups), len(bars), len(sections))"""
        assert len(cmes) == len(LAYERS_TO_PLOT)
        f = lambda x: cme_to_array(x)
        return np.array([f(cmes[0]), f(cmes[1]) + f(cmes[2]), f(cmes[3]) + f(cmes[4])])

    data = cmes_to_array_grouped(cmes)
    p = Plotter(groups, bars, sections)
    p.bar_width = 1
    p.bar_spacing = 0
    p.group_spacing = 0.8
    p.group_name_offset = 0
    p.group_name_dy = -1
    p.ylabel = "Latency (cycles)"
    p.plot(data, filename)


class Plotter:
    def __init__(self, groups: list[str], bars: list[str], sections: list[str]):
        self.groups = groups
        self.bars = bars
        self.sections = sections

        # Config
        colors = seaborn.color_palette("pastel", len(self.sections))
        self.colors = colors[2:] + colors[:2]  # Because green is at idx 2
        self.bar_width = 0.6
        self.bar_spacing = 0.2
        self.group_spacing = 1.2
        # Group names
        self.group_name_offset = (len(self.bars) * (self.bar_width + self.bar_spacing)) * 0.4
        self.group_name_dy = -4
        # Axis and legend
        self.ylabel = ""
        self.legend_cols = 1

    def plot(self, data: np.ndarray, filename: str) -> None:
        assert data.shape == (len(self.groups), len(self.bars), len(self.sections))

        _, ax = plt.subplots(figsize=(12, 6))
        plt.rc("font", family="DejaVu Serif")
        plt.style.use("ggplot")

        indices = np.arange(len(self.groups)) * (
            len(self.bars) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )
        group_name_positions = indices + self.group_name_offset

        for i, _ in enumerate(self.bars):
            bottom = np.zeros(len(self.groups))
            for j, section in enumerate(self.sections):
                positions = indices + i * (self.bar_width + self.bar_spacing)
                heights = data[:, i, j]
                ax.bar(
                    positions,
                    heights,
                    self.bar_width,
                    bottom=bottom,
                    label=f"{section}" if i == 0 else "",
                    color=self.colors[j],
                    edgecolor="black",
                )
                bottom += heights

        # Add group names
        for i, group in enumerate(self.groups):
            ax.annotate(
                group,
                xy=(group_name_positions[i], 0),  # Reference in coordinate system
                xycoords="data",  # Use coordinate system of data points
                xytext=(0, self.group_name_dy),  # Offset from reference
                textcoords="offset fontsize",  # Offset value is relative to fontsize
                ha="center",
                va="top",
                weight="normal",
                fontsize=14,
                rotation=0,
            )

        # Set operand names
        xtick_labels: list[str] = []
        for idx, _ in enumerate(self.groups):
            xtick_labels += self.bars
        xticks_positions = [
            indices[i] + j * (self.bar_width + self.bar_spacing) + self.bar_width / 2
            for i in range(len(self.groups))
            for j in range(len(self.bars))
        ]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xtick_labels, fontsize=14, ha="right")

        # Add labels and title
        # ax.set_xlabel("Module", fontsize=16)
        ax.set_ylabel(self.ylabel, fontsize=16)
        # ax.set_title(f"Energy distribution of {'placeholder'}", fontsize=16)

        ax.legend(loc="upper center", ncol=self.legend_cols, fontsize=14)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename, transparent=False)
