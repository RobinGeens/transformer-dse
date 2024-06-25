import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from src.util import ARRAY_T, CME_T, GROUPS, group_results, group_results_single_bar


def plot_energy_clean(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    bars = ["MAC", "RF", "SRAM", "DRAM"]
    sections = ["MAC", "weight", "act", "act2", "output"]
    non_weight_layers = [1, 2]  # Indices in `LAYERS_TO_PLOT`

    def cme_to_array_per_mem(cme: CME_T, is_weight_layer: bool = True):
        """Energy per memory, per operand"""
        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = np.zeros((len(bars), len(sections)))
        result[0] = [data["operational_energy"]] + (len(sections) - 1) * [0]
        for mem_level, _ in enumerate(bars[1:]):
            energy_per_op = [data["memory_energy_breakdown_per_level"][op][mem_level] for op in operands]
            if is_weight_layer:
                # Put the energy for `W` at label `weight`, set `act2` to 0
                result[mem_level + 1] = [0] + energy_per_op[:2] + [0, energy_per_op[2]]
            else:
                # Put the energy for `W` at label `act2`, set `weight` to 0
                result[mem_level + 1] = [0, 0, energy_per_op[1], energy_per_op[0], energy_per_op[2]]

        return result

    data = group_results(
        [cme_to_array_per_mem(cme, is_weight_layer=idx not in non_weight_layers) for idx, cme in enumerate(cmes)]
    )
    p = BarPlotter(GROUPS, bars, sections, legend_cols=4, ylabel="Energy (pJ)")
    p.plot(data, filename)


def plot_energy_minimal(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

    bars = [""]
    sections = ["MAC", "RF", "SRAM", "DRAM"]

    def cme_to_array(cme: CME_T) -> ARRAY_T:
        """Energy per memory, per operand"""
        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = [data["operational_energy"]]
        result += [
            np.sum([data["memory_energy_breakdown_per_level"][op][mem_level] for op in operands])
            for mem_level in range(3)
        ]
        return np.array([result])

    data = group_results([cme_to_array(cme) for cme in cmes])
    p = BarPlotter(GROUPS, bars, sections, legend_cols=4, ylabel="Energy (pJ)", group_name_offset=0, group_name_dy=-1)
    p.plot(data, filename)


def plot_latency_clean(cmes: list[CME_T], filename: str = "plot.png"):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert len(cmes) == 5, "Are these the CMEs from `LAYERS_TO_PLOT`?"

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

    data = group_results([cme_to_array(cme) for cme in cmes])
    p = BarPlotter(
        GROUPS,
        bars,
        sections,
        bar_width=1,
        bar_spacing=0,
        group_spacing=0.8,
        group_name_offset=0,
        group_name_dy=-1,
        ylabel="Latency (cycles)",
    )
    p.plot(data, filename)


def plot_energy_compare(
    cmes_all: list[list[CME_T]], supergroups: list[str], title: str = "", filename: str = "plot.png"
):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(supergroups)

    groups = [f"{group}\n{supergroup}" for supergroup, group in itertools.product(supergroups, GROUPS)]
    bars = ["MAC", "RF", "SRAM", "DRAM"]
    sections = ["MAC", "weight", "act", "act2", "output"]
    non_weight_layers = [1, 2]  # Indices in `LAYERS_TO_PLOT`

    def cme_to_array_per_mem(cme: CME_T, is_weight_layer: bool = True):
        """Energy per memory, per operand"""
        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = np.zeros((len(bars), len(sections)))
        result[0] = [data["operational_energy"]] + (len(sections) - 1) * [0]
        for mem_level, _ in enumerate(bars[1:]):
            energy_per_op = [data["memory_energy_breakdown_per_level"][op][mem_level] for op in operands]
            if is_weight_layer:
                # Put the energy for `W` at label `weight`, set `act2` to 0
                result[mem_level + 1] = [0] + energy_per_op[:2] + [0, energy_per_op[2]]
            else:
                # Put the energy for `W` at label `act2`, set `weight` to 0
                result[mem_level + 1] = [0, 0, energy_per_op[1], energy_per_op[0], energy_per_op[2]]

        return result

    data: ARRAY_T = np.zeros((0, len(bars), len(sections)))
    for cmes_per_supergroup in cmes_all:
        data_per_supergroup = group_results(
            [
                cme_to_array_per_mem(cme, is_weight_layer=idx not in non_weight_layers)
                for idx, cme in enumerate(cmes_per_supergroup)
            ]
        )
        data = np.concatenate([data, data_per_supergroup], axis=0)

    assert data.shape == (len(supergroups) * len(GROUPS), len(bars), len(sections))

    p = BarPlotter(groups, bars, sections, legend_cols=5, ylabel="Energy (pJ)")
    p.plot(data, filename)


def plot_latency_compare(
    cmes_all: list[list[CME_T]], supergroups: list[str], title: str = "", filename: str = "plot.png"
):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(supergroups)

    groups = [f"{group}\n{supergroup}" for supergroup, group in itertools.product(supergroups, GROUPS)]
    bars = [""]
    sections = ["Ideal computation", "Spatial underutilization", "Memory stall"]

    def cme_to_array(cme: CME_T):
        """Latency per category.
        Shape = (len(bars), len(sections))"""
        # Hard-copied from zigzag `plot_cme`
        result = np.array(
            [
                [
                    cme.ideal_cycle,  # Ideal computation
                    (cme.ideal_temporal_cycle - cme.ideal_cycle),  # Spatial stall
                    (cme.latency_total0 - cme.ideal_temporal_cycle)  # Temporal stall
                    + (cme.latency_total1 - cme.latency_total0)  # Data loading
                    + (cme.latency_total2 - cme.latency_total1),  # Data off-loading
                ]
            ]
        )
        return result

    data: ARRAY_T = np.zeros((0, len(bars), len(sections)))
    for cmes_per_supergroup in cmes_all:
        data_per_supergroup = group_results([cme_to_array(cme) for cme in cmes_per_supergroup])
        data = np.concatenate([data, data_per_supergroup], axis=0)

    assert data.shape == (len(supergroups) * len(GROUPS), len(bars), len(sections))

    p = BarPlotter(
        groups,
        bars,
        sections,
        bar_width=1,
        bar_spacing=0,
        group_spacing=0.8,
        group_name_offset=0,
        group_name_dy=-1,
        ylabel="Latency (cycles)",
    )
    p.plot(data, filename)


def plot_energy_compare_minimal(
    cmes_all: list[list[CME_T]], groups: list[str], title: str = "", filename: str = "plot.png"
):
    """`cmes` correspond to `LAYERS_TO_PLOT`"""
    assert all([len(cmes) == 5 for cmes in cmes_all]), "Are these the CMEs from `LAYERS_TO_PLOT`?"
    assert len(cmes_all) == len(groups)

    bars = GROUPS
    sections = ["MAC", "RF", "SRAM", "DRAM"]

    def cme_to_array_single_bar(cme: CME_T) -> ARRAY_T:
        """Energy per memory"""
        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = [data["operational_energy"]]
        result += [
            np.sum([data["memory_energy_breakdown_per_level"][op][mem_level] for op in operands])
            for mem_level in range(3)
        ]
        return np.array(result)

    def cmes_to_array_single_group(cmes: list[CME_T]) -> ARRAY_T:
        return group_results_single_bar([cme_to_array_single_bar(cme) for cme in cmes])

    data = np.array([cmes_to_array_single_group(cmes_single_group) for cmes_single_group in cmes_all])

    p = BarPlotter(
        groups,
        bars,
        sections,
        # bar_width = 1,
        # bar_spacing = 0,
        group_spacing=0.5,
        # group_name_offset = 0,
        # xtick_rotation=0,
        # xtick_ha="center",
        legend_cols=4,
        # group_name_dy = -1,
        ylabel="Energy (pJ)",
        title=title,
    )
    p.plot(data, filename)


class BarPlotter:
    def __init__(
        self,
        groups: list[str],
        bars: list[str],
        sections: list[str],
        *,
        sections_alt: list[str] | None = None,
        sections_alt_idx: list[int] | None = None,
        # Layout
        bar_width: float = 0.6,
        bar_spacing: float = 0.2,
        group_spacing: float = 1.2,
        group_name_dy: float = -4,
        group_name_offset: float | None = None,
        # Labels
        xtick_rotation: int = 45,
        xtick_ha: str = "right",
        ylabel: str = "",
        title: str = "",
        legend_cols: int = 1,
        # Other
        colors: None = None,
    ):
        assert sections_alt is None or len(sections_alt) == len(sections)
        self.groups = groups
        self.bars = bars
        self.sections = sections
        self.sections_alt = sections_alt
        self.sections_alt_idx = sections_alt_idx

        # Layout
        self.bar_width = bar_width
        self.bar_spacing = bar_spacing
        self.group_spacing = group_spacing
        self.group_name_dy = group_name_dy
        self.group_name_offset = (
            (len(self.bars) * (self.bar_width + self.bar_spacing)) * 0.4
            if group_name_offset is None
            else group_name_offset
        )

        # Labels
        self.xtick_rotation = xtick_rotation
        self.xtick_ha = xtick_ha
        # Offset from bar center
        self.xtick_offset = self.bar_width / 2 if xtick_ha == "right" else 0
        self.ylabel = ylabel
        self.title = title
        self.legend_cols = legend_cols

        # Other
        colors_default = seaborn.color_palette("pastel", len(self.sections))
        colors_default = colors_default[2:] + colors_default[:2]  # Because green is at idx 2
        self.colors = colors_default if colors is None else colors

    def plot(self, data: ARRAY_T, filename: str) -> None:
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

        # Bar names (as xticks)
        xtick_labels: list[str] = []
        for idx, _ in enumerate(self.groups):
            xtick_labels += self.bars
        xticks_positions = [
            indices[i] + j * (self.bar_width + self.bar_spacing) + self.xtick_offset
            for i in range(len(self.groups))
            for j in range(len(self.bars))
        ]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xtick_labels, fontsize=14, ha=self.xtick_ha)
        plt.xticks(rotation=self.xtick_rotation)

        # Group names
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

        # Add labels and title
        # ax.set_xlabel("Module", fontsize=16)
        ax.set_ylabel(self.ylabel, fontsize=16)
        ax.set_title(self.title, fontsize=16)
        ax.legend(ncol=self.legend_cols, fontsize=14)  # loc="upper center"

        plt.tight_layout()
        plt.savefig(filename, transparent=False)
