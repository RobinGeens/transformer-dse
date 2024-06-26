import itertools
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from src.util import ARRAY_T, CME_T, GROUPS, LAYERS_TO_PLOT


def group_results(data: list[ARRAY_T]) -> ARRAY_T:
    """Here, we group the data of the CMEs in `LAYERS_TO_PLOT` to `GROUPS`
    Shape in: (5, len(bars), len(sections))
    Shape out: (3, len(bars), len(sections))"""
    assert len(data) == len(LAYERS_TO_PLOT)
    assert len(data[0].shape) == 2
    return np.array([data[0], data[1] + data[2], data[3] + data[4]])


def group_results_single_bar(data: list[ARRAY_T]) -> ARRAY_T:
    """Here, we group the data of the CMEs in `LAYERS_TO_PLOT` to `GROUPS`
    Shape in: (5, len(sections))
    Shape out: (3, len(sections))"""
    assert len(data) == len(LAYERS_TO_PLOT)
    assert len(data[0].shape) == 1
    return np.array([data[0], data[1] + data[2], data[3] + data[4]])


class PlotCMEMinimal:
    bars = GROUPS
    energy_sections = ["MAC", "RF", "SRAM", "DRAM"]

    @staticmethod
    def cme_to_energy_array_single_bar(cme: CME_T) -> ARRAY_T:
        """Energy per memory, summed up for all operands"""
        operands = ["W", "I", "O"]
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = [data["operational_energy"]]
        result += [
            np.sum([PlotCMEDetailed.get_mem_energy(data, op, mem_level) for op in operands]) for mem_level in range(3)
        ]
        return np.array(result)

    @staticmethod
    def cmes_to_energy_array_single_group(cmes: list[CME_T]) -> ARRAY_T:
        return group_results_single_bar([PlotCMEMinimal.cme_to_energy_array_single_bar(cme) for cme in cmes])

    @staticmethod
    def cmes_to_array_single_group(cmes: list[CME_T]) -> ARRAY_T:
        return group_results_single_bar([PlotCMEDetailed.cme_to_latency_array_single_bar(cme) for cme in cmes])


class PlotCMEDetailed:
    energy_bars = ["MAC", "RF", "SRAM", "DRAM"]
    energy_sections = ["MAC", "weight", "act", "act2", "output"]
    non_weight_layers = [1, 2]  # Indices in `LAYERS_TO_PLOT`

    latency_sections = ["Ideal computation", "Spatial underutilization", "Memory stall"]

    @staticmethod
    def get_mem_energy(data: Any, op: str, mem_level: int):
        # There should be 3 mem levels. Insert 0 at lowest level otherwise
        energy_per_level = data["memory_energy_breakdown_per_level"][op]
        if len(energy_per_level) == 2:
            energy_per_level = [0] + energy_per_level
        return energy_per_level[mem_level]

    @staticmethod
    def cme_to_energy_array_single_group(cme: CME_T, is_weight_layer: bool = True):
        """Energy per memory, per operand. This will return a single group"""

        operands = ["W", "I", "O"]  # Same order as `sections`
        data = cme.__jsonrepr__()["outputs"]["energy"]
        result = np.zeros((len(PlotCMEDetailed.energy_bars), len(PlotCMEDetailed.energy_sections)))
        result[0] = [data["operational_energy"]] + (len(PlotCMEDetailed.energy_sections) - 1) * [0]
        for mem_level, _ in enumerate(PlotCMEDetailed.energy_bars[1:]):
            energy_per_op = [PlotCMEDetailed.get_mem_energy(data, op, mem_level) for op in operands]
            if is_weight_layer:
                # Put the energy for `W` at label `weight`, set `act2` to 0
                result[mem_level + 1] = [0] + energy_per_op[:2] + [0, energy_per_op[2]]
            else:
                # Put the energy for `W` at label `act2`, set `weight` to 0
                result[mem_level + 1] = [0, 0, energy_per_op[1], energy_per_op[0], energy_per_op[2]]

        return result

    @staticmethod
    def cmes_to_energy_array_all(cmes: list[CME_T]):
        return group_results(
            [
                PlotCMEDetailed.cme_to_energy_array_single_group(
                    cme, is_weight_layer=idx not in PlotCMEDetailed.non_weight_layers
                )
                for idx, cme in enumerate(cmes)
            ]
        )

    @staticmethod
    def cme_to_latency_array_single_bar(cme: CME_T):
        """Latency per category.
        Shape = (len(sections))"""
        # Hard-copied from zigzag `plot_cme`
        result = np.array(
            [
                cme.ideal_cycle,  # Ideal computation
                (cme.ideal_temporal_cycle - cme.ideal_cycle),  # Spatial stall
                (cme.latency_total0 - cme.ideal_temporal_cycle)  # Temporal stall
                + (cme.latency_total1 - cme.latency_total0)  # Data loading
                + (cme.latency_total2 - cme.latency_total1),  # Data off-loading
            ]
        )
        return result

    @staticmethod
    def cme_to_latency_array_single_group(cme: CME_T):
        # Single bar per group
        return np.array([PlotCMEDetailed.cme_to_latency_array_single_bar(cme)])

    @staticmethod
    def cme_to_latency_array_all_single_group(cmes: list[CME_T]):
        assert len(cmes) == len(LAYERS_TO_PLOT)
        # All CMEs in single group
        return group_results_single_bar([PlotCMEDetailed.cme_to_latency_array_single_bar(cme) for cme in cmes])

    @staticmethod
    def cme_to_latency_array_all(cmes: list[CME_T]):
        return group_results([PlotCMEDetailed.cme_to_latency_array_single_group(cme) for cme in cmes])


class BarPlotter:
    def __init__(
        self,
        groups: list[str],
        bars: list[str],
        sections: list[str],
        *,
        # sections_alt: list[str] | None = None,
        # sections_alt_idx: list[int] | None = None,
        # Layout
        bar_width: float = 0.6,
        bar_spacing: float = 0.1,
        group_spacing: float = 1,
        group_name_dy: float = -4,
        group_name_offset: float | None = None,
        scale: str = "linear",
        # Labels
        xtick_labels: list[str] | None = None,
        xtick_rotation: int = 45,
        xtick_fontsize: int = 14,
        xtick_ha: str = "right",
        ylabel: str = "",
        title: str = "",
        legend_cols: int = 1,
        # Other
        colors: None = None,
    ):
        # assert sections_alt is None or len(sections_alt) == len(sections)
        assert xtick_labels is None or len(xtick_labels) == len(groups) * len(bars)
        self.groups = groups
        self.bars = bars
        self.sections = sections
        # self.sections_alt = sections_alt
        # self.sections_alt_idx = sections_alt_idx

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
        self.scale = scale

        # Labels
        self.xtick_labels = xtick_labels if xtick_labels is not None else len(groups) * bars
        self.xtick_rotation = xtick_rotation
        self.xtick_fontsize = xtick_fontsize
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

    def construct_subplot(self, ax: Any, data: ARRAY_T):
        assert data.shape == (len(self.groups), len(self.bars), len(self.sections))

        indices = np.arange(len(self.groups)) * (
            len(self.bars) * (self.bar_width + self.bar_spacing) + self.group_spacing
        )
        group_name_positions = indices + self.group_name_offset

        # Make bars
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
        xticks_positions = [
            indices[i] + j * (self.bar_width + self.bar_spacing) + self.xtick_offset
            for i in range(len(self.groups))
            for j in range(len(self.bars))
        ]
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(
            self.xtick_labels, fontsize=self.xtick_fontsize, ha=self.xtick_ha, rotation=self.xtick_rotation
        )

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

    def plot(self, data: ARRAY_T, filename: str) -> None:

        _, ax = plt.subplots(figsize=(12, 6))
        self.construct_subplot(ax, data)
        plt.rc("font", family="DejaVu Serif")
        plt.style.use("ggplot")

        # plt.xticks(rotation=self.xtick_rotation)
        plt.yscale(self.scale)

        plt.tight_layout()
        plt.savefig(filename, transparent=False)


class BarPlotterSubfigures:
    def __init__(
        self,
        bar_plotters: list[BarPlotter],
        *,
        subplot_rows: int = 1,
        subplot_cols: int = 1,
        width_ratios: list[int | float] | None = None,
        title: str = "",
    ):
        self.nb_plots = subplot_rows * subplot_cols
        assert width_ratios is None or len(width_ratios) == subplot_cols
        assert len(bar_plotters) == self.nb_plots
        self.bar_plotters = bar_plotters
        self.subplot_rows = subplot_rows
        self.subplot_cols = subplot_cols
        self.width_ratios = width_ratios if width_ratios is not None else subplot_cols * [1]
        self.title = title

    def plot(self, data: list[ARRAY_T], filename: str) -> None:
        assert len(data) == self.nb_plots

        fig, axises = plt.subplots(
            nrows=self.subplot_rows, ncols=self.subplot_cols, width_ratios=self.width_ratios, figsize=(12, 6)
        )

        for ax, data_subplot, plotter in zip(axises, data, self.bar_plotters):
            plotter.construct_subplot(ax, data_subplot)

        fig.suptitle(self.title)
        plt.rc("font", family="DejaVu Serif")
        plt.style.use("ggplot")
        plt.tight_layout()
        plt.savefig(filename, transparent=False)
