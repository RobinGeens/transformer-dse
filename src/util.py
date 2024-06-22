from typing import TypeAlias
from typing import Any, TypeVar
import numpy as np


from src.config import LLMConfig

LAYERS_TO_PLOT = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]
GROUPS = ["Linear projection", "Attention", "FFN"]

CME_T = TypeVar("CME_T", Any, Any)  # CME type not available here
ARRAY_T: TypeAlias = np.ndarray[Any, Any]


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
