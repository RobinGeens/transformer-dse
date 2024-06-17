from datetime import datetime
import os
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from export_onnx import export_transformer_to_onnx
from src.config import LLAMA_1_7B, W4A16

model = LLAMA_1_7B
quant = W4A16
output_dir = f"/outputs/{datetime.now()}"
workload_path = "inputs/workload/matmul_bit_unrolling.yaml"
accelerator_path = "inputs/hardware/generic_array_bit_unrolling.yaml"
mapping_path = "inputs/mapping/output_st_256_bit_unrolling.yaml"
pickle_filename = "outputs/TPU-cmes.pickle"

energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="energy",
    pickle_filename=pickle_filename,
    nb_spatial_mappings_generated=1,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)

layers_to_plot = ["key_proj", "mul_qk_t", "mul_logits", "feedforward_expand", "feedforward_contract"]
cmes_to_plot = [next(filter(lambda x: name in x.layer.name, cmes), None) for name in layers_to_plot]

bar_plot_cost_model_evaluations_breakdown(cmes, save_path="all_layers.png")
bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path="interesting_layers_single.png")

# Compute generalized results for full LLM
complete_result_cmes = [cme * model.get_post_simulation_factor(cme.layer.name) for cme in cmes_to_plot]
bar_plot_cost_model_evaluations_breakdown(complete_result_cmes, save_path="interesting_layers_full.png")

print("Layers currently shown in plot:")
for idx, cme in enumerate(cmes_to_plot):
    print(f"\t{idx}: {cme.layer.name}")
