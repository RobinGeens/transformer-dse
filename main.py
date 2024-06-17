from datetime import datetime
import os
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from export_onnx import export_transformer_to_onnx
from src.config import GPT3_175B, LLAMA_1_7B, W16A32, W4A16, W4A8, W8A8

model = LLAMA_1_7B
quant = W4A16
output_dir = f"/outputs/{datetime.now()}"
workload_path = f"outputs/onnx/{model.name}_{quant.name}.onnx"
# workload_path = "inputs/workload/matmul.yaml"
accelerator_path = "inputs/hardware/generic_array.yaml"
mapping_path = "inputs/mapping/output_st_256.yaml"
pickle_filename = "outputs/TPU-cmes.pickle"
RE_RUN = True

if RE_RUN:
    if not os.path.exists(workload_path):
        export_transformer_to_onnx(model.to_simulatable_config(), quant, path=workload_path)
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

bar_plot_cost_model_evaluations_breakdown(cmes, save_path="plot_breakdown_all.png")
bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path="plot_breakdown.png")

print("Layers currently shown in plot:")
for idx, cme in enumerate(cmes_to_plot):
    print(f"\t{idx}: {cme.layer.name}")
