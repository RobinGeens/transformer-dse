from datetime import datetime
import os
import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from export_onnx import export_transformer_to_onnx
from src.config import GPT3_175B, LLAMA_1_7B, OPT_125M, W16A32, W4A16, W4A8, W8A8
from src.util import clean_zigzag_plot, get_cmes_full_model, get_cmes_to_plot

model = OPT_125M
quant = W4A16
do_prefill = False
output_dir = f"outputs/{datetime.now()}"
workload_path = f"outputs/onnx/{model.name}_{quant.name}_{'prefill' if do_prefill else 'decode'}.onnx"
# workload_path = "inputs/workload/matmul.yaml"
accelerator_path = "inputs/hardware/generic_array.yaml"
mapping_path = "inputs/mapping/weight_st_256.yaml"
pickle_filename = "outputs/TPU-cmes.pickle"


if not os.path.exists(workload_path):
    export_transformer_to_onnx(model.to_simulatable_config(), quant, path=workload_path, prefill=do_prefill)

energy, latency, cmes = api.get_hardware_performance_zigzag(
    workload=workload_path,
    accelerator=accelerator_path,
    mapping=mapping_path,
    opt="energy",
    dump_folder=output_dir,
    pickle_filename=pickle_filename,
    nb_spatial_mappings_generated=1,
)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")

with open(pickle_filename, "rb") as fp:
    cmes = pickle.load(fp)

cmes_to_plot = get_cmes_to_plot(cmes)

bar_plot_cost_model_evaluations_breakdown(cmes, save_path=f"{output_dir}/all_layers.png")
bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path=f"{output_dir}/interesting_layers_single.png")

# Compute generalized results for full LLM
complete_result_cmes = get_cmes_full_model(cmes_to_plot, model)
bar_plot_cost_model_evaluations_breakdown(complete_result_cmes, save_path=f"{output_dir}/interesting_layers_full.png")

print("Layers currently shown in plot:")
for idx, cme in enumerate(cmes_to_plot):
    print(f"\t{idx}: {cme.layer.name}")


clean_zigzag_plot(complete_result_cmes, "temp.png")
