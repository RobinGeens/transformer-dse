import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph

from config import LLAMA_13B
from export_onnx import export_transformer_to_onnx

model = LLAMA_13B
workload_path = "out/custom_transformer.onnx"
accelerator_path = "inputs/hardware/tpu_like.yaml"
mapping_path = "inputs/mapping/default.yaml"
pickle_filename = "out/TPU-saved_list_of_cmes.pickle"
RE_RUN = True

if RE_RUN:
    export_transformer_to_onnx(model.to_simulatable_config())
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

# TODO no add
layers_to_plot = ["key_proj", "mul_qk_t", "mul_logits", "out_proj", "feedforward_expand", "feedforward_contract"]
cmes_to_plot = [next(filter(lambda x: name in x.layer.name, cmes)) for name in layers_to_plot]

bar_plot_cost_model_evaluations_breakdown(cmes, save_path="plot_breakdown_all.png")
bar_plot_cost_model_evaluations_breakdown(cmes_to_plot, save_path="plot_breakdown.png")

visualize_memory_hierarchy_graph(
    cmes[0].accelerator.cores[0].memory_hierarchy,
    save_path="out/mem_hierarchy.png",
)


print("Layers currently shown in plot:")
for idx, cme in enumerate(cmes_to_plot):
    print(f"Layer{idx}: {cme.layer.name}")
    # print_mapping(cme)
