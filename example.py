import pickle
from zigzag import api
from zigzag.visualization.results.plot_cme import (
    bar_plot_cost_model_evaluations_breakdown,
)
from zigzag.visualization.results.print_mapping import print_mapping
from zigzag.visualization.graph.memory_hierarchy import visualize_memory_hierarchy_graph

model = "custom_transformer"
workload_path = "out/custom_transformer.onnx"
accelerator_path = "inputs/hardware/tpu_like.yaml"
mapping_path = "inputs/mapping/default.yaml"
pickle_filename = f"out/TPU-{model}-saved_list_of_cmes.pickle"
RE_RUN = True

if RE_RUN:
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


bar_plot_cost_model_evaluations_breakdown(cmes, save_path="plot_breakdown.png")

visualize_memory_hierarchy_graph(
    cmes[0].accelerator.cores[0].memory_hierarchy,
    save_path="out/mem_hierarchy.png",
)

for cme in cmes:
    print_mapping(cme)
