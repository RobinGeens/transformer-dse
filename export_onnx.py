from typing import Any
import onnx
from onnx import NodeProto
import torch

from src.transformer_model import LanguageModel
from src.config import LLAMA_7B, W8A8, LLMConfig, QuantConfig


def export_transformer_to_onnx(llm_config: LLMConfig, quant_config: QuantConfig):
    path = "out/custom_transformer.onnx"
    model = LanguageModel(llm_config)
    dummy_input = torch.randint(low=0, high=255, size=(llm_config.batch_size, llm_config.seq_len))

    torch.onnx.export(  # type: ignore
        model,
        dummy_input,
        path,
        export_params=False,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    # Perform shape inference in-place
    onnx.shape_inference.infer_shapes_path(path, path)  # type: ignore

    # Prepare for quantization # NOTE this seems to do the same as `infer_shapes_path`
    # quantization.shape_inference.quant_pre_process(path, path)  # type: ignore

    # Add attribute with quantization info, to be used in Zigzag
    onnx_model = onnx.load(path)  # type: ignore
    onnx.checker.check_model(onnx_model)  # type: ignore
    for node in onnx_model.graph.node:
        if node.op_type in ["Add", "Matmul", "Gemm", "Softmax"]:
            add_attribute_to_onnx_node(node, "weight_size", quant_config.weight_bits)
            add_attribute_to_onnx_node(node, "act_size", quant_config.act_bits)

    onnx.save_model(onnx_model, path)  # type: ignore


def add_attribute_to_onnx_node(node: NodeProto, key: str, val: Any):
    attr = onnx.helper.make_attribute(key, val)
    node.attribute.extend([attr])


if __name__ == "__main__":
    llm_config = LLAMA_7B.to_simulatable_config()
    quant_config = W8A8
    export_transformer_to_onnx(llm_config, quant_config)
