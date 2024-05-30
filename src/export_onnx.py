import onnx
import torch
from src.transformer_model import LanguageModel

from src.config import LLAMA_7B, LLMConfig


def export_transformer_to_onnx(cfg: LLMConfig):
    path = "out/custom_transformer.onnx"
    model = LanguageModel(cfg)
    dummy_input = torch.randint(low=0, high=255, size=(cfg.batch_size, cfg.seq_len))
    torch.onnx.export(  # type: ignore
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )

    onnx.shape_inference.infer_shapes_path(path, path)  # type: ignore


if __name__ == "__main__":
    export_transformer_to_onnx(LLAMA_7B.to_simulatable_config())
