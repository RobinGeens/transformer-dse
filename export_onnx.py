# Taken from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py

import math
import onnx
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

import config as cfg


device = "cpu"
dropout = 0.3
VOCAB_SIZE = 1000


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size: int):
        super().__init__()  # type: ignore
        self.key = nn.Linear(cfg.EMBEDDING_DIM, head_size, bias=False)
        self.query = nn.Linear(cfg.EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(cfg.EMBEDDING_DIM, head_size, bias=False)
        # in Pytorch convention a variable that's not a parameter of the model is called a buffer
        self.register_buffer("tril", torch.tril(torch.ones(cfg.SEQ_LEN, cfg.SEQ_LEN)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        _, T, _ = x.shape
        key: Tensor = self.key(x)  # (B, T, hs)
        query: Tensor = self.query(x)  # (B, T, hs)
        attention: Tensor = query @ key.transpose(-2, -1)  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        attention = attention * math.sqrt(key.shape[-1])
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        logits = F.softmax(attention, dim=-1)  # (B, T, T)
        logits = self.dropout(logits)
        value = self.value(x)  # (B, T, hs)
        out = logits @ value  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size: int):
        super().__init__()  # type: ignore
        self.heads = nn.ModuleList([Head(head_size) for _ in range(cfg.NUM_HEAD)])
        self.proj = nn.Linear(cfg.EMBEDDING_DIM, cfg.EMBEDDING_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    "simple linear layer followed by non linearity"

    def __init__(self):
        super().__init__()  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(cfg.EMBEDDING_DIM, cfg.DIM_FF),
            nn.ReLU(),
            nn.Linear(cfg.DIM_FF, cfg.EMBEDDING_DIM),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    """a transformer block : communication then computation"""

    def __init__(self):
        super().__init__()  # type: ignore
        head_size = cfg.EMBEDDING_DIM // cfg.NUM_HEAD
        self.sa = MultiHeadAttention(head_size)
        self.feed_forward = FeedForward()
        self.ln1 = nn.LayerNorm(cfg.EMBEDDING_DIM)
        self.ln2 = nn.LayerNorm(cfg.EMBEDDING_DIM)

    def forward(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()  # type: ignore
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, cfg.EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(cfg.SEQ_LEN, cfg.EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[Block() for _ in range(cfg.NUM_LAYER)])
        self.layer_norm_final = nn.LayerNorm(cfg.EMBEDDING_DIM)
        self.de_embed = nn.Linear(cfg.EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, idx: Tensor):
        _, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.layer_norm_final(x)
        logits = self.de_embed(x)  # (B,T,VOCAB_SIZE)

        return logits


if __name__ == "__main__":
    path = "out/custom_transformer.onnx"
    model = LanguageModel()
    dummy_input = torch.randint(low=0, high=255, size=(cfg.BATCH_SIZE, cfg.SEQ_LEN))
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
