# Taken from https://github.com/suvash/nnze2he/blob/main/makemore/src/gpt.py

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

import config as cfg


device = "cpu"
dropout = 0.3
vocab_size = 1_000_000


# single head of self attention
class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size: int):
        super().__init__()  # type: ignore
        self.key = nn.Linear(cfg.HIDDEN_DIM, head_size, bias=False)
        self.query = nn.Linear(cfg.HIDDEN_DIM, head_size, bias=False)
        self.value = nn.Linear(cfg.HIDDEN_DIM, head_size, bias=False)
        # in Pytorch convention a variable that's not a parameter of the model is called a buffer
        self.register_buffer("tril", torch.tril(torch.ones(cfg.SEQ_LEN, cfg.SEQ_LEN)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        _, T, _ = x.shape
        # emit keys and queries for x
        k: Tensor = self.key(x)  # (B, T, hs)
        q: Tensor = self.query(x)  # (B, T, hs)
        # compute attention
        wei: Tensor = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)  # dropout some of the affinities
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, head_size: int):
        super().__init__()  # type: ignore
        self.heads = nn.ModuleList([Head(head_size) for _ in range(cfg.NUM_HEAD)])
        self.proj = nn.Linear(cfg.HIDDEN_DIM, cfg.HIDDEN_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)  # outcome of the linear layer to project back into the residual pathway
        out = self.dropout(out)  # final dropout
        return out


class FeedForward(nn.Module):
    "simple linear layer followed by non linearity"

    def __init__(self):
        super().__init__()  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(cfg.HIDDEN_DIM, 4 * cfg.HIDDEN_DIM),  # as mentioned in the paper
            nn.ReLU(),
            nn.Linear(
                4 * cfg.HIDDEN_DIM, cfg.HIDDEN_DIM
            ),  # projection layer : the final projection back into the residual pathway
            nn.Dropout(dropout),  # dropout before final projection
        )

    def forward(self, x: Tensor):
        return self.net(x)


class Block(nn.Module):
    """a transformer block : communication then computation"""

    def __init__(self):
        super().__init__()  # type: ignore
        head_size = cfg.HIDDEN_DIM // cfg.NUM_HEAD
        self.sa = MultiHeadAttention(head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(cfg.HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(cfg.HIDDEN_DIM)

    def forward(self, x: Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()  # type: ignore
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, cfg.HIDDEN_DIM)
        self.position_embedding_table = nn.Embedding(cfg.SEQ_LEN, cfg.HIDDEN_DIM)
        self.blocks = nn.Sequential(*[Block() for _ in range(cfg.NUM_LAYER)])
        self.ln_f = nn.LayerNorm(cfg.HIDDEN_DIM)
        self.lm_head = nn.Linear(cfg.HIDDEN_DIM, vocab_size)

    def forward(self, idx: Tensor):
        _, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        return logits

    def generate(self, idx: Tensor, max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block size token
            idx_cond = idx[:, -cfg.SEQ_LEN :]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    model = LanguageModel()
    dummy_input = torch.randint(low=0, high=255, size=(cfg.BATCH_SIZE, cfg.SEQ_LEN))
    torch.onnx.export(  # type: ignore
        model,
        dummy_input,
        "out/custom_transformer.onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        verbose=False,
    )
