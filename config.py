from dataclasses import dataclass


@dataclass
class LLMConfig:
    batch_size: int
    seq_len: int
    embedding_dim: int
    dim_ff: int
    num_head: int
    num_layer: int

    @property
    def head_size(self):
        return self.embedding_dim // self.num_head

    def to_simulatable_config(self):
        """Return a new LLMConfig instance with reduced parameters to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        return LLMConfig(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            embedding_dim=self.embedding_dim,
            dim_ff=self.dim_ff,
            num_head=1,
            num_layer=1,
        )


LLAMA_7B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
)

LLAMA_13B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=20480,
    num_head=40,
    num_layer=40,
)

OPT_125M = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
)
