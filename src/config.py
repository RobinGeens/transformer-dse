from dataclasses import dataclass


@dataclass
class LLMConfig:
    batch_size: int
    seq_len: int
    embedding_dim: int
    dim_ff: int
    num_head: int
    num_layer: int
    vocab_size: int = 1000
    name: str = ""

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
            vocab_size=self.vocab_size,
            name=self.name,
        )


@dataclass
class QuantConfig:
    act_bits: int
    weight_bits: int

    @property
    def name(self):
        return f"W{self.weight_bits}A{self.act_bits}"


W8A8 = QuantConfig(8, 8)

LLAMA_7B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="LLAMA_7B",
)

LLAMA_13B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="LLAMA_13B",
)

OPT_125M = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
    name="OPT_125M",
)
