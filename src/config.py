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
        )


@dataclass
class QuantConfig:
    act_bits: int
    weight_bits: int


# class FullConfig:

#     def __init__(self, llm_config: LLMConfig, quant_config: QuantConfig):

#         # Unpack
#         self.batch_size = llm_config.batch_size
#         self.seq_len = llm_config.seq_len
#         self.embedding_dim = llm_config.embedding_dim
#         self.dim_ff = llm_config.dim_ff
#         self.num_head = llm_config.num_head
#         self.num_layer = llm_config.num_layer
#         self.vocab_size = llm_config.vocab_size
#         self.act_bits = quant_config.act_bits
#         self.weight_bits = quant_config.weight_bits


W8A8 = QuantConfig(8, 8)

LLAMA_7B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
)

LLAMA_13B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
)

OPT_125M = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
)
