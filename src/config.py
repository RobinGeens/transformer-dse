from dataclasses import dataclass


class LLMConfig:
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        embedding_dim: int,
        dim_ff: int,
        num_head: int,
        num_layer: int,
        head_size: int | None = None,
        vocab_size: int = 1000,
        name: str = "",
    ):
        if head_size is None:
            head_size = embedding_dim // num_head

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dim_ff = dim_ff
        self.num_head = num_head
        self.num_layer = num_layer
        self.head_size = head_size
        self.vocab_size = vocab_size
        self.name = name

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
            head_size=self.head_size,  # Keep original so it doesn't get re-computed
            vocab_size=self.vocab_size,
            name=self.name,
        )


@dataclass
class QuantConfig:
    weight_bits: int
    act_bits: int

    @property
    def name(self):
        return f"W{self.weight_bits}A{self.act_bits}"


W8A8 = QuantConfig(8, 8)
W4A8 = QuantConfig(4, 8)
W4A16 = QuantConfig(4, 16)

LLAMA_1_7B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="LLAMA_1_7B",
)

LLAMA_1_13B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="LLAMA_1_13B",
)

LLAMA_1_30B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=6_656,
    dim_ff=17_920,
    num_head=52,
    num_layer=52,  # ! or 60?
    vocab_size=32_000,
    name="LLAMA_1_30B",
)

LLAMA_2_7B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="LLAMA_2_7B",
)

LLAMA_2_13B = LLMConfig(
    batch_size=1,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="LLAMA_2_13B",
)


OPT_125M = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
    vocab_size=50_272,
    name="OPT_125M",
)


OPT_1_3B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=2048,
    dim_ff=8_192,
    num_head=32,
    num_layer=24,
    vocab_size=50_272,
    name="OPT_1_3B",
)

OPT_2_7B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=2560,
    dim_ff=10240,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT_2_7B",
)

OPT_6_7B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT_6_7B",
)

OPT_13B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=20480,
    num_head=40,
    num_layer=40,
    vocab_size=50_272,
    name="OPT_13B",
)

OPT_30B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=7_168,
    dim_ff=28_672,
    num_head=56,
    num_layer=48,
    vocab_size=50_272,
    name="OPT_30B",
)

GPT3_175B = LLMConfig(
    batch_size=1,
    seq_len=2048,
    embedding_dim=12_288,
    dim_ff=49_152,
    num_head=96,
    num_layer=96,
    vocab_size=50257,
    name="GPT3_175B",
)

ALL_MODELS = [
    LLAMA_1_7B,
    # LLAMA_1_13B,
    LLAMA_1_30B,
    # LLAMA_2_7B,
    LLAMA_2_13B,
    OPT_125M,
    # OPT_1_3B,
    # OPT_2_7B,
    # OPT_6_7B,
    # OPT_13B,
    # OPT_30B,
    GPT3_175B,
]
