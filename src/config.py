from copy import deepcopy
from dataclasses import dataclass

BATCH_SIZE = 8


class LLMConfig:
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        dim_ff: int,
        num_head: int,
        num_layer: int,
        batch_size: int = 1,
        vocab_size: int = 1000,
        name: str = "",
    ):

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.dim_ff = dim_ff
        self.num_head = num_head
        self.num_layer = num_layer
        self.head_size = embedding_dim // num_head
        self.vocab_size = vocab_size
        self.__name = name

    @property
    def name(self):
        return f"{self.__name}"

    @property
    def parameterized_name(self):
        return f"{self.name.replace('.', '_')}_B={self.batch_size}"

    def to_simulatable_config(self):
        """Return a new LLMConfig instance with reduced parameters to make the simulation go faster. The results
        can then be multiplied to get the actual energy/latency values"""
        cfg = deepcopy(self)
        cfg.seq_len = cfg.seq_len // 2  # Prefill half the context window
        cfg.num_layer = 1
        cfg.num_head = 1  # Keep the original `head_size`!
        return cfg

    def get_post_simulation_factor(self, layer: str):
        """The model is simulated with reduced parameters i.e. only one layer. This function returns the factor with
        which the results for the given layer have to be multiplied in order to come to the result for the full model
        Moreover, the results are normalized to a single inference instead of a full batch"""
        # K, Q, V and output projection
        if "_proj" in layer:
            return 4 * self.num_layer / self.batch_size
        elif "mul_" in layer:
            return self.num_head * self.num_layer / self.batch_size
        # Special case: gate layer in Llama models
        elif "feedforward_expand" in layer and "llama" in self.name.lower():
            return 2 * self.num_layer / self.batch_size
        elif "feedforward_" in layer:
            return self.num_layer / self.batch_size
        else:
            return 1


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
W16A32 = QuantConfig(16, 32)

LLAMA_1_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="Llama1-7B",
)

LLAMA_1_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama1-13B",
)

LLAMA_1_30B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=6_656,
    dim_ff=17_920,
    num_head=52,
    num_layer=60,
    vocab_size=32_000,
    name="Llama1-30B",
)

LLAMA_2_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=4096,
    dim_ff=11_008,
    num_head=32,
    num_layer=32,
    vocab_size=32_000,
    name="Llama2-7B",
)

LLAMA_2_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=4096,
    embedding_dim=5120,
    dim_ff=13_824,
    num_head=40,
    num_layer=40,
    vocab_size=32_000,
    name="Llama2-13B",
)


OPT_125M = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=768,
    dim_ff=3072,
    num_head=12,
    num_layer=12,
    vocab_size=50_272,
    name="OPT-125M",
)


OPT_1_3B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2048,
    dim_ff=8_192,
    num_head=32,
    num_layer=24,
    vocab_size=50_272,
    name="OPT-1.3B",
)

OPT_2_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=2560,
    dim_ff=10240,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-2.7B",
)

OPT_6_7B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=4096,
    dim_ff=16384,
    num_head=32,
    num_layer=32,
    vocab_size=50_272,
    name="OPT-6.7B",
)

OPT_13B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=5120,
    dim_ff=20480,
    num_head=40,
    num_layer=40,
    vocab_size=50_272,
    name="OPT-13B",
)

OPT_30B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=7_168,
    dim_ff=28_672,
    num_head=56,
    num_layer=48,
    vocab_size=50_272,
    name="OPT-30B",
)

GPT3_175B = LLMConfig(
    batch_size=BATCH_SIZE,
    seq_len=2048,
    embedding_dim=12_288,
    dim_ff=49_152,
    num_head=96,
    num_layer=96,
    vocab_size=50257,
    name="GPT3-175B",
)

ALL_MODELS = [
    LLAMA_1_7B,
    # LLAMA_1_13B,
    LLAMA_1_30B,
    # LLAMA_2_7B,
    # LLAMA_2_13B,
    OPT_125M,
    # OPT_1_3B,
    # OPT_2_7B,
    # OPT_6_7B,
    # OPT_13B,
    # OPT_30B,
    GPT3_175B,
]
