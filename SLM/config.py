
from slm_arch import GPT, GPTConfig

class config_details:

    config = GPTConfig(
    vocab_size= 50257, # Tokenizer's vocab size
    block_size = 128, # context size we are training the model with
    n_layer = 6, # number of transforms blocks
    n_head = 6, # num of attention heads
    n_embd = 384, # embedding dimensions
    dropout = 0.1,
    bias = True,
)

model = GPT(config_details)