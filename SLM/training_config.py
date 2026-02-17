import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR
from slm_arch import GPT
from config import config_details as cf

class glob:
    learning_rate = 1e-4
    max_iters = 20000
    warmup_steps = 1000
    min_lr = 5e-4
    eval_iters = 500
    batch_size = 32
    block_size = 128

    gradient_accumulation_steps = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_type = 'cuda' if 'cuda' in device else 'cpu' 

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    pdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type = device_type, dtype = pdtype)

    torch.set_default_device(device)
    torch.manual_seed(42)

class optimizer:

    model = GPT(cf.config)

    optimizer = torch.optim.Adam(model.parameters(), lr = glob.learning_rate, betas=(0.9, 0.95), weight_decay=0.1) #eps=lr-9

    scheduler_warmup = LinearLR(optimizer, total_iters=glob.warmup_steps)
    scheduler_decay = CosineAnnealingLR(optimizer, T_max = glob.max_iters - glob.warmup_steps, eta_min = glob.min_lr)
    scheduler = SequentialLR(optimizer, schedulers = [scheduler_warmup, scheduler_decay], milestones = [glob.warmup_steps])

    scaler = torch.cuda.amp.GradScaler(enabled = (glob.dtype == 'float16'))