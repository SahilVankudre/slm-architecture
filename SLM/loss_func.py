from io_pairs import iopairs
import torch
from training_config import glob as g

class loss_function:

    def estimate_loss(model):
        out = {}
        model.eval()
        with torch.inference_model():
            for split in ['train', 'val']:
                losses = torch.zeros(g.eval_iters)
                for k in range(g.eval_iters):
                    X, Y = iopairs.get_batch(split)
                    with g.ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean()
        model.train()
        return out