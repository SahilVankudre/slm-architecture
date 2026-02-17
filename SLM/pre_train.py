import matplotlib.pyplot as plt
import tiktoken
import os
import torch
import numpy as np
from tqdm.auto import tqdm
from training_config import glob as g, optimizer as o
from config import config_details as cf
from io_pairs import iopairs
from loss_func import loss_function as ls
from slm_arch import GPT

best_val_loss = float('inf')
best_model_params_path = "best_model_params.pt"
train_loss_list, validation_loss_list = [], []

class pre_train:

    model = GPT(cf.config)
    model = model.to(g.device)

    for epoch in tqdm(range(g.max_iters)):
        if epoch % g.eval_iters == 0 and epoch != 0:
            losses = ls.estimate_loss(model)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {o.optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]
            validation_loss_list += [losses['val']]

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)

        X, y = iopairs.get_batch("train")
        X, y = X.to(g.device), y.to(g.device)

        with g.ctx:
            logits, loss = model(X, y)
            loss = loss / g.gradient_accumulation_steps
            o.scaler.scale(loss).backward()

        if ((epoch + 1) % g.gradient_accumulation_steps == 0) or (epoch + 1 == g.max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 0.5)
            o.scaler.step(o.optimizer)
            o.scaler.update()
            o.optimizer.zero_grad(set_to_none = True)

        o.scheduler.step()

class loss_ploting:

    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    plt.plot(train_loss_list_converted, 'g', label = 'train_loss')
    plt.plot(validation_loss_list_converted, 'r',label = 'validation_loss')
    plt.xlabel("Steps - Every 100 epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()