import torch
from slm_arch import GPT
from config import config_details as cf
from tokenization import tokenizer as t

class run:

    model = GPT(cf.config)
    device = "cuda" if torch.cuda.is_avaliable() else "cpu"
    best_model_params_path = "best_model_params.pt"

    model.load_state_dict(torch.load(best_model_params_path, map_location = torch.device(device)))

    sentence = "Once upon a time there was a pumpkin."
    context = (torch.tensor(t.enc.encode_ordinary(sentence)), torch.unsqueeze(dim = 0))
    y = model.generate(context, 200)
    print(t.enc.decode(y.squeeze().tolist()))