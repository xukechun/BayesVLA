import torch
from ..va import VA
from ..layers.attn_pre import AdaLN


device = "cpu"
ckpt = torch.load("checkpoints/VA/m0411_general_va/ckpt_best.pt", map_location=device)
va = VA(256).to(device)
va.load_state_dict(ckpt["weights"])


for m in va.modules():
    if isinstance(m, AdaLN):
        print("weight:")
        print(m.modulation[-1].weight)
        print()
        print("bias:")
        print(m.modulation[-1].bias)

