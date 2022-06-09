import torch
from barlow.main import BarlowTwins
from collections import OrderedDict

def load_backbone(backbone, load_bb):
    if load_bb:
        checkpoint = torch.load(load_bb)
        state_dic = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dic.items():
            if "backbone" not in k:
                continue
            #if "bn" in k:
            #    continue
            key = ".".join(k.split(".")[2:])
            new_state_dict[key] = v
        torch.save({
            "model_state_dict": new_state_dict
        }, "resnet50_params.pth")

        missing_keys, unexpected_keys = backbone.load_state_dict(new_state_dict,strict=False)
        #print("missing", missing_keys)
        #print("unexpected", unexpected_keys)
        #raise Exception
    return backbone

if __name__ == "__main__":
    checkpoint = torch.load("barlow/checkpoint/checkpoint.pth")
    print(checkpoint)
