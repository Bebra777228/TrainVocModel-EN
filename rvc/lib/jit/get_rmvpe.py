import torch

from rvc.lib.predictors.RMVPE import E2E

def get_rmvpe(model_path="assets/rmvpe/rmvpe.pt", device=torch.device("cpu")):
    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
