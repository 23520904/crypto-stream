from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .models import build_model


def rmse_steps(y, p):
    return np.sqrt(((np.asarray(y) - np.asarray(p)) ** 2).mean(0))


@torch.no_grad()
def predict(model, X, device="cpu", bs=4096):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def train_run(model_name, Xtr, ytr, hidden, bcfg, seed, device="cpu", epochs=None):
    """Adam (tham số mặc định như [1]), tối đa `epochs`, early-stopping trên 10% cuối của train (theo thời gian)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = epochs or bcfg["epochs"]
    n_val = max(1, int(len(Xtr) * bcfg["val_ratio"]))
    Xt, yt, Xv, yv = Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]
    model = build_model(model_name, Xtr.shape[2], Xtr.shape[1], ytr.shape[1], hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=bcfg["lr"])    # betas=(0.9,0.999), eps=1e-8
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xt), torch.from_numpy(yt)), batch_size=bcfg["batch_size"], shuffle=True)
    Xv_t, yv_t = torch.from_numpy(Xv).to(device), torch.from_numpy(yv).to(device)
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xv_t), yv_t).item()
        if v < best - 1e-10:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= bcfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model.cpu(), best, ep
