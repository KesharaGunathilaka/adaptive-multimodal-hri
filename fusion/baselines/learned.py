"""Learned baselines: per-modality logistic regression and concat-MLP.

Both consume the window-level probability vectors from features_v1.parquet.
The concat-MLP (28-dim input incl. the 4 obs flags -> 128 -> 64 -> 10) is the
floor the attention fusion model must beat (handover §7.2).
"""
import numpy as np
import torch
import torch.nn as nn

from .common import MODALITY_COLS, OBS_COLS, xy


def unimodal_logreg(splits, modality, seed=0):
    """Train logistic regression on one modality's probs; return window preds."""
    from sklearn.linear_model import LogisticRegression
    cols = MODALITY_COLS[modality]
    Xtr = splits["train"][cols].fillna(0).to_numpy(np.float32)
    ytr = splits["train"]["y"].to_numpy()
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(Xtr, ytr)
    return {name: clf.predict(f[cols].fillna(0).to_numpy(np.float32))
            for name, f in splits.items()}


class ConcatMLP(nn.Module):
    def __init__(self, in_dim=28, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


def train_concat_mlp(splits, seed=0, epochs=60, patience=8, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, ytr = xy(splits["train"])
    Xva, yva = xy(splits["val"])
    model = ConcatMLP(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=512, shuffle=True)
    xva = torch.from_numpy(Xva).to(device)

    best_acc, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            loss_fn(model(xb.to(device)), yb.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xva).argmax(1).cpu().numpy() == yva).mean())
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)

    preds = {}
    model.eval()
    with torch.no_grad():
        for name, f in splits.items():
            X, _ = xy(f)
            preds[name] = model(torch.from_numpy(X).to(device)).argmax(1).cpu().numpy()
    return preds, best_acc
