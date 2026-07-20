"""Training / evaluation loop for the attention fusion model."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..baselines import common
from .datasets import WindowDataset
from .model import AttentionFusion


def _eval_arrays(model, X, obs, device, bs=2048):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            ob = torch.from_numpy(obs[i:i + bs]).to(device)
            preds.append(model(xb, ob).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def frame_arrays(frame):
    X = frame[common.PROB_COLS].fillna(0.0).to_numpy(np.float32)
    obs = frame[common.OBS_COLS].to_numpy(np.float32)
    return X, obs


def _masked_val_acc(model, Xva, obs_va, yva, device):
    """Mean val accuracy over unmasked + each single-modality mask.
    Selecting on this keeps the [MISSING] tokens trained at the checkpoint we
    keep — selecting on unmasked-only val acc picked robustness-poor epochs
    (see WORKLOG 2026-07-17: first sweep degraded worse than concat-MLP)."""
    from .datasets import CUE_SLICES
    from .model import MODALITIES
    accs = [float((_eval_arrays(model, Xva, obs_va, device) == yva).mean())]
    for m in MODALITIES:
        X2, o2 = Xva.copy(), obs_va.copy()
        X2[:, CUE_SLICES[m]] = 0.0
        o2[:, MODALITIES.index(m)] = 0.0
        accs.append(float((_eval_arrays(model, X2, o2, device) == yva).mean()))
    return float(np.mean(accs)), accs[0]


def train_fusion(splits, seed=0, dropout_p=0.0, jitter_sigma=0.0, extra=None,
                 epochs=80, patience=10, lr=1e-3, device=None,
                 select_masked=False, missing_mode="token"):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds = WindowDataset(splits["train"], dropout_p=dropout_p,
                       jitter_sigma=jitter_sigma, seed=seed, extra=extra)
    dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=False)
    Xva, obs_va = frame_arrays(splits["val"])
    yva = splits["val"]["y"].to_numpy()

    model = AttentionFusion(missing_mode=missing_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

    best_score, best_acc, best_state, bad = 0.0, 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, ob, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb.to(device), ob.to(device)), yb.to(device)).backward()
            opt.step()
        if select_masked:
            score, acc = _masked_val_acc(model, Xva, obs_va, yva, device)
        else:
            score = acc = float((_eval_arrays(model, Xva, obs_va, device) == yva).mean())
        if score > best_score:
            best_score, best_acc, bad = score, acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_acc


def evaluate_masked(model, frame, mask_modalities, device):
    """Evaluate with the given modalities force-masked (T03 sweep)."""
    from .model import MODALITIES
    X, obs = frame_arrays(frame)
    X, obs = X.copy(), obs.copy()
    from .datasets import CUE_SLICES
    for m in mask_modalities:
        k = MODALITIES.index(m)
        X[:, CUE_SLICES[m]] = 0.0
        obs[:, k] = 0.0
    pred = _eval_arrays(model, X, obs, device)
    return common.evaluate(frame, pred)
