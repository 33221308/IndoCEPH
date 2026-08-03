# utils/angle_loss.py
import torch
import torch.nn.functional as F

def _safe_cosine(u, v, dim=-1):
    u = F.normalize(u, dim=dim)
    v = F.normalize(v, dim=dim)
    return (u * v).sum(dim=dim).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

def angle_true(a, b, c):
    # sudut di titik b dari (a-b) dan (c-b); a,b,c: (...,2)
    ba = a - b
    bc = c - b
    cosang = _safe_cosine(ba, bc)
    return torch.acos(cosang)  # radian

def angle_line(p1, p2, p3, p4):
    # sudut antar garis (p1-p2) dan (p3-p4)
    v1 = p2 - p1
    v2 = p4 - p3
    cosang = _safe_cosine(v1, v2)
    return torch.acos(cosang)  # radian

def _all_present(mask, idxs):
    # mask: (B,17) bool; idxs: tuple/list of indices
    need = torch.stack([mask[:, i] for i in idxs], dim=1)  # (B,k)
    return need.all(dim=1)  # (B,)

@torch.no_grad()
def rad2deg(x):
    return x * 180.0 / torch.pi

def compute_angle_loss(
    pred_xy, tgt_xy,
    true_angle_triplets,      # list of (name, (i,j,k))
    line_angle_pairs,         # list of (name, ((i,j),(k,l)))
    angle_weight_map=None,    # dict name->weight
    valid_mask=None,          # (B,17) bool
    reduction='mean',
    return_breakdown=False
):
    device = pred_xy.device
    B, N, _ = pred_xy.shape
    if valid_mask is None:
        valid_mask = torch.ones((B, N), dtype=torch.bool, device=device)
    if angle_weight_map is None:
        angle_weight_map = {}

    losses = []
    details = {}

    # true-angle
    for name, (i, j, k) in true_angle_triplets:
        present = _all_present(valid_mask, (i, j, k))
        if present.any():
            pa = angle_true(pred_xy[:, i], pred_xy[:, j], pred_xy[:, k])
            ta = angle_true(tgt_xy[:, i],  tgt_xy[:, j],  tgt_xy[:, k])
            diff = (pa - ta)[present]
            loss = (diff ** 2).mean() if diff.numel() else torch.zeros([], device=device)
        else:
            loss = torch.zeros([], device=device)
        w = angle_weight_map.get(name, 1.0)
        losses.append(w * loss)
        if return_breakdown:
            details[name] = {
                'loss': loss.detach(),
                'w': w,
                'present': present.float().mean(),
            }

    # line-angle
    for name, ((i, j), (k, l)) in line_angle_pairs:
        present = _all_present(valid_mask, (i, j, k, l))
        if present.any():
            pa = angle_line(pred_xy[:, i], pred_xy[:, j], pred_xy[:, k], pred_xy[:, l])
            ta = angle_line(tgt_xy[:, i],  tgt_xy[:, j],  tgt_xy[:, k],  tgt_xy[:, l])
            diff = (pa - ta)[present]
            loss = (diff ** 2).mean() if diff.numel() else torch.zeros([], device=device)
        else:
            loss = torch.zeros([], device=device)
        w = angle_weight_map.get(name, 1.0)
        losses.append(w * loss)
        if return_breakdown:
            details[name] = {
                'loss': loss.detach(),
                'w': w,
                'present': present.float().mean(),
            }

    total = torch.stack(losses).sum() if len(losses) else torch.zeros([], device=device)
    if reduction == 'mean' and len(losses) > 0:
        total = total / len(losses)
    return (total, details) if return_breakdown else total
