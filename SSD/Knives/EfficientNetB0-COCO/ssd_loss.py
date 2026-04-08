import torch
import torch.nn.functional as F


def ssd_loss(
    cls_preds,  # [A, num_classes]
    loc_preds,  # [A, 4]
    cls_targets,  # [A]
    loc_targets,  # [A, 4]
    positive_mask,  # [A]
):
    """
    cls_preds  : raw logits
    loc_preds  : predicted offsets
    cls_targets: GT class labels (0 = background)
    loc_targets: GT box offsets
    positive_mask: boolean mask for positive anchors
    """

    # =========================
    # 1. LOCALIZATION LOSS
    # =========================
    pos_loc_preds = loc_preds[positive_mask]
    pos_loc_targets = loc_targets[positive_mask]

    if pos_loc_preds.numel() == 0:
        loc_loss = torch.tensor(0.0, device=cls_preds.device)
    else:
        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, reduction="sum")

    # =========================
    # 2. CLASSIFICATION LOSS
    # =========================
    cls_loss_all = F.cross_entropy(cls_preds, cls_targets, reduction="none")

    # positives
    pos_cls_loss = cls_loss_all[positive_mask]

    # negatives
    negative_mask = cls_targets == 0
    neg_cls_loss = cls_loss_all[negative_mask]

    num_pos = positive_mask.sum().item()
    num_neg = min(3 * num_pos, neg_cls_loss.numel())

    if num_neg > 0:
        neg_cls_loss, _ = torch.topk(neg_cls_loss, num_neg)
        cls_loss = pos_cls_loss.sum() + neg_cls_loss.sum()
    else:
        cls_loss = pos_cls_loss.sum()

    # =========================
    # NORMALIZE
    # =========================
    num_pos = max(1, num_pos)
    total_loss = (cls_loss + loc_loss) / num_pos

    return total_loss, cls_loss / num_pos, loc_loss / num_pos


# -------------------------
# DUMMY TARGETS (TEST ONLY)
# -------------------------
A = 1600
num_classes = 4


cls_targets = torch.zeros(A, dtype=torch.long)
positive_mask = torch.zeros(A, dtype=torch.bool)

# assume 10 positive anchors
positive_mask[:10] = True
cls_targets[:10] = 1  # pistol

loc_targets = torch.randn(A, 4)


cls_preds = torch.randn(A, num_classes)
loc_preds = torch.randn(A, 4)

# assume these from STEP 9
# cls_targets, loc_targets, positive_mask

loss, cls_l, loc_l = ssd_loss(
    cls_preds, loc_preds, cls_targets, loc_targets, positive_mask
)

print(loss, cls_l, loc_l)
