import torch

MIN_EPSILON = 1e-10

def lambdaLoss(
    scores, 
    labels, 
    k=None, 
):
    sigma=1.0, 
    mu=10.0,
    pad_val=-1, 
    epsilon=MIN_EPSILON, 
    
    device = scores.device
    
    pred_scores = scores.clone()
    true_labels = labels.clone().float()

    is_padded = true_labels == pad_val
    pred_scores[is_padded] = float("-inf")
    true_labels[is_padded] = float("-inf")

    scores_sorted, sort_indices = pred_scores.sort(descending=True, dim=-1)
    
    labels_reordered = torch.gather(true_labels, dim=1, index=sort_indices)
    
    ideal_labels, _ = true_labels.sort(descending=True, dim=-1)

    label_diff_matrix = labels_reordered.unsqueeze(2) - labels_reordered.unsqueeze(1)
    
    valid_pair_mask = torch.isfinite(label_diff_matrix)

    labels_reordered_clamped = labels_reordered.clamp(min=0.0)
    ideal_labels_clamped = ideal_labels.clamp(min=0.0)

    n_items = pred_scores.shape[1]
    positions = torch.arange(1, n_items + 1, device=device).float()
    discounts = torch.log2(1.0 + positions).unsqueeze(0)

    ideal_gains = (torch.pow(2, ideal_labels_clamped) - 1) / discounts
    idcg = ideal_gains[:, :k].sum(dim=-1).clamp(min=epsilon)

    current_gains = (torch.pow(2, labels_reordered_clamped) - 1) / idcg.unsqueeze(1)

    lambda_weights = 1.0

    score_diff_matrix = (scores_sorted.unsqueeze(2) - scores_sorted.unsqueeze(1))
    
    score_diff_matrix = score_diff_matrix.clamp(min=-1e8, max=1e8)
    score_diff_matrix = torch.nan_to_num(score_diff_matrix, nan=0.0)

    sigmoid_probs = torch.sigmoid(sigma * score_diff_matrix).clamp(min=epsilon)
    weighted_probs = (sigmoid_probs ** lambda_weights).clamp(min=epsilon)

    pair_losses = torch.log2(weighted_probs)

    top_k_mask = torch.zeros((n_items, n_items), dtype=torch.bool, device=device)
    top_k_mask[:k, :k] = True

    final_mask = valid_pair_mask & top_k_mask

    active_losses = pair_losses[final_mask]
    
    total_loss = -torch.sum(active_losses)

    return total_loss