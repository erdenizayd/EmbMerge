import torch
import torch

def collate_fn(batch):
    batched = {'list1': {}, 'list2': {}, 'list3': {}}
    for i in [1, 2, 3]:
        key = f'list{i}'
        all_item_ids = []
        all_scores = []
        all_labels = []
        all_ranks = []

        for sample in batch:
            item_ids = sample[key]['item_ids'] 
            
            scores = sample[key]['scores']      
            pos_idx = sample[key]['positive_index']

            labels = torch.zeros_like(item_ids, dtype=torch.long)
            labels[pos_idx] = 1

            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            ranks = torch.empty_like(sorted_indices)
            ranks[sorted_indices] = torch.arange(len(scores), dtype=torch.long)

            sorted_by_ids, indices_by_id = torch.sort(item_ids)
            sorted_scores = scores[indices_by_id]
            sorted_labels = labels[indices_by_id]
            sorted_ranks = ranks[indices_by_id]

            all_item_ids.append(sorted_by_ids)
            all_scores.append(sorted_scores)
            all_labels.append(sorted_labels)
            all_ranks.append(sorted_ranks)
    
        batched[key]['item_ids'] = torch.stack(all_item_ids)
        batched[key]['scores'] = torch.stack(all_scores)
        batched[key]['labels'] = torch.stack(all_labels)
        batched[key]['ranks'] = torch.stack(all_ranks)

    return batched
