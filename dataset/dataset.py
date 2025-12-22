import torch

class DigixDataset(torch.utils.data.Dataset):
    def __init__(self, data_rows):
        self.data_rows = data_rows

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        row = self.data_rows[idx]
        samples = {}

        for i, lst in enumerate(row, start=1):
            item_ids = [x[0] for x in lst[1]]
            scores = [x[1] for x in lst[1]]
            pos_index = item_ids.index(lst[0])
            samples[f'list{i}'] = {
                'item_ids': torch.tensor(item_ids),
                'scores': torch.tensor(scores),
                'positive_index': pos_index
            }

        return samples
