import bisect
import random

import torch.utils.data as data


class MixDataset(data.Dataset):
    def __init__(self, dataset_list, partition) -> None:
        super(MixDataset, self).__init__()

        self.db_list = dataset_list
        self.partition = partition

        self._subset_size = [len(item) for item in self.db_list]
        self.tot_size = 2 * max(self._subset_size)

        self.cumulative_sizes = self.cumsum(self.partition)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0

        p = random.uniform(0, 1)

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

        _db_len = self._subset_size[dataset_idx]

        # last batch: random sampling
        if idx >= _db_len * (self.tot_size // _db_len):
            sample_idx = random.randint(0, _db_len - 1)
        else:  # before last batch: use modular
            sample_idx = idx % _db_len

        return self.db_list[dataset_idx][sample_idx]
