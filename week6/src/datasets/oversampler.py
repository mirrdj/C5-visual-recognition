from collections import Counter

import torch
from torch.utils.data import Dataset

#label_key example: lambda d,i: d.age[i]
class OverSampler(Dataset):
    def __init__(self, dataset: Dataset, desired_dist, label_key) -> None:
        super().__init__()

        self.dataset = dataset
        self.desired_dist = {c: p / sum(desired_dist.values()) for c, p in desired_dist.items()}

        self.actual_counts = Counter()
        for i in range(len(dataset)):
            self.actual_counts[label_key(dataset, i)] += 1
        
        self.sampling_rates = {c: self.actual_counts.total() / self.actual_counts[c] * p for c,p in self.desired_dist.items()}
        min_sampling = min(self.sampling_rates.values())
        assert min_sampling < 1
        self.sampling_rates = {c: s / min_sampling for c,s in self.sampling_rates.items()}
        self.target_counts = {c: self.actual_counts[c] * s for c,s in self.sampling_rates.items()}

        self.indices = []
        added_count = Counter()
        fractional_count = Counter()
        for i in range(len(dataset)):
            label = label_key(dataset, i)
            fractional_count[label] += self.sampling_rates[label]
            for i in range(int(fractional_count[label]) - added_count[label]):
                self.indices.append(i)
                added_count[label] += 1
        
        print("Created oversampled dataset. Final dataset:")
        for label in sorted(added_count):
            frequency = added_count[label] / sum(added_count.values())
            print(f'• {label}: {frequency:.2%} ({added_count[label]})')
        print("Original dataset:")
        for label in sorted(self.actual_counts):
            frequency = self.actual_counts[label] / sum(self.actual_counts.values())
            print(f'• {label}: {frequency:.2%} ({self.actual_counts[label]})')
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
        