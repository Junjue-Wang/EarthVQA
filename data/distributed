import math

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

__all__ = ['StepDistributedSampler',
           'DistributedNonOverlapSeqSampler',
           'StepDistributedRandomSubsetSampler',
           'as_ddp_inference_loader'
           ]


class StepDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        super(StepDistributedSampler, self).__init__(dataset, num_replicas, rank)
        self.step = 0

    def set_step(self, step):
        self.step = step

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.step)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class StepDistributedRandomSubsetSampler(StepDistributedSampler):
    def __init__(self, indices, num_replicas=None, rank=None):
        super(StepDistributedRandomSubsetSampler, self).__init__([], num_replicas, rank)

        self.indices = indices
        self.num_samples = int(math.ceil(len(self.indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on step
        g = torch.Generator()
        g.manual_seed(self.step)
        indices = [self.indices[i] for i in torch.randperm(len(self.indices), generator=g)]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_step(self, step):
        self.step = step


class DistributedNonOverlapSeqSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        super(DistributedNonOverlapSeqSampler, self).__init__(dataset, num_replicas, rank)

        self.num_samples = [len(self.dataset) // self.num_replicas] * self.num_replicas
        for i in range(len(self.dataset) % self.num_replicas):
            self.num_samples[i] += 1
        self.total_size = len(self.dataset)
        assert sum(self.num_samples) == self.total_size

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        # subsample
        start = sum(self.num_samples[0:self.rank])
        end = sum(self.num_samples[0:self.rank + 1])

        indices = indices[start:end]
        assert len(indices) == self.num_samples[self.rank]

        return iter(indices)

    def __len__(self):
        return self.num_samples[self.rank]


def as_ddp_inference_loader(dataloader):
    if not isinstance(dataloader.sampler, DistributedNonOverlapSeqSampler):
        loader = DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=DistributedNonOverlapSeqSampler(dataloader.dataset),
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            worker_init_fn=dataloader.worker_init_fn
        )
    else:
        loader = dataloader
    return loader
