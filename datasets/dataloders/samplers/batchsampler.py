from typing import List
from torch.utils.data import Sampler
import torch.distributed as dist
from datasets.GroundAerialDataset import GroundAerialDataset
class BatchSampler(Sampler[List[int]]):
    """
    # Wrapper for all sampler
    """
    def __init__(
        self, 
        dataset:GroundAerialDataset, 
        name:str,
        batch_size:int, 
        batch_size_limit:int,
        batch_expansion_rate:float, 
        **kw,
    ):
        # sample factory
        self.sample_fn = None
        if name == "BatchSample":
            from datasets.dataloders.samplers.batch import BatchSample
            self.sample_fn = BatchSample(dataset=dataset, **kw)
        elif name == "HeteroTripletSample":
            from datasets.dataloders.samplers.hetero import HeteroTripletSample
            self.sample_fn = HeteroTripletSample(dataset=dataset, **kw)
        else:
            raise NotImplementedError("GroundAerialBatchSampler: %s sample_fn not implemented" % name)

        # gpu mode
        self.use_dist = False
        if dist.is_initialized():
            # multi-gpu
            self.use_dist = True
            if dist.get_rank() == 0: print("GroundAerialBatchSampler: multi-gpu mode")
        else:
            # single-gpu
            print("GroundAerialBatchSampler: single-gpu mode")
        

        self.batch_size = batch_size - batch_size%self.sample_fn.get_k()
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        if batch_expansion_rate is not None:
            assert batch_expansion_rate > 1., "GroundAerialBatchSampler: batch_expansion_rate must be greater than 1"
            assert batch_size <= batch_size_limit, "GroundAerialBatchSampler: batch_size_limit must be greater or equal to batch_size"

        self.batch_idx = []
        


    def __iter__(self):
        """
        # Generate A Bacth_idx
        """
        self.batch_idx = self.sample_fn(self.batch_size)

        for batch in self.batch_idx: yield batch

    def __len__(self):
        return len(self.batch_idx)
