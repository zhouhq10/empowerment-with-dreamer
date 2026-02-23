from . import generic
from . import selectors
from . import limiters


class Uniform(generic.Generic):

    def __init__(
        self,
        length, # length of the replay buffer
        capacity=None, # maximum number of samples that can be stored in the replay buffer
        directory=None, # directory to store the replay buffer
        online=False, # whether to sample online or offline
        chunks=1024, # the number of chunks to divide the replay buffer into
        min_size=1, 
        samples_per_insert=None,
        tolerance=1e4,
        seed=0,
    ):
        if samples_per_insert:
            limiter = limiters.SamplesPerInsert(samples_per_insert, tolerance, min_size)
        else:
            limiter = limiters.MinSize(min_size)
        assert not capacity or min_size <= capacity
        super().__init__(
            length=length,
            capacity=capacity,
            remover=selectors.Fifo(),
            sampler=selectors.Uniform(seed),
            limiter=limiter,
            directory=directory,
            online=online,
            chunks=chunks,
        )
