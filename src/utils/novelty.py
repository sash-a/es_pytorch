import heapq
from typing import Sequence

import numpy as np
from mpi4py import MPI
from scipy.spatial import distance


def update_archive(comm: MPI.Comm, behaviour: Sequence[float], archive: np.ndarray) -> np.ndarray:
    size = len(behaviour)
    rcv_buff = np.zeros(size, dtype=np.float_)
    comm.Scatter((np.array(behaviour, dtype=np.float_), size, MPI.FLOAT), (rcv_buff, size, MPI.FLOAT))
    if archive is None:
        return np.array([rcv_buff])
    return np.concatenate((archive, [rcv_buff]))


def novelty(behaviour: np.ndarray, archive: np.ndarray, n: int) -> float:
    dists = heapq.nsmallest(n, distance.cdist(behaviour, archive, 'sqeuclidean')[0])
    return sum(dists)
