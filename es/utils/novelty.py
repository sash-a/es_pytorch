import heapq
from typing import Sequence, Optional

import numpy as np
from mpi4py import MPI
from scipy.spatial import distance


def update_archive(comm: MPI.Comm, behaviour: Sequence[float], archive: Optional[np.ndarray]) -> np.ndarray:
    behaviour = comm.scatter([behaviour] * comm.size)
    if archive is None:
        return np.array([behaviour])
    return np.concatenate((archive, [behaviour]))


def novelty(behaviour: np.ndarray, archive: np.ndarray, n: int) -> float:
    dists = heapq.nsmallest(n, distance.cdist(np.array([behaviour]), archive, 'sqeuclidean')[0])
    return sum(dists)
