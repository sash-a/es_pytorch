import numpy as np
from mpi4py import MPI


def sum_obstat(obstat1, obstat2, dtype):
    obstat1 += obstat2
    return obstat1


sumobstat_op = MPI.Op.Create(sum_obstat, commute=True)


class ObStat:
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def inc(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    def __iadd__(self, other):
        self.inc(other.sum, other.sumsq, other.count)
        return self

    def __repr__(self):
        return f'sum:{self.sum} sumsq:{self.sumsq} count:{self.count}'

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def mpi_inc(self, comm: MPI.Comm):
        stat = comm.allreduce(self, op=sumobstat_op)
        self.sum = stat.sum
        self.sumsq = stat.sumsq
        self.count = stat.count
