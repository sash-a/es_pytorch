import numpy as np


class NoiseTable:
    def __init__(self, seed=123, table_size=250000000):
        import ctypes
        import multiprocessing

        # seed = 123
        # count = 250000000  # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        # logger.info('Sampling {} random numbers with seed {}'.format(count, seed))

        self._shared_mem = multiprocessing.Array(ctypes.c_float, table_size)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())

        assert self.noise.dtype == np.float32

        self.noise[:] = np.random.RandomState(seed).randn(table_size)  # 64-bit to 32-bit conversion here
        print(f'Sampled {self.noise.size * 4} bytes')
        # logger.info('Sampled {} bytes'.format(self.noise.size * 4))

    def get(self, i, size) -> np.ndarray:
        return self.noise[i:i + size]

    def sample_index(self, stream, size) -> np.ndarray:
        return stream.randint(0, len(self.noise) - size + 1)

    def __len__(self):
        return len(self.noise)
