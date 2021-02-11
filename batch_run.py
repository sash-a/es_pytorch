import json
from typing import Optional

from filelock import FileLock
from mpi4py import MPI
from munch import munchify, Munch

import nsra
import obj
from es_pytorch.src.utils import utils


def merge(a: dict, b: dict):
    """
    merges 2 nested dicts (b overrides a)
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries
    """
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key])
            else:  # if not dict then I know its a leaf and want to replace content of a with b
                a[key] = b[key]
        else:
            raise ValueError("Expecting all keys from overriding dictionary to be in overridden")
    return a


RUNS = 'runs'

if __name__ == '__main__':
    comm: MPI.Comm = MPI.COMM_WORLD
    cfg: Optional[Munch] = None

    if comm.rank == 0:
        batch_cfg_file = utils.parse_args()
        with FileLock(f'{batch_cfg_file}.lock'):
            with open(batch_cfg_file) as f:
                batch_cfg_dict: dict = json.load(f)

            for run_cfg_file, run_params in batch_cfg_dict.items():
                if run_params[RUNS] > 0:
                    with open(run_cfg_file) as f:
                        cfg_dict = json.load(f)
                    cfg = munchify(merge(cfg_dict, run_params['overrides']))
                    cfg.general.name = f'{cfg.general.name}-{run_params[RUNS]}'

                    batch_cfg_dict[run_cfg_file][RUNS] -= 1
                    json.dump(batch_cfg_dict, open(batch_cfg_file, 'w'), indent=2)
                    break

    cfg = comm.scatter([cfg] * comm.size)
    if 'obj' in cfg.general.name:
        if comm.rank == 0:
            print(f'Starting run: {cfg.general.name}')
        obj.main(cfg)
    elif 'ns' in cfg.general.name:
        if comm.rank == 0:
            print(f'Starting run: {cfg.general.name}')
        nsra.main(cfg)
    else:
        raise Exception(f'Could not recognise run type from run name: {cfg.general.name}')
