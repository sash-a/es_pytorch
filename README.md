# Evolutionary search in pytorch with MPI

This implementation to be as simple and efficient as possible.  
Reference implementation can be found [here](https://github.com/uber-research/deep-neuroevolution) (in tensorflow using redis).  
Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).

### Implementation

It was made for use on a cluster using MPI (however it can be used on a single machine). With regards to efficiency it only scatters 2 values, per policy evaluated, to all other processes each generation and the noise is placed in a block of shared memory on each node for fast access and low memory footprint.

### How to use

Packages in `env.yml` and example useage in `main.py`.  
If using a single machine instantiate the NoiseTable using its constructor, but if on a cluster with MPI use `NoiseTable.create_shared_noisetable`, to create a block of shared memory on each node for the noise.
