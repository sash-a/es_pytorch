# Evolutionary search in pytorch using MPI

This implementation was made to be as simple and efficient as possible.  
Reference implementation can be found [here](https://github.com/uber-research/deep-neuroevolution) (in tensorflow using redis).  
Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).

### Implementation

This was made for use on a cluster using MPI (however it can be used on a single machine). With regards to efficiency it 
only scatters 2 values, per policy evaluated, to all other processes each generation and the noise is placed in a block 
of shared memory on each node for fast access and low memory footprint.

### How to use

* Packages in `env.yml` and example useage in `main.py`.  
* In order to define a policy all one needs to do is create an `nn.Module` and pass it to a `Policy`. Then create an 
evaluation function that can return the fitness of that policy, an example of this can be seen in `main.py`.  
* If you wish to share the noise using shared memory and MPI, then instantiate the `NoiseTable` using 
`NoiseTable.create_shared_noisetable(...)`, otherwise if you wish to use your own method of sharing the noise then simply 
create the noise table using its constructor and pass your noise to it like this: `NoiseTable(my_noise, n_params)`