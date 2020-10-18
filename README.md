# Evolutionary strategies (deep neuroevolution) in pytorch using MPI

This implementation was made to be as simple and efficient as possible.  
Reference implementation can be found [here](https://github.com/uber-research/deep-neuroevolution) (in tensorflow using redis).  
Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).

### Implementation
This was made for use on a cluster using MPI (however it can be used on a single machine). With regards to efficiency it 
only scatters the positive fitness, negative fitness and noise index, per policy evaluated, to all other processes each generation. The noise is placed in a block 
of shared memory (on each node) for fast access and low memory footprint.

### How to run
* conda install: `conda install -n es_env -f env.yml`
* example usages: `simple_example.py` `obj.py` `nsra.py`
* example configs are in `config/`

```
conda activate es_env
mpirun -np {num_procs} python simple_example.py configs/simple_conf.json
```

Make sure that you insert this line before you create your neural network as the initial creation sets the 
initial parameters, which must be deterministic across all threads
```
torch.random.manual_seed({seed})
```

### General info
* In order to define a policy all one needs to do is create an `nn.Module` and pass it to a `Policy`, an example of this
 can be seen in `simple_example.py`.
* If you wish to share the noise using shared memory and MPI, then instantiate the `NoiseTable` using 
`NoiseTable.create_shared(...)`, otherwise if you wish to use your own method of sharing noise/running 
sequentially then simply create the noise table using its constructor and pass your noise to it like this: 
`NoiseTable(my_noise, n_params)`
* `NoiseTable.create_shared(...)` will throw an error if less than 2 MPI procs are used
