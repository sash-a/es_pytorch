# Evolutionary search (deep neuroevolution) in pytorch using MPI

This implementation was made to be as simple and efficient as possible.  
Reference implementation can be found [here](https://github.com/uber-research/deep-neuroevolution) (in tensorflow using redis).  
Based on two papers by uber AI labs [here](https://arxiv.org/abs/1712.06567) and [here](https://arxiv.org/abs/1712.06560).

### Implementation
This was made for use on a cluster using MPI (however it can be used on a single machine). With regards to efficiency it 
only scatters 2 values, per policy evaluated, to all other processes each generation and the noise is placed in a block 
of shared memory on each node for fast access and low memory footprint.


### How to run
* conda install: `conda install -n es_env -f env.yml`
* example main file is in `src/main`
* example configs are in `config/`
* If you change the env name in the config you need to edit the network since the input and output sizes of the network 
will likely be incorrect

Make sure that you insert this line before you create your neural network as the initial creation sets the 
initial parameters, which must be deterministic across processors
```
torch.random.manual_seed(cfg.seed)
```

##### mpi4py
```
mpirun -np {num_procs} --hostfile {hostfile} python {your_main.py} {your_config.json}
```

##### Sequential
If you wish to run sequentially/use a different parallelism library, then you need to adapt `src/es/es_runner.py`,
however a main benefit of this approach is that it is highly parallelisable, so sequential running is not recommended.

### General info
* In order to define a policy all one needs to do is create an `nn.Module` and pass it to a `Policy`, an example of this
 can be seen in `main.py`.
* If you wish to share the noise using shared memory and MPI, then instantiate the `NoiseTable` using 
`NoiseTable.create_shared_noisetable(...)`, otherwise if you wish to use your own method of sharing noise/running 
sequentially then simply create the noise table using its constructor and pass your noise to it like this: 
`NoiseTable(my_noise, n_params)`
