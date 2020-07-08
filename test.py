import numpy as np


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)


def get_ref_batch(env, batch_size=32):
    ref_batch = []
    ob = env.reset()
    while len(ref_batch) < batch_size:
        ob, rew, done, info = env.step(env.action_space.sample())
        ref_batch.append(ob)
        if done:
            ob = env.reset()
    return ref_batch


def batched_weighted_sum(weights, vecs, batch_size):
    # print(f'in func: \nrews:{weights}\nnoise:\n{vecs}')

    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


def noise(size):
    return np.random.rand(size)


n_rews = 10
params = 20

rewards = np.array(list(range(n_rews)))
noises = [noise(params) for _ in range(n_rews)]

np_noise = np.array(list(noises))
print(np_noise.shape, rewards.shape)
mine = np.dot(rewards, np_noise)
print(mine)

g, count = batched_weighted_sum(rewards, noises, 5)

print(g)

print(mine == g)
# print(g.shape)
# print(count)

# print(my_method)
# print(my_method.shape)
