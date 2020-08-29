from matplotlib import pyplot as plt


def graph(file: str):
    dists = []
    rews = []
    obj_0_avgs = []
    obj_0_maxs = []
    obj_1_avgs = []
    obj_1_maxs = []

    with open(file) as f:
        for line in f.readlines():
            if 'obj 0 avg' in line:
                obj_0_avgs.append(float(line.split(':')[3][:-1]))
            if 'obj 0 max' in line:
                obj_0_maxs.append(float(line.split(':')[3][:-1]))
            if 'obj 1 avg' in line:
                obj_1_avgs.append(float(line.split(':')[3][:-1]))
            if 'obj 1 max' in line:
                obj_1_maxs.append(float(line.split(':')[3][:-1]))
            if 'dist' in line:
                dists.append(float(line.split(':')[3][:-1]))
            if 'rew' in line:
                rews.append(float(line.split(':')[3][:-1]))

        # plt.plot(rews, label='rew')
        # plt.plot(obj_0_avgs, label='avg (0)')
        # plt.plot(obj_0_maxs, label='max (0)')

        plt.plot(dists, label='dist')
        plt.plot(obj_1_avgs, label='avg (1)')
        plt.plot(obj_1_maxs, label='max (1)')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    graph('../../logs/hopper_nsra.log')
