from matplotlib import pyplot as plt


class Objective:
    def __init__(self):
        self.avg = 0
        self.max = 0


class Result:
    def __init__(self):
        self.gen = 0
        self.dist = 0
        self.rew = 0
        self.idx = 0
        self.w = 0
        self.obj0 = Objective()
        self.obj1 = Objective()


def get_value(line: str):
    return float(line.split(':')[3][:-1])


def graph(file: str):
    results = []
    result = Result()

    with open(file) as f:
        for line in f.readlines():
            if 'gen' in line:
                results.append(result)
                result = Result()
                result.gen = get_value(line)
            if 'obj 0 avg' in line:
                result.obj0.avg = get_value(line)
            if 'obj 0 max' in line:
                result.obj0.max = get_value(line)
            if 'obj 1 avg' in line:
                result.obj1.avg = get_value(line)
            if 'obj 1 max' in line:
                result.obj1.max = get_value(line)
            if 'dist' in line:
                result.dist = get_value(line)
            if 'rew' in line:
                result.rew = get_value(line)
            if ':w:' in line:
                result.w = get_value(line)
            if 'idx' in line:
                result.idx = get_value(line)

    results = results[1:]
    # plt.plot([r.dist for r in results], label='dist')
    # plt.plot([r.obj1.avg for r in results], label='avg')
    # plt.plot([r.obj1.max for r in results], label='max')

    for i in range(1):
        # plt.plot([r.w for r in results if r.idx == i], label=f'w {i}')
        print([r.gen for r in results if r.idx == i])
        plt.plot([r.gen for r in results if r.idx == i], [r.obj0.max for r in results if r.idx == i], label=f'max {i}')
        plt.plot([r.gen for r in results if r.idx == i], [r.rew for r in results if r.idx == i], label=f'rew {i}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    graph('../../logs/other.log')
    # graph('../../logs/hopper_mujoco.log')

