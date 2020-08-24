from matplotlib import pyplot as plt


def graph(file: str):
    noiseless = []
    avgs = []
    maxs = []
    with open(file) as f:
        for line in f.readlines():
            if 'avg' in line:
                avgs.append(float(line.split(':')[3][:-1]))
            if 'max' in line:
                maxs.append(float(line.split(':')[3][:-1]))
            if 'noiseless' in line:
                noiseless.append(float(line.split(':')[3][:-1]))

        plt.plot(noiseless, label='noiseless')
        plt.plot(avgs, label='avg')
        plt.plot(maxs, label='max')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    graph('../logs/test.log')
