from matplotlib import pyplot as plt


def graphresults(file: str):
    noiseless = []
    avgs = []
    maxs = []
    with open(file) as f:
        for line in f.readlines():
            if 'noiseless' in line:
                noiseless.append(float(line.split('-')[1].split(':')[1][:-1]))
            else:
                avg, mx = line.split('-')[1:]
                avgs.append(float(avg.split(':')[1]))
                maxs.append(float(mx.split(':')[1][:-1]))

        # plt.plot(noiseless, label='noiseless')
        plt.plot(avgs, label='avg')
        # plt.plot(maxs, label='max')
        plt.show()


if __name__ == '__main__':
    graphresults('../../results.log')
