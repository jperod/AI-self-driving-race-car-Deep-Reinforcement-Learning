import matplotlib.pyplot as plt
import numpy as np

file_to_read = 'train22' + '.txt'

try:
    text_results = open(file_to_read, "r")

    cnt = 0

    episode = []
    score = []
    total_steps = []
    epsilon = []
    avg_100_score = []
    last_10_score = np.zeros(10)
    ymax = []
    ymin = []

    mask = np.ones(11, dtype=bool)
    mask[0] = False

    line = text_results.readline()
    while line:
        line_list = line.split()
        episode = np.append(episode, line_list[2])
        score = np.append(score, line_list[5])
        total_steps = np.append(total_steps, line_list[9])
        epsilon = np.append(epsilon, line_list[12])
        avg_100_score = np.append(avg_100_score, line_list[17])

        last_10_score = np.append(last_10_score, line_list[5])
        # last_10_score = [float(i) for i in last_10_score]
        last_10_score = last_10_score[mask]
        last_10_score = [float(i) for i in last_10_score]
        ymax = np.append(ymax, max(last_10_score))
        ymin = np.append(ymin, min(last_10_score))

        line = text_results.readline()
        cnt += 1

    episode = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score = [float(i) for i in avg_100_score]

    fig = plt.figure()
    plt.grid(True)
    ax = plt.axes()
    plt.rc('grid', linestyle="-", color='black')
    plt.plot(total_steps, avg_100_score, color='#CC4F1B');
    plt.fill_between(total_steps, ymin, ymax,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.title('Figure from ' + file_to_read)
    plt.xlabel('Total Time steps')
    plt.ylabel('Score')

    plt.show()

    print('Done 1')

finally:
    text_results.close()