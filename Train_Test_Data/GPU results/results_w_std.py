import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns

fn = 'train24'
file_to_read = fn + '.txt'

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
    std = np.round(statistics.stdev(avg_100_score[0:150]),2)
    fig = plt.figure()
    plt.grid(True)
    ax = plt.axes()
    plt.rc('grid', linestyle="-", color='black')
    plt.plot(episode, avg_100_score, color='#CC4F1B')
    plt.plot(episode, score, color='#CC4F1B', alpha=0.3)
    plt.axvline(x=100, color='red')
    maxavg = float(max(avg_100_score[100:150]))
    plt.title('Final Test (' + file_to_read + ') -> ' + str(np.round(np.mean(avg_100_score[100:150]),2)) + '+- ' + str(std) + ' | Max: ' + str(maxavg))
    plt.xlabel('Episodes')
    plt.ylabel('Score')

    plt.show()

    # plt.title('Scores Distribution')
    fig2 = plt.figure()
    # bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    # plt.hist(score, bins, histtype='bar', rwidth=0.8)
    plt.hist(score, color='blue', edgecolor='black',
             bins=int(180 / 5))

    # seaborn histogram
    sns.distplot(score, hist=True, kde=False,
                 bins=int(180 / 5), color='blue',
                 hist_kws={'edgecolor': 'black'})
    # Add labels
    plt.title('Histogram of Test Scores')
    plt.xlabel('Episode Test Score')
    plt.ylabel('Count')



    plt.show()





    print('Done 1')
    fig.savefig('Final_test_'+ fn +'.png')
finally:
    text_results.close()