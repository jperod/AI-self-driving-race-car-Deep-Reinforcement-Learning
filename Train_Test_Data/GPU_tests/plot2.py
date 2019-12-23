import matplotlib.pyplot as plt
import numpy as np

file_to_read = 'train24' + '.txt'

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

    episode_1 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_1 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_1 = [float(i) for i in avg_100_score]

finally:
    text_results.close()

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

    episode_2 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_2 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_2 = [float(i) for i in avg_100_score]

finally:
    text_results.close()

file_to_read = 'train20(21)' + '.txt'

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

    episode_3 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_3 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_3 = [float(i) for i in avg_100_score]

finally:
    text_results.close()
file_to_read = 'train19' + '.txt'

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

    episode_4 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_4 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_4 = [float(i) for i in avg_100_score]

finally:
    text_results.close()

file_to_read = 'train18' + '.txt'

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

    episode_5 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_5 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_5 = [float(i) for i in avg_100_score]

finally:
    text_results.close()

file_to_read = 'train17' + '.txt'

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

    episode_6 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_6 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_6 = [float(i) for i in avg_100_score]

finally:
    text_results.close()


file_to_read = 'train16' + '.txt'

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

    episode_7 = [float(i) for i in episode]
    score = [float(i) for i in score]
    total_steps_7 = [float(i) for i in total_steps]
    epsilon = [float(i) for i in epsilon]
    avg_100_score_7 = [float(i) for i in avg_100_score]

finally:
    text_results.close()

fig = plt.figure()
plt.grid(True)
ax = plt.axes()
plt.rc('grid', linestyle="-", color='black')
ax.plot(episode_7, avg_100_score_7, color='gray', label = 'line3');
ax.plot(episode_6, avg_100_score_6, color='purple', label = 'line3');
ax.plot(episode_5, avg_100_score_5, color='yellow', label = 'line3');
ax.plot(episode_4, avg_100_score_4, color='orange', label = 'line3');
# ax.plot(episode_3, avg_100_score_3, color='green', label = 'line3');
ax.plot(episode_2, avg_100_score_2, color='red', label = 'line2');
ax.plot(episode_1, avg_100_score_1, color='blue', label='line1');

ax.legend(['Train 16','Train 17', 'Train 18','Train 19','Train 22','Train 24'])


plt.title('Training Comparison')
plt.xlabel('Training Episodes')
plt.ylabel('Score')

plt.show()

print('Done 1')