from plot_train import PlotData
import matplotlib.pyplot as plt
import numpy as np


get_plot_info = PlotData.get_plot_info

file_to_read = 'train14' + '.txt'

total_steps_10, episode_10, last_10_mean, total_steps, total_episode, total_score = get_plot_info(file_to_read)
ymax = []
ymin = []
for i in range(len(total_steps_10)):
    y = total_score[0+i*10:10+i*10]
    ymx = max(y)
    ymax.append(ymx)
    ymn = min(y)
    ymin.append(ymn)


fig = plt.figure()
ax = plt.axes()
ax.plot(total_steps_10, last_10_mean, color='#CC4F1B');
plt.fill_between(total_steps_10, ymin, ymax,
                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
plt.title('Figure from ' + file_to_read)
plt.xlabel('Total Time steps')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# fig = plt.figure()
# ax = plt.axes()
# ax.plot(episode_10, last_10_mean, color='#CC4F1B');
# plt.fill_between(episode_10, ymin, ymax,
#                  alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
# plt.title('Figure from ' + file_to_read)
# plt.xlabel('Episodes')
# plt.ylabel('Score')
# plt.show()
