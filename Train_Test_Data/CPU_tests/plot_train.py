import matplotlib.pyplot as plt
import numpy as np


class PlotData:
    #file_to_read = 'train05' + '.txt'
    def get_plot_info(file_to_read):
        try:
            text_results = open(file_to_read, "r")
            cnt = 0
            total_steps_10 = []
            total_steps=[]
            last_10_mean = []
            episode_10 = []
            total_episode = []
            total_score = []
            line = text_results.readline()
            while line:
                line_list = line.split()
                #Models without last 100 implementation
                if 'train05' in file_to_read or 'train06' in file_to_read:

                    for i in range(len(line_list)):
                        if 'steps' in line_list[i] and 'total' in line_list[i - 1]:
                            total_steps.append(line_list[i + 1])
                        if 'episode' in line_list[i]:
                            ep = line_list[i + 1]
                            ep = ep.replace(',', '')
                            total_episode.append(ep)

                        if 'score:' in line_list[i] and 'episode' in line_list[i-2]:
                            total_score.append(line_list[i + 1])

                    if "Last 10 mean: " in line:
                        # print("Line {} : {}".format(cnt+1, line.strip()))
                        for i in range(len(line_list)):
                            if 'steps' in line_list[i] and 'total' in line_list[i-1]:
                                total_steps_10.append(line_list[i+1])
                            if 'episode' in line_list[i]:
                                ep = line_list[i+1]
                                ep = ep.replace(',','')
                                episode_10.append(ep)

                            if 'mean' in line_list[i] and '10' in line_list[i - 1]:
                                last_10_mean.append(line_list[i + 1])

                else:

                    for i in range(len(line_list)):
                        if 'steps' in line_list[i] and 'total' in line_list[i - 1]:
                            total_steps.append(line_list[i + 1])
                        if 'episode' in line_list[i]:
                            ep = line_list[i + 1]
                            ep = ep.replace(',', '')
                            total_episode.append(ep)

                        if 'score:' in line_list[i] and 'episode' in line_list[i-2]:
                            total_score.append(line_list[i + 1])

                    if "Last 100 mean: " in line:
                        # print("Line {} : {}".format(cnt+1, line.strip()))
                        line_list = line.split()
                        for i in range(len(line_list)):
                            if 'steps' in line_list[i] and 'total' in line_list[i-1]:
                                total_steps_10.append(line_list[i+1])
                            if 'episode' in line_list[i]:
                                ep = line_list[i+1]
                                ep = ep.replace(',','')
                                episode_10.append(ep)

                            if 'mean' in line_list[i] and '100' in line_list[i - 1]:
                                last_10_mean.append(line_list[i + 1])

                    elif "Last 10 mean: " in line:
                        # print("Line {} : {}".format(cnt+1, line.strip()))
                        line_list = line.split()
                        for i in range(len(line_list)):
                            if 'steps' in line_list[i] and 'total' in line_list[i-1]:
                                total_steps_10.append(line_list[i+1])
                            if 'episode' in line_list[i]:
                                ep = line_list[i+1]
                                ep = ep.replace(',','')
                                episode_10.append(ep)

                            if 'mean' in line_list[i] and '10' in line_list[i - 1]:
                                last_10_mean.append(line_list[i + 1])

                line = text_results.readline()
                cnt += 1

            total_steps_10 = [float(i) for i in total_steps_10]
            episode_10 = [float(i) for i in episode_10]
            last_10_mean = [float(i) for i in last_10_mean]

            total_steps = [float(i) for i in total_steps]
            total_episode = [float(i) for i in total_episode]
            total_score = [float(i) for i in total_score]



        finally:
            text_results.close()

        return total_steps_10, episode_10, last_10_mean, total_steps, total_episode, total_score



