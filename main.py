from __future__ import absolute_import, division, print_function, unicode_literals
from car_dqn import CarRacingDQN
import os
import tensorflow as tf
import gym
import _thread
import re
import sys
import numpy as np

#Ensure its running og GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

load_checkpoint = True
checkpoint_path = "data/checkpoints/train24"
train_episodes = 15000
save_freq_episodes = train_episodes/100 ###############333
finished = False
opendir = checkpoint_path + '.txt'
text_results = open(opendir, "w")
render = False

frame_skip = 3 #frame_skip number n. model is trained n to n times only
model_config = dict(
    min_epsilon=0.05,
    max_negative_rewards=8,
    min_experience_size=int(100), #######################################33
    experience_capacity=int(150000),
    num_frame_stack=frame_skip,
    frame_skip=frame_skip,
    train_freq=frame_skip,
    batchsize=64,
    epsilon_decay_steps=int(100000),
    target_network_update_freq=int(1000), #Updates the target network every 10000 global steps by copying them from the prediction network to the target network
    gamma=0.95,
    render=False,
)

dqn_scores = []
eps_history = []
avg_score_all = [0]

env = gym.make('CarRacing-v0',  verbose=False)

tf.compat.v1.reset_default_graph

dqn_agent = CarRacingDQN(env=env, **model_config)
dqn_agent.build_graph()
sess = tf.InteractiveSession()
dqn_agent.session = sess

#Initialize save checkpoints
saver = tf.train.Saver(max_to_keep=1000) #max number of checkpoints = 500
#Choice to load checkpoints
if load_checkpoint:
    train_episodes = 150
    save_freq_episodes = 0
    print("loading the latest checkpoint from %s" % checkpoint_path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    assert ckpt, "checkpoint path %s not found" % checkpoint_path
    global_counter = int(re.findall("-(\d+)$", ckpt.model_checkpoint_path)[0])
    saver.restore(sess, ckpt.model_checkpoint_path)
    dqn_agent.global_counter = global_counter
    render = True
else:
    if checkpoint_path is not None:
        assert not os.path.exists(checkpoint_path), \
            "checkpoint path already exists but load_checkpoint is false"

    tf.global_variables_initializer().run()


def save_checkpoint():
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    p = os.path.join(checkpoint_path, "m.ckpt")
    saver.save(sess, p, dqn_agent.global_counter)
    print("saved to %s - %d" % (p, dqn_agent.global_counter))


def one_episode(eps_history,dqn_scores,avg_score_all,render,load_checkpoint):
    score, reward, frames, epsilon = dqn_agent.play_episode(render, load_checkpoint)

    eps_history.append(epsilon)

    dqn_scores.append(score)
    i = dqn_agent.episode_counter
    avg_score = np.mean(dqn_scores[max(0, i - 100):(i + 1)])
    avg_score_all.append(avg_score)
    max_avg_score = max(avg_score_all)
    if avg_score >= max_avg_score:
        new_max = ' => New HighScore! <= '
        highscore = True
    else:
        new_max = ''
        highscore = False



    strm = ("#> episode: %i | score: %.2f | total steps: %i | epsilon: %.5f | average 100 score: %.2f" %
            (i, score, dqn_agent.global_counter, epsilon, avg_score))

    print(strm + new_max)

    text_results = open(opendir, "a")
    text_results.write(strm + new_max + '\n')
    text_results.close()

    if not load_checkpoint:
        save_cond = (
            dqn_agent.episode_counter % save_freq_episodes == 0
            and checkpoint_path is not None
            and dqn_agent.do_training
        )
        if save_cond or (highscore and dqn_agent.episode_counter > 100):
            save_checkpoint()

    return eps_history,dqn_scores,avg_score_all

def input_thread(list):
    input("...enter to stop after current episode\n")
    list.append("OK")


def main_loop(eps_history,dqn_scores,avg_score_all,render,load_checkpoint):
    #call training loop
    list = []
    _thread.start_new_thread(input_thread, (list,))
    while True:
        if list:
            break
        if dqn_agent.do_training and dqn_agent.episode_counter >= train_episodes:

            break
        eps_history,dqn_scores,avg_score_all = one_episode(eps_history,dqn_scores,avg_score_all,render,load_checkpoint)

    print("done")
    text_results.close()
    exit()
    return eps_history,dqn_scores,avg_score_all


if train_episodes > 0 and dqn_agent.episode_counter < train_episodes and not load_checkpoint :
    print("now training... you can early stop with enter...")
    print("##########")
    sys.stdout.flush()
    main_loop(eps_history,dqn_scores,avg_score_all,render,load_checkpoint)
    save_checkpoint()
    print("ok training done")

else:
    print("now just playing...")
    sys.stdout.flush()
    main_loop(eps_history,dqn_scores,avg_score_all,render,load_checkpoint)


