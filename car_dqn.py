from dqn import DQN
import numpy as np
from skimage import color
import itertools as it



class CarRacingDQN(DQN):
    #CarRacing specific part of the DQN-agent


    # ** is used for unpacking the model configurations
    def __init__(self, max_negative_rewards=100, **model_config):

        #Define all 12 actions possible:
        # all_actions = np.array([k for k in it.product([-1, 0, 1], [1, 0], [0.5, 0])])

        #selected 5 actions:
        all_actions = np.array([[-1, 0, 0],  [0, 1, 0], [0, 0, 0.5], [0, 0, 0],[1, 0, 0]])

        #Set self parameters
        super().__init__(
            action_map=all_actions,
            pic_size=(96, 96),
            **model_config
        )

        self.gas_actions = np.array([a[1] == 1 and a[2] == 0 for a in all_actions])
        self.break_actions = np.array([a[2] > 0 for a in all_actions])
        self.n_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_rewards = max_negative_rewards



    def get_random_action(self):
# give priority to acceleration actions
        action_weights = 14.0 * self.gas_actions + 1.0
        action_weights /= np.sum(action_weights)

        return np.random.choice(self.dim_actions, p=action_weights)

    def check_early_stop(self, reward, totalreward, fie):
        if reward < 0 and fie > 10:
            self.neg_reward_counter += 1
            done = (self.neg_reward_counter > self.max_neg_rewards)

            if done and totalreward <= 500:
                punishment = -20.0
            else:
                punishment = 0.0
            if done:
                self.neg_reward_counter = 0

            return done, punishment
        else:
            self.neg_reward_counter = 0
            return False, 0.0