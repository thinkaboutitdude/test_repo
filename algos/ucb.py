import numpy as np

def calc_delta(alpha, num_pulled, t):
    if num_pulled > 0:
        return np.sqrt(alpha * np.log(t) / num_pulled)
    else:
        return 1e20


def calc_av_reward(sum_rewards, num_pulled):
    if num_pulled > 0:
        return sum_rewards / num_pulled
    else:
        return 0


calc_delta = np.vectorize(calc_delta)
calc_av_reward = np.vectorize(calc_av_reward)


class UCB:
    def __init__(self, alpha: float, num_arms: int):
        self.num_arms = num_arms

        self.num_pulled = np.zeros(num_arms)
        self.sum_rewards = np.zeros(num_arms)
        self.t = 0
        self.alpha = alpha


    def select_arm(self):
        delta = calc_delta(self.alpha, self.num_pulled, self.t)
        av_reward = calc_av_reward(self.sum_rewards, self.num_pulled)

        value = av_reward + delta
        one_max_arm = np.argmax(value)

        return one_max_arm

    def update_state(self, arm, reward):
        self.num_pulled[arm] += 1
        self.sum_rewards[arm] += reward
        self.t += 1