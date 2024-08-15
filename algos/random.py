import numpy as np

class Random:
    def __init__(self, num_arms: int):
        self.num_arms = num_arms

    def select_arm(self):
        act = np.random.randint(low=0, high=self.num_arms)

        return act

    def update_state(self, arm, reward):
        pass