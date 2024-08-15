from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Config:
    train_seed: int 
    eval_seed: int 

    num_arms: int 
    num_train_envs: int 
    num_eval_envs: int 


def skewed_reward_dist(rng: np.random.Generator, num_arms: int, num_envs: int):
    """
    This function creates a set of bandits which have the higher rewards distributed over the even arms.
    """
    num_even_arms = num_arms // 2
    num_odd_arms = num_arms - num_even_arms
    means = np.zeros((num_envs, num_arms))
    # 95% of the time - even arms have higher return
    means[:, ::2] = rng.uniform(size=(num_envs, num_odd_arms), low=0.0, high=0.5)
    means[:, 1::2] = rng.uniform(
        size=(num_envs, num_even_arms),
        low=0.5,
        high=1.0,
    )

    return means


def mixed_skewed_reward_dist(
    rng: np.random.Generator, num_arms: int, num_envs: int, frac_first: float
):
    """
    This function creates two sets of bandits. The first one contains bandits with the higher rewards
    distributed over the odd arms. The second set is similar but with the even arms.
    :param max_arms: controls the maximum amount of arms in all bandits
    :param num_envs: the total amount of envs in two sets combined
    :param frac_first: the relative size of the first set compared to the num_envs
    """
    offset = int(num_envs * frac_first)

    # create the first 'odd' set
    means1 = skewed_reward_dist(rng, num_arms=num_arms, num_envs=offset)
    # create the second 'even' set
    means2 = skewed_reward_dist(rng, num_arms=num_arms, num_envs=num_envs - offset)
    means2 = 1 - means2

    # check that the bandits in the sets are correct
    assert means1[0, 0] < means1[0, 1], (means1[0, 0], means1[0, 1])
    assert means2[0, 0] > means2[0, 1], (means2[0, 0], means2[0, 1])

    # combine the two sets
    means = np.concatenate([means1, means2], axis=0)
    means = rng.permutation(means)

    return means



def make_envs(
    config: Config,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:

    # Higher rewards are more likely to be distributed under
    # odd arms 95% of the time during training
    rng = np.random.default_rng(config.train_seed)

    train_means = mixed_skewed_reward_dist(
        rng,
        num_arms=config.num_arms,
        num_envs=config.num_train_envs,
        frac_first=0.95,
    )

    eval_dists = {}
    rng = np.random.default_rng(config.eval_seed)
    # Higher rewards are more likely to be distributed under
    # even arms 95% of the time during eval
    means = mixed_skewed_reward_dist(
        rng,
        num_arms=config.num_arms,
        num_envs=config.num_eval_envs,
        frac_first=0.05,
    )

    eval_dists["inverse"] = means

    # The rewards are distributed uniformly over the arms
    means = rng.uniform(
        size=(config.num_eval_envs, config.num_arms), low=0.0, high=1.0
    )
    eval_dists["all_new"] = means

    return train_means, eval_dists