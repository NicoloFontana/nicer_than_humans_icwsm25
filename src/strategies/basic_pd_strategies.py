import numpy as np


def always_coop_strategy():
    return 1


def always_defect_strategy():
    return 0


def rnd_strategy():
    rng = np.random.default_rng()
    choice = rng.choice(np.array(list({1, 0})))
    return choice
