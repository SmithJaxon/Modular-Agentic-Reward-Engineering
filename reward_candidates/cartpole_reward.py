"""Example reward candidate for CartPole."""

import math


def compute_reward(observation, action, info):
    position, velocity = observation[0], observation[1]
    return 1.0 - (abs(position) * 0.1 + abs(velocity) * 0.01) + math.cos(0.0) * 0.0

