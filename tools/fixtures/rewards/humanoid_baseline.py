def reward(observation, x_velocity, reward_alive, reward_quadctrl, terminated, truncated):
    if terminated or truncated:
        return float(reward_alive) - float(reward_quadctrl)

    del observation
    velocity_bonus = float(x_velocity)
    upright_bonus = float(reward_alive)
    control_penalty = float(reward_quadctrl)
    return velocity_bonus + upright_bonus - control_penalty
