from __future__ import division

import argparse

import numpy as np
import pygame

from catch_ball import CatchBall  # NOQA
from dqn_agent import DQNAgent
from pin_ball import PinBall  # NOQA


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="CatchBall")
    parser.add_argument("-m", "--model_path")
    args = parser.parse_args()

    # environment, agent
    env = eval(args.environment)()
    agent = DQNAgent(env.enable_actions, env.name)
    agent.load_model(args.model_path)
    clock = pygame.time.Clock()

    # variables
    testing = True

    while testing:
        # reset
        env.reset()
        screen_t_1, reward_t, terminal = env.observe()
        screen_t_1 = agent.process_image(screen_t_1)
        state_t_1 = np.stack((screen_t_1, screen_t_1, screen_t_1, screen_t_1), axis=2)

        while not terminal:
            clock.tick(env.frame_rate)
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, 0.0)
            env.execute_action(action_t)

            # observe environment
            screen_t_1, reward_t, terminal = env.observe()
            screen_t_1 = agent.process_image(screen_t_1)
            screen_t_1 = np.reshape(screen_t_1, (84, 84, 1))
            state_t_1 = np.append(screen_t_1, state_t[:, :, :3], axis=2)

    pygame.quit()
