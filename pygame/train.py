import argparse

import cv2
import numpy as np

from catch_ball import CatchBall  # NOQA
from dqn_agent import DQNAgent
from pin_ball import PinBall  # NOQA


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="CatchBall")
    args = parser.parse_args()

    # environment, agent
    env = eval(args.environment)()
    agent = DQNAgent(env.enable_actions, env.name)
    global_step = agent.load_model()

    while True:
        # reset
        env.reset()
        screen_t_1, reward_t, terminal = env.observe()
        screen_t_1 = cv2.cvtColor(cv2.resize(screen_t_1, (84, 84)), cv2.COLOR_BGR2GRAY)
        _, screen_t_1 = cv2.threshold(screen_t_1, 1, 255, cv2.THRESH_BINARY)
        state_t_1 = np.stack((screen_t_1, screen_t_1, screen_t_1, screen_t_1), axis=2)

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            # TODO: frame skipping
            epsilon_t = agent.current_epsilon(global_step)
            action_t = agent.select_action(state_t, epsilon_t)
            env.execute_action(action_t)

            # observe environment
            screen_t_1, reward_t, terminal = env.observe()
            screen_t_1 = cv2.cvtColor(cv2.resize(screen_t_1, (84, 84)), cv2.COLOR_BGR2GRAY)
            _, screen_t_1 = cv2.threshold(screen_t_1, 1, 255, cv2.THRESH_BINARY)
            screen_t_1 = np.reshape(screen_t_1, (84, 84, 1))
            state_t_1 = np.append(screen_t_1, state_t[:, :, :3], axis=2)

            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            # experience replay
            # TODO: check target_update_frequency
            if global_step > agent.replay_start_size and global_step % agent.target_update_frequency == 0:
                agent.experience_replay()

            # logging
            print("STEP: {:d} | EPSILON: {:.4f} | ACTION: {:>3} | REWARD: {:>2} | LOSS: {:.4f} | Q_MAX: {:.4f}".format(
                global_step, epsilon_t, action_t, reward_t, agent.current_loss, np.max(agent.Q_values(state_t))))

            # update step
            global_step += 1

            # save model
            if global_step % 1000 == 0:
                agent.save_model(global_step)
