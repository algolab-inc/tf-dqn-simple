from __future__ import division

import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from catch_ball import CatchBall
from dqn_agent import DQNAgent


def animate(step):
    global win, lose
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()

        # for log
        if reward_t == 1:
            win += 1
        elif reward_t == -1:
            lose += 1

        print("WIN: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win / (win + lose)))
    else:
        state_t = state_t_1

        # execute action in environment
        action_t = agent.select_action(state_t, 0.0)
        env.execute_action(action_t)

    # observe environment
    state_t_1, reward_t, terminal = env.observe()

    # animate image
    img.set_array(state_t_1)
    return img,


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    args = parser.parse_args()

    # environmet, agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions)
    agent.load_model(args.model_path)

    # states
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    # animate
    fig = plt.figure()
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, interval=(1000 / env.frame_rate), blit=True)
    plt.axis("off")
    plt.show()
