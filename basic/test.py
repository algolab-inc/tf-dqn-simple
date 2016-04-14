from __future__ import division

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from catch_ball import CatchBall
from dqn_agent import DQNAgent


def init():
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


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

    # animate
    img.set_array(state_t_1)
    plt.axis("off")
    return img,


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-s", "--save", dest="save", action="store_true")
    parser.set_defaults(save=False)
    args = parser.parse_args()

    # environmet, agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions, env.name)
    agent.load_model(args.model_path)

    # variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    # animate
    fig = plt.figure(figsize=(env.screen_n_rows / 2, env.screen_n_cols / 2))
    fig.canvas.set_window_title("{}-{}".format(env.name, agent.name))
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=(1000 / env.frame_rate), blit=True)

    if args.save:
        # save animation (requires ImageMagick)
        ani_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "tmp", "demo-{}.gif".format(env.name))
        ani.save(ani_path, writer="imagemagick", fps=env.frame_rate)
    else:
        # show animation
        plt.show()
