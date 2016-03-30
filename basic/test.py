from __future__ import division

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from catch_ball import CatchBall
from dqn_agent import DQNAgent


def animate(step):
    global win, lose
    global state_t_1, reward_t, terminal

    if terminal:
        env.reset()
        if reward_t == 1:
            win += 1
        elif reward_t == -1:
            lose += 1
        print("Win: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win/(win+lose)))
    else:
        state_t = state_t_1
        action_t = agent.select_action(state_t, 0.0)
        env.action(action_t)

    state_t_1, reward_t, terminal = env.observe()
    img.set_array(state_t_1)

    return img,


if __name__ == "__main__":
    # initialize environmet and agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions)
    agent.load_model()

    # initialize variables
    win, lose = 0, 0
    state_t_1, reward_t, terminal = env.observe()

    # animate
    fig = plt.figure()
    img = plt.imshow(state_t_1, interpolation="none", cmap="gray")
    ani = animation.FuncAnimation(fig, animate, interval=200, blit=True)
    plt.axis("off")
    plt.show()
