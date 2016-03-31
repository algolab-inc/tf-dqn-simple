from catch_ball import CatchBall
from dqn_agent import DQNAgent


if __name__ == "__main__":
    # variables
    epsilon = 0.1
    n_epochs = 1000

    # environment, agent
    env = CatchBall()
    agent = DQNAgent(env.enable_actions)

    # states
    win = 0

    for e in range(n_epochs):
        # reset
        loss = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            # execute action in environment
            action_t = agent.select_action(state_t, epsilon)
            env.execute_action(action_t)

            # observe environment
            state_t_1, reward_t, terminal = env.observe()

            # store experience
            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            # experience replay
            agent.experience_replay()

            # for log
            loss += agent.current_loss
            if reward_t == 1:
                win += 1

        print("EPOCH: {:03d}/{:03d} | LOSS: {:.4f} | WIN: {:03d}".format(e, n_epochs - 1, loss, win))

    agent.save_model()
