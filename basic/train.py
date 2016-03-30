from catch_ball import CatchBall
from dqn_agent import DQNAgent


if __name__ == "__main__":
    EPSILON = 0.1
    N_EPOCHS = 1000

    env = CatchBall()
    agent = DQNAgent(env.enable_actions)

    win = 0
    for e in range(N_EPOCHS):
        loss = 0.0
        env.reset()
        state_t_1, reward_t, terminal = env.observe()

        while not terminal:
            state_t = state_t_1

            action_t = agent.select_action(state_t, EPSILON)
            env.action(action_t)
            state_t_1, reward_t, terminal = env.observe()

            if reward_t == 1:
                win += 1

            agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)
            agent.experience_replay()

            loss += agent.current_loss

        print("Epoch: {:03d}/{:03d} | Loss: {:.4f} | Win: {:03d}".format(e, N_EPOCHS-1, loss, win))

    agent.save_model()
