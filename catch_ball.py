import os
import numpy as np


class CatchBall:
    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.screen_n_rows = 8
        self.screen_n_cols = 8
        self.player_length = 3
        self.enable_actions = (0, 1, 2)
        self.frame_rate = 5

        # variables
        self.reset()

    def update(self, action):
        """
        action:
            0: do nothing
            1: move left
            2: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_col = max(0, self.player_col - 1)
        elif action == self.enable_actions[2]:
            # move right
            self.player_col = min(self.player_col + 1, self.screen_n_cols - self.player_length)
        else:
            # do nothing
            pass

        # update ball position
        self.ball_row += 1

        # collision detection
        self.reward = 0
        self.terminal = False
        if self.ball_row == self.screen_n_rows - 1:
            self.terminal = True
            if self.player_col <= self.ball_col < self.player_col + self.player_length:
                # catch
                self.reward = 1
            else:
                # drop
                self.reward = -1

    def draw(self):
        # reset screen
        self.screen = np.zeros((self.screen_n_rows, self.screen_n_cols))

        # draw player
        self.screen[self.player_row, self.player_col:self.player_col + self.player_length] = 1

        # draw ball
        self.screen[self.ball_row, self.ball_col] = 1

    def observe(self):
        self.draw()
        return self.screen, self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_row = self.screen_n_rows - 1
        self.player_col = np.random.randint(self.screen_n_cols - self.player_length)

        # reset ball position
        self.ball_row = 0
        self.ball_col = np.random.randint(self.screen_n_cols)

        # reset other variables
        self.reward = 0
        self.terminal = False
