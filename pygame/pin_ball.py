import os

import numpy as np
import pygame
from pygame.locals import K_LEFT, K_RIGHT, Rect


class PinBall:
    def __init__(self):
        # parameters
        self.name = os.path.splitext(os.path.basename(__file__))[0]
        self.unit_size = 8
        self.screen_width = 32 * self.unit_size
        self.screen_height = 32 * self.unit_size
        self.player_width = 8 * self.unit_size
        self.player_height = 2 * self.unit_size
        self.player_dx = 1 * self.unit_size
        self.ball_round = 2 * self.unit_size
        self.ball_dx = 1 * self.unit_size
        self.ball_dy = 1 * self.unit_size
        self.enable_actions = (0, K_LEFT, K_RIGHT)
        self.frame_rate = 30

        # pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))

        # variables
        self.reset()

    def update(self, action):
        """
        action:
            0:       do nothing
            K_LEFT:  move left
            L_RIGHT: move right
        """
        # update player position
        if action == self.enable_actions[1]:
            # move left
            self.player_x = max(0, self.player_x - self.player_dx)
        elif action == self.enable_actions[2]:
            # move right
            self.player_x = min(self.player_x + self.player_dx, self.screen_width - self.player_width)
        else:
            # do nothing
            pass

        # update ball position
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # collision detection
        self.reward = 0
        self.terminal = False
        if self.ball_y + self.ball_round == self.screen_height - self.player_height and self.ball_dy > 0:
            if self.player_x <= self.ball_x <= self.player_x + self.player_width:
                self.reward = 1
                self.ball_dy *= -1
            else:
                self.reward = -1
                self.terminal = True

        # update ball direction
        if not self.terminal:
            if self.ball_y - self.ball_round == 0 and self.ball_dy < 0:
                self.ball_dy *= -1

            if self.ball_x - self.ball_round == 0 and self.ball_dx < 0:
                self.ball_dx *= -1

            if self.ball_x + self.ball_round == self.screen_width and self.ball_dx > 0:
                self.ball_dx *= -1

    def draw(self):
        # reset screen
        self.screen.fill((0, 0, 0))

        # draw player
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            Rect(self.player_x, self.player_y, self.player_width, self.player_height))

        # draw ball
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (self.ball_x, self.ball_y),
            self.ball_round)

        # update display
        pygame.display.update()

    def observe(self):
        self.draw()
        return pygame.surfarray.array3d(pygame.display.get_surface()), self.reward, self.terminal

    def execute_action(self, action):
        self.update(action)

    def reset(self):
        # reset player position
        self.player_x = np.random.randint((self.screen_width - self.player_width) / self.unit_size) * self.unit_size
        self.player_y = self.screen_height - self.player_height

        # reset ball position
        self.ball_x = np.random.randint(
            self.ball_round + abs(self.ball_dx) / self.unit_size,
            (self.screen_width - self.ball_round - abs(self.ball_dx)) / self.unit_size + 1) \
            * self.unit_size
        self.ball_y = self.ball_round

        # reset other variables
        self.reward = 0
        self.terminal = False
