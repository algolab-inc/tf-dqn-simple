import argparse

import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE

from catch_ball import CatchBall  # NOQA
from pin_ball import PinBall  # NOQA


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--environment", default="PinBall")
    args = parser.parse_args()

    # environment
    env = eval(args.environment)()
    clock = pygame.time.Clock()
    pygame.display.set_caption(env.name)

    # variables
    playing = True

    while playing:
        # reset
        env.reset()
        _, _, terminal = env.observe()

        while not terminal:
            # stop playing
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    playing = False

            # update frame
            clock.tick(env.frame_rate)
            env.execute_action(np.argmax(pygame.key.get_pressed()))
            _, _, terminal = env.observe()

    pygame.quit()
