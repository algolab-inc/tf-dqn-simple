import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE

from catch_ball import CatchBall


if __name__ == "__main__":
    # environment
    env = CatchBall()
    clock = pygame.time.Clock()
    pygame.display.set_caption(env.name)

    # values
    win, lose = 0, 0
    playing = True

    while playing:
        # reset
        env.reset()
        state, reward, terminal = env.observe()

        while not terminal:
            # stop playing
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    playing = False

            # update frame
            clock.tick(env.frame_rate)
            env.execute_action(np.argmax(pygame.key.get_pressed()))
            state, reward, terminal = env.observe()

            # for log
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1

        print("WIN: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win / (win + lose)))

    pygame.quit()
