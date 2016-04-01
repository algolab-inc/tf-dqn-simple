import pygame
from pygame.locals import KEYDOWN, K_ESCAPE
import numpy as np
from catch_ball import CatchBall


if __name__ == "__main__":
    # environment
    env = CatchBall()
    clock = pygame.time.Clock()

    # states
    win, lose = 0, 0
    playing = True

    while playing:
        # reset
        env.reset()
        _, reward, terminal = env.observe()
        pygame.display.flip()

        while not terminal:
            # frame
            clock.tick(env.frame_rate)
            env.execute_action(np.argmax(pygame.key.get_pressed()))
            _, reward, terminal = env.observe()
            pygame.display.flip()

            # quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    playing = False

            # for log
            if reward == 1:
                win += 1
            elif reward == -1:
                lose += 1

        print("WIN: {:03d}/{:03d} ({:.1f}%)".format(win, win + lose, 100 * win / (win + lose)))

    pygame.quit()
