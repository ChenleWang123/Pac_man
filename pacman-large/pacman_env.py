import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

from pacman_core import (
    player,
    ghosts,
    thisGame,
    thisLevel,
    thisFruit,
    TILE_WIDTH,
    TILE_HEIGHT,
    tileID,
    img_Background
)


class PacmanEnv(gym.Env):
    def __init__(self, visual=False):
        super(PacmanEnv, self).__init__()

        self.visual = visual
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        self.last_score = 0
        self.last_positions = []  # ✅ 记录最近位置
        self.display_initialized = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        thisGame.StartNewGame()
        self.last_score = 0
        self.last_positions.clear()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._apply_action(action)

        player.Move()
        for i in range(4):
            ghosts[i].Move()
        thisFruit.Move()

        obs = self._get_obs()
        reward = 0

        # ✅ 吃豆奖励
        score_delta = thisGame.score - self.last_score
        reward += score_delta
        self.last_score = thisGame.score

        # ✅ 幽灵靠近惩罚
        ghost_threat = 0
        for g in ghosts.values():
            if abs(g.x - player.x) < TILE_WIDTH * 1.5 and abs(g.y - player.y) < TILE_HEIGHT * 1.5:
                if g.state == 1 and player.invisible_timer == 0:
                    ghost_threat = 1
                    break
        if ghost_threat:
            reward -= 5

        # ✅ 每步微惩罚
        reward -= 1

        # ✅ 卡角落惩罚
        current_pos = (int(player.x), int(player.y))
        self.last_positions.append(current_pos)
        if len(self.last_positions) > 10:
            self.last_positions.pop(0)
        if self.last_positions.count(current_pos) > 8:
            reward -= 10  # ✅ 长时间不动惩罚

        # ✅ 终止判断
        terminated = False
        if thisGame.mode == 2:  # 死亡
            reward -= 500
            terminated = True
        elif thisGame.mode == 3 or thisGame.lives < 0:
            terminated = True
        elif thisLevel.pellets == 0:  # 通关
            reward += 1000
            terminated = True

        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        if not self.visual:
            return

        pygame.event.pump()

        if not self.display_initialized:
            pygame.display.set_mode(thisGame.screenSize)
            pygame.display.set_caption("Pacman RL")
            self.display_initialized = True

        screen = pygame.display.get_surface()
        screen.blit(img_Background, (0, 0))

        thisLevel.DrawMap()
        for i in range(4):
            ghosts[i].Draw()
        thisFruit.Draw()
        player.Draw()
        thisGame.DrawScore()

        pygame.display.update()

    def close(self):
        pygame.quit()

    def _apply_action(self, action):
        if action == 0:
            player.velX, player.velY = 0, -player.speed
        elif action == 1:
            player.velX, player.velY = 0, player.speed
        elif action == 2:
            player.velX, player.velY = -player.speed, 0
        elif action == 3:
            player.velX, player.velY = player.speed, 0

    def _get_obs(self):
        x = player.x / (thisLevel.lvlWidth * TILE_WIDTH)
        y = player.y / (thisLevel.lvlHeight * TILE_HEIGHT)

        wall_up = int(thisLevel.CheckIfHitWall((player.x, player.y - TILE_HEIGHT), (player.nearestRow, player.nearestCol)))
        wall_down = int(thisLevel.CheckIfHitWall((player.x, player.y + TILE_HEIGHT), (player.nearestRow, player.nearestCol)))
        wall_left = int(thisLevel.CheckIfHitWall((player.x - TILE_WIDTH, player.y), (player.nearestRow, player.nearestCol)))
        wall_right = int(thisLevel.CheckIfHitWall((player.x + TILE_WIDTH, player.y), (player.nearestRow, player.nearestCol)))

        ghost_near = 0
        for g in ghosts.values():
            if abs(g.x - player.x) < TILE_WIDTH * 2 and abs(g.y - player.y) < TILE_HEIGHT * 2:
                ghost_near = 1
                break

        pellet = 1 if thisLevel.GetMapTile((player.nearestRow, player.nearestCol)) in [tileID['pellet'], tileID['pellet-power']] else 0

        obs = np.array([
            x, y,
            wall_up, wall_down, wall_left, wall_right,
            ghost_near, pellet,
            thisGame.lives / 3.0,
            thisGame.ghostTimer / 360.0
        ], dtype=np.float32)

        return obs
