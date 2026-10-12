"""
env_wrapper.py — Super Contra Environment Wrapper v4.0
======================================================
Game    : Super Contra (NES) via stable-retro 0.9.9
Policy  : CnnPolicy (PPO)
Obs     : (84, 84, 1) grayscale uint8

RAM VARIABLES (from data.json):
    lives          addr:83    — Player lives (0 = game over)
    current_level  addr:80    — Current level number
    game_mode      addr:24    — 0=title/gameover, >0=active
    level_type     addr:64    — 0=outdoor scroll, 1=indoor 3D depth
    screen_x       addr:204   — X within scroll page (0-255)
    scroll_x       addr:253   — Scroll page (world_x = scroll_x*256 + screen_x)
    screen_y       addr:206   — Player Y
    scroll_y       addr:252   — Vertical scroll (indoor)
    indoor_depth   addr:146   — Depth in 3D indoor levels
    score_low      addr:2018  — Score lower byte
    score_mid      addr:2019  — Score upper byte
    boss_dead      addr:59    — 1 = boss killed (level clear!)
    weapon         addr:170   — Current weapon (0=default, higher=powerup)

ACTION MAP (8 discrete):
    0 : Forced Right    — Right + 0.5 bonus (anti-freeze)
    1 : Right           — Normal rightward movement
    2 : Left            — PENALIZED (escalating when stagnating)
    3 : Down/Crouch     — PENALIZED (anti-camping)
    4 : Up              — Aim upward
    5 : Right + Fire    — Move and shoot (+0.1 bribe)
    6 : Jump            — Jump only (+1.0 reflex if moving)
    7 : Jump + Fire     — Jump and shoot (+0.1 bribe, +1.0 reflex)

NES BUTTONS: B(0)|None(1)|Select(2)|Start(3)|Up(4)|Down(5)|Left(6)|Right(7)|A(8)
"""

import gymnasium as gym
import numpy as np
import cv2

LEVEL_OUTDOOR = 0
LEVEL_INDOOR = 1


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeat each action for N frames and accumulate reward.
    Standard for NES/Atari RL — gives ~4x training speedup.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class SuperContraWrapper(gym.Wrapper):
    """
    Production reward-shaping wrapper for Super Contra (NES).

    Features:
      - Built-in FrameSkipWrapper (4 frames per action)
      - Indoor/outdoor level-aware progress tracking
      - Boss kill detection (+2000 one-time)
      - Weapon powerup tracking (+200)
      - Escalating stagnation penalties
      - Combat rewards (score-based kill detection)
      - Death penalty (-600 per life)
      - Game over penalty (-500, hard terminate)
    """

    def __init__(self, env: gym.Env, frame_skip: int = 4):
        env = FrameSkipWrapper(env, skip=frame_skip)
        super().__init__(env)

        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        # Episode state trackers
        self.max_world_x = 0
        self.max_indoor_depth = 0
        self.prev_total_score = 0
        self.prev_lives = 3
        self.prev_boss_dead = 0
        self.prev_weapon = 0
        self.stagnation_timer = 0

    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        """RGB (224,256,3) -> Grayscale (84,84,1). INTER_NEAREST for GPU speed."""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_NEAREST)
        return np.expand_dims(resized, axis=-1)

    def reset(self, **kwargs):
        """Reset all trackers and sync RAM from save state."""
        self.max_world_x = 0
        self.max_indoor_depth = 0
        self.prev_total_score = 0
        self.prev_lives = 3
        self.prev_boss_dead = 0
        self.prev_weapon = 0
        self.stagnation_timer = 0

        result = self.env.reset(**kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})

        # RAM SYNC: save-state needs 1 no-op step to populate info dict
        if not info or "lives" not in info:
            obs, _, _, _, info = self.env.step([0] * 9)

        self.prev_lives = info.get("lives", 3)
        self.prev_weapon = info.get("weapon", 0)
        return self.preprocess(obs), info

    def step(self, action: int):
        retro_act = [0] * 9
        reward = 0.0

        # --- Action -> NES button mapping ---
        if action == 0:
            retro_act[7] = 1
            reward += 0.5  # Forced Right + bonus
        elif action == 1:
            retro_act[7] = 1  # Right
        elif action == 2:
            retro_act[6] = 1  # Left (penalized)
        elif action == 3:
            retro_act[5] = 1  # Down/Crouch (penalized)
        elif action == 4:
            retro_act[4] = 1  # Up
        elif action == 5:
            retro_act[7], retro_act[0] = 1, 1  # Right + Fire
        elif action == 6:
            retro_act[8] = 1  # Jump
        elif action == 7:
            retro_act[8], retro_act[0] = 1, 1  # Jump + Fire

        obs, _, terminated, truncated, info = self.env.step(retro_act)

        # --- Read RAM variables ---
        curr_lives = info.get("lives", 0)
        game_mode = info.get("game_mode", 0)
        level_type = info.get("level_type", LEVEL_OUTDOOR)
        screen_x = int(info.get("screen_x", 0))
        scroll_x = int(info.get("scroll_x", 0))
        indoor_depth = int(info.get("indoor_depth", 0))
        boss_dead = int(info.get("boss_dead", 0))
        curr_weapon = int(info.get("weapon", 0))
        score_low = int(info.get("score_low", 0))
        score_mid = int(info.get("score_mid", 0))

        # --- Game over / title screen -> hard terminate ---
        if curr_lives == 0 or game_mode == 0:
            return self.preprocess(obs), -500.0, True, truncated, info

        # --- Behavioural penalties ---
        if action == 2:
            reward -= 20.0 if self.stagnation_timer > 30 else 10.0  # Left
        if action == 3:
            reward -= 5.0  # Crouch
        if action in (5, 7):
            reward += 0.1  # Fire bribe
        if action in (6, 7) and self.stagnation_timer < 5:
            reward += 1.0  # Jump reflex bonus

        # --- Progress tracking ---
        made_progress = False

        if level_type == LEVEL_OUTDOOR:
            curr_world_x = (scroll_x * 256) + screen_x
            if curr_world_x > self.max_world_x:
                # Distance guides forward progress, but is now a MINOR signal
                # (was *100) so it no longer drowns out the boss/clear reward.
                reward += (curr_world_x - self.max_world_x) * 20.0
                self.max_world_x = curr_world_x
                made_progress = True
        else:
            # Indoor 3D level — use depth instead of scroll_x
            if indoor_depth > self.max_indoor_depth:
                reward += (indoor_depth - self.max_indoor_depth) * 20.0
                self.max_indoor_depth = indoor_depth
                made_progress = True

        # --- Combat reward (kill detection via score) ---
        score = (score_mid * 100) + score_low
        if score > self.prev_total_score:
            reward += (score - self.prev_total_score) * 250.0
            self.prev_total_score = score
            made_progress = True

        # --- Boss kill bonus (one-time) — THE GOAL: clearing Level 1 ---
        # Made huge so killing the boss dominates all other reward. This is
        # what teaches the agent to WIN, not just walk far.
        if boss_dead == 1 and self.prev_boss_dead == 0:
            reward += 50000.0
            self.prev_boss_dead = 1

        # --- Weapon powerup ---
        if curr_weapon > self.prev_weapon:
            reward += 200.0
            self.prev_weapon = curr_weapon
        elif curr_weapon < self.prev_weapon:
            self.prev_weapon = curr_weapon  # Died, lost weapon

        # --- Stagnation ---
        if made_progress:
            self.stagnation_timer = 0
        else:
            self.stagnation_timer += 1

        if self.stagnation_timer > 90:
            reward -= 10.0

        # --- Death penalty ---
        if curr_lives < self.prev_lives:
            reward -= 600.0
            self.prev_lives = curr_lives

        return self.preprocess(obs), reward, terminated, truncated, info
