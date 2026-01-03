"""
环境Wrapper模块 - 所有算法共享
包含图像预处理、裁剪和帧堆叠功能
"""

import gymnasium as gym
from collections import deque
import numpy as np
import cv2
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium import spaces

class SOTARewardWrapper(gym.Wrapper):
    """
    SOTA Reward Shaping with Track Geometry
    Based on CarRacing-PPO-SOTA implementation
    Features:
    - Frame Skip (default 4)
    - Effective Progress Reward
    - Centerline Reward
    - Anti-Wobble & Smoothness Penalty
    - Pedal Conflict Penalty
    """
    def __init__(self, env, frame_skip=4):
        super().__init__(env)
        self.frame_skip = frame_skip
        
        # Reward coefficients
        self.progress_coef = 0.1
        self.centering_coef = 0.5
        self.offtrack_penalty = 10.0
        self.pedal_conflict_penalty = 0.5
        
        # Track state
        self.prev_steering = 0.0
        self.track_width = 10.0  # CarRacing track width approximation
        self.cached_closest_idx = 0
        self.step_count = 0
        
        # Keep original observation space (RGB 96x96)
        # Subsequent wrappers (GrayScale, Crop) will handle obs transformation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_steering = 0.0
        self.cached_closest_idx = 0
        self.step_count = 0
        return obs, info

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = None
        
        # Frame skip loop
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            last_obs = obs
            if terminated or truncated:
                break
        
        # --- SOTA Reward Shaping ---
        shaped_reward = total_reward
        
        try:
            # Get car state
            car = self.env.unwrapped.car
            if car is None:
                return last_obs, shaped_reward, terminated, truncated, info
            
            car_pos = np.array([car.hull.position.x, car.hull.position.y])
            car_vel = np.array([car.hull.linearVelocity.x, car.hull.linearVelocity.y])
            
            # Get track points
            track = self.env.unwrapped.track
            if not track:
                return last_obs, shaped_reward, terminated, truncated, info
            
            track_points = np.array([[t[2], t[3]] for t in track])
            
            # Performance optimization: search nearby only
            self.step_count += 1
            if self.step_count % 3 == 0:
                search_start = max(0, self.cached_closest_idx - 30)
                search_end = min(len(track_points), self.cached_closest_idx + 30)
                search_indices = list(range(search_start, search_end))
                
                if search_indices:
                    distances = np.linalg.norm(track_points[search_indices] - car_pos, axis=1)
                    local_min_idx = np.argmin(distances)
                    self.cached_closest_idx = search_indices[local_min_idx]
            
            closest_idx = self.cached_closest_idx
            dist_to_center = np.linalg.norm(track_points[closest_idx] - car_pos)
            
            # Track direction
            p_current = track_points[closest_idx]
            p_next = track_points[(closest_idx + 1) % len(track_points)]
            track_dir = p_next - p_current
            track_dir_norm = np.linalg.norm(track_dir)
            if track_dir_norm > 1e-6:
                track_dir = track_dir / track_dir_norm
            else:
                track_dir = np.array([1.0, 0.0])
            
            # A. Effective Progress Reward
            progress_speed = np.dot(car_vel, track_dir)
            reward_progress = progress_speed * self.progress_coef * self.frame_skip
            
            # B. Centerline Reward (Gaussian)
            reward_centering = np.exp(-(dist_to_center / (self.track_width / 4))**2) * self.centering_coef * self.frame_skip
            
            # C. Balanced Anti-Wobble System
            speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
            steer = abs(action[0])
            
            # Layer 1: High-Speed Micro-Wobble Suppression
            wobbly_penalty = 0
            if speed > 40.0 and 0.02 < steer < 0.2:
                penalty = steer * speed * 0.02
                wobbly_penalty = min(penalty, 0.5)
            
            # Layer 2: L2 Smoothness Penalty
            steering_diff = action[0] - self.prev_steering
            if abs(steering_diff) > 0.3:
                smoothness_penalty = (steering_diff ** 2) * 0.5
            else:
                smoothness_penalty = 0
            
            steering_penalty = wobbly_penalty + smoothness_penalty
            self.prev_steering = action[0]
            
            # D. Pedal Conflict Penalty
            pedal_penalty = 0
            if action[1] > 0.3 and action[2] > 0.3:
                pedal_penalty = self.pedal_conflict_penalty * action[1] * action[2]
            
            # E. Off-track termination logic (optional, keeping strict punishment)
            if dist_to_center > self.track_width * 1.5:
                shaped_reward -= self.offtrack_penalty
                terminated = True
            else:
                shaped_reward += reward_progress + reward_centering - steering_penalty - pedal_penalty
                
        except Exception:
            # Fallback to original reward
            pass
            
        return last_obs, shaped_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """将RGB图像转换为灰度图"""
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        if observation.ndim == 3:
            return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation


class CropObservation(gym.ObservationWrapper):
    """裁剪图像到84x84，去除底部状态条"""
    def __init__(self, env, crop_bottom=12):
        super().__init__(env)
        self.crop_bottom = crop_bottom
        obs_shape = env.observation_space.shape
        
        # Target: 84x84
        # Original: 96x96 (after grayscale)
        new_h = obs_shape[0] - crop_bottom  # 84
        new_w = 84
        self.crop_x = (obs_shape[1] - new_w) // 2
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(new_h, new_w), dtype=env.observation_space.dtype
        )

    def observation(self, observation):
        # Crop bottom (remove status bar)
        obs = observation[:-self.crop_bottom, :]
        # Crop sides to 84
        obs = obs[:, self.crop_x:-self.crop_x]
        return obs


class FrameStack(gym.Wrapper):
    """堆叠最近的k帧作为观测"""
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return np.array(self.frames), info
    
    def step(self, action):
        # #region agent log
        # (Logging code removed for cleaner diff)
        # #endregion
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return np.array(self.frames), reward, terminated, truncated, info


class StrictTimeLimit(gym.Wrapper):
    """强制时间限制，防止僵尸车和死循环"""
    def __init__(self, env, max_steps=1000):
        super().__init__(env)
        self.max_steps = max_steps
        self._elapsed_steps = 0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


def make_env(env_name, seed=None, continuous=True, max_steps=1000):
    """
    创建环境工厂函数
    
    Args:
        env_name: 环境名称
        seed: 随机种子
        continuous: True表示连续动作空间，False表示离散动作空间
        max_steps: 最大步数限制
    
    Returns:
        环境工厂函数
    """
    def _thunk():
        env = gym.make(env_name, continuous=continuous)
        # 强制时间限制
        env = StrictTimeLimit(env, max_steps=max_steps)
        # SOTA Improvements: Reward Shaping + Frame Skip
        env = SOTARewardWrapper(env, frame_skip=4)
        # Original Refactoring Preprocessing: Gray -> Crop (84x84) -> Stack
        env = GrayScaleObservation(env)
        env = CropObservation(env, crop_bottom=12)
        env = FrameStack(env, num_stack=4)
        env = RecordEpisodeStatistics(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _thunk
