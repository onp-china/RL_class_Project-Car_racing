
"""
Continuous Action Environment Wrapper for CarRacing
连续动作空间的环境包装器 - 用于 PPO, DDPG, SAC 等算法
"""
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

class ContinuousCarRacingWrapper(gym.Wrapper):
    """
    CarRacing Wrapper for Continuous Control Algorithms (PPO, DDPG, SAC)
    - Keeps continuous action space (no discretization)
    - Preprocesses observation (GrayScale, Normalize, Frame Stacking)
    - Optional: Frame skip for faster training
    """
    def __init__(self, env, frame_stack=4, frame_skip=2):
        super().__init__(env)
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip  # Action repeat for speed
        
        # --- Keep Continuous Action Space ---
        # Action: [steer, gas, brake]
        # steer: -1.0 (left) to 1.0 (right)
        # gas: 0.0 to 1.0
        # brake: 0.0 to 1.0
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # --- Observation Space ---
        # (frame_stack, 96, 96) Grayscale, Normalized, Stacked
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.frame_stack, 96, 96), dtype=np.float32
        )
        
        # Frame buffer for stacking
        self.frames = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_frame = self._process_obs(obs)
        
        # Initialize frame buffer with the first frame repeated
        self.frames = np.repeat(processed_frame, self.frame_stack, axis=0)
        
        return self.frames.copy(), info

    def step(self, action):
        """
        Execute action with frame skip
        action: [steer, gas, brake] - continuous values
        """
        # Clip action to valid range (safety)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Frame skip: repeat action for multiple frames
        total_reward = 0.0
        terminated = False
        truncated = False
        
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Process observation
        processed_frame = self._process_obs(obs)
        
        # Update frame stack: shift left and append new frame
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = processed_frame[0]
        
        return self.frames.copy(), total_reward, terminated, truncated, info

    def _process_obs(self, obs):
        """
        Input: (96, 96, 3) uint8 RGB
        Output: (1, 96, 96) float32 Grayscale [0, 1]
        """
        # Convert to Grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Normalize to [0, 1]
        normalized = gray.astype(np.float32) / 255.0
        
        # Add channel dimension (H, W) -> (1, H, W) for PyTorch
        channel_first = np.expand_dims(normalized, axis=0)
        
        return channel_first

def make_continuous_env(render_mode=None, frame_stack=4, frame_skip=2):
    """Helper to create a continuous action environment"""
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    env = ContinuousCarRacingWrapper(env, frame_stack=frame_stack, frame_skip=frame_skip)
    return env

def make_vec_continuous_env(num_envs=4, frame_stack=4, frame_skip=2):
    """
    Create vectorized continuous environments for parallel training
    
    Args:
        num_envs: Number of parallel environments
        frame_stack: Number of frames to stack
        frame_skip: Action repeat count (2-4 recommended for speed)
    
    Returns:
        Vectorized continuous environment
    """
    def env_fn():
        return make_continuous_env(render_mode=None, frame_stack=frame_stack, frame_skip=frame_skip)
    
    from gymnasium.vector import SyncVectorEnv
    
    vec_env = SyncVectorEnv([env_fn for _ in range(num_envs)])
    return vec_env


