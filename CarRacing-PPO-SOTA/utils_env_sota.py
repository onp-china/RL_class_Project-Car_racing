
"""
SOTA Reward Shaping Wrapper for CarRacing
基于赛道几何的奖励重塑 - 终极版本
核心思想：
1. 有效进度奖励：速度投影到赛道方向
2. 中心线奖励：距离赛道中心越近越好
3. 稳定性奖励：惩罚抖动和冲突
"""
import gymnasium as gym
import numpy as np
import cv2
from gymnasium import spaces

class SOTARewardWrapper(gym.Wrapper):
    """
    SOTA Reward Shaping with Track Geometry
    """
    def __init__(self, env, frame_stack=4, frame_skip=4):
        super().__init__(env)
        
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        # Reward coefficients (Tuned for balance)
        self.progress_coef = 0.1
        self.centering_coef = 0.5
        self.steering_penalty_coef = 2.0  # L2 penalty: allows micro-adjustments, punishes jitter
        self.pedal_conflict_penalty = 0.5
        self.offtrack_penalty = 10.0
        
        # Track state
        self.prev_steering = 0.0
        self.track_width = 10.0  # CarRacing track width approximation
        
        # Cache for performance
        self.cached_closest_idx = 0
        self.step_count = 0
        
        # Action Space
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation Space: (H, W, C)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(96, 96, self.frame_stack), dtype=np.float32
        )
        
        self.frames = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        processed_frame = self._process_obs(obs)
        self.frames = np.repeat(processed_frame, self.frame_stack, axis=-1)
        
        self.prev_steering = 0.0
        self.cached_closest_idx = 0
        self.step_count = 0
        
        return self.frames.copy(), info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        last_obs = None
        
        # Frame skip
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
                # Car not initialized yet
                processed_frame = self._process_obs(last_obs)
                self.frames = np.roll(self.frames, shift=-1, axis=-1)
                self.frames[:, :, -1] = processed_frame[:, :, 0]
                return self.frames.copy(), shaped_reward, terminated, truncated, info
            
            car_pos = np.array([car.hull.position.x, car.hull.position.y])
            car_vel = np.array([car.hull.linearVelocity.x, car.hull.linearVelocity.y])
            
            # Get track points
            track = self.env.unwrapped.track
            if len(track) == 0:
                # Track not generated yet
                processed_frame = self._process_obs(last_obs)
                self.frames = np.roll(self.frames, shift=-1, axis=-1)
                self.frames[:, :, -1] = processed_frame[:, :, 0]
                return self.frames.copy(), shaped_reward, terminated, truncated, info
            
            track_points = np.array([[t[2], t[3]] for t in track])
            
            # Performance optimization: search nearby only
            self.step_count += 1
            if self.step_count % 3 == 0:  # Update every 3 steps
                # Search range
                search_start = max(0, self.cached_closest_idx - 30)
                search_end = min(len(track_points), self.cached_closest_idx + 30)
                search_indices = list(range(search_start, search_end))
                
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
            
            # C. Balanced Anti-Wobble System (精细分段策略)
            
            speed = np.sqrt(car_vel[0]**2 + car_vel[1]**2)
            steer = abs(action[0])
            
            # Layer 1: High-Speed Micro-Wobble Suppression
            # 只惩罚"高速 + 小幅度晃动"（真正的抖动）
            # 放过"高速 + 大转向"（真正的过弯）
            wobbly_penalty = 0
            if speed > 40.0:  # 高速状态
                if 0.02 < steer < 0.2:  # 小幅度转向（疑似抖动）
                    # 设定上限，防止超过速度奖励
                    penalty = steer * speed * 0.02
                    wobbly_penalty = min(penalty, 0.5)  # 最多扣 0.5
            
            # Layer 2: L2 Smoothness Penalty (只罚剧烈突变)
            # 微小修正（<0.3）：允许
            # 剧烈抽搐（>0.3）：平方重罚
            steering_diff = action[0] - self.prev_steering
            if abs(steering_diff) > 0.3:  # 阈值提高到 0.3
                smoothness_penalty = (steering_diff ** 2) * 0.5  # 系数降低
            else:
                smoothness_penalty = 0
            
            # Total Steering Penalty
            steering_penalty = wobbly_penalty + smoothness_penalty
            
            self.prev_steering = action[0]
            
            # D. Pedal Conflict Penalty
            pedal_penalty = 0
            if action[1] > 0.3 and action[2] > 0.3:
                pedal_penalty = self.pedal_conflict_penalty * action[1] * action[2]
            
            # E. Off-track termination
            if dist_to_center > self.track_width * 1.5:
                shaped_reward -= self.offtrack_penalty
                terminated = True
            else:
                # Apply shaping
                shaped_reward += reward_progress + reward_centering - steering_penalty - pedal_penalty
                
        except Exception as e:
            # Fallback: if any error, use original reward
            pass
        
        # Update frames
        processed_frame = self._process_obs(last_obs)
        self.frames = np.roll(self.frames, shift=-1, axis=-1)
        self.frames[:, :, -1] = processed_frame[:, :, 0]
        
        return self.frames.copy(), shaped_reward, terminated, truncated, info

    def _process_obs(self, obs):
        """Fast preprocessing"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        normalized = gray.astype(np.float32) / 255.0
        channel_last = np.expand_dims(normalized, axis=-1)
        return channel_last

def make_sota_env(render_mode=None, frame_stack=4, frame_skip=4, physics_fps=50):
    env = gym.make("CarRacing-v3", render_mode=render_mode)
    
    # 修改物理引擎 FPS (默认是 50)
    if physics_fps != 50:
        if hasattr(env, 'unwrapped'):
            env.unwrapped.fps = physics_fps
            # 关键：同时修改 Box2D 的 dt 参数来控制物理频率
            # dt = 1.0 / fps，所以 fps 越高，dt 越小
            if hasattr(env.unwrapped, 'world'):
                world = env.unwrapped.world
                if hasattr(world, 'dt'):
                    world.dt = 1.0 / physics_fps
                    print(f"[INFO] Box2D dt set to {world.dt:.6f} (physics_fps={physics_fps} Hz)")
            
    env = SOTARewardWrapper(env, frame_stack, frame_skip)
    return env

def make_vec_sota_env(num_envs=8, frame_stack=4, frame_skip=4, physics_fps=50):
    """
    Create vectorized SOTA environments
    """
    def env_fn():
        return make_sota_env(None, frame_stack, frame_skip, physics_fps)
    
    from gymnasium.vector import AsyncVectorEnv
    
    try:
        envs = AsyncVectorEnv([env_fn for _ in range(num_envs)])
        return envs
    except Exception as e:
        print(f"Warning: AsyncVectorEnv failed ({e}), falling back to SyncVectorEnv")
        from gymnasium.vector import SyncVectorEnv
        return SyncVectorEnv([env_fn for _ in range(num_envs)])

