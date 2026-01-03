"""
Multi-Agent CarRacing Environment
支持两辆车在同一赛道上比赛，具有真实的物理碰撞
"""
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import (
    CarRacing, STATE_W, STATE_H, WINDOW_W, WINDOW_H, SCALE, ZOOM, FPS, PLAYFIELD,
    TRACK_DETAIL_STEP, TRACK_WIDTH, GRASS_DIM, MAX_SHAPE_DIM
)
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.utils import EzPickle

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


class MultiCarRacing(CarRacing, EzPickle):
    """
    多智能体 CarRacing 环境
    支持2辆车在同一赛道上比赛
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }
    
    def __init__(
        self,
        num_agents: int = 2,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
    ):
        # 初始化父类，但不创建车（我们会在reset中创建多辆车）
        EzPickle.__init__(
            self,
            num_agents,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
        )
        
        self.num_agents = num_agents
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self._init_colors()
        
        # 初始化 Box2D 世界
        from gymnasium.envs.box2d.car_racing import FrictionDetector
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        try:
            import Box2D
            from Box2D.b2 import contactListener, fixtureDef, polygonShape
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
            ) from e
        
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface | None = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        
        # 多辆车列表
        self.cars: list[Car] = []
        
        # 每辆车的奖励和状态
        self.rewards = [0.0] * num_agents
        self.prev_rewards = [0.0] * num_agents
        self.tile_visited_counts = [0] * num_agents
        self.new_laps = [False] * num_agents
        
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        
        # 动作空间：每个智能体一个动作
        if self.continuous:
            # 每个智能体的动作空间是 Box([-1,0,0], [1,1,1])
            self.action_space = spaces.Tuple([
                spaces.Box(
                    np.array([-1, 0, 0]).astype(np.float32),
                    np.array([+1, +1, +1]).astype(np.float32),
                ) for _ in range(num_agents)
            ])
        else:
            # 离散动作空间
            self.action_space = spaces.Tuple([
                spaces.Discrete(5) for _ in range(num_agents)
            ])
        
        # 观察空间：每个智能体一个观察
        self.observation_space = spaces.Tuple([
            spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
            for _ in range(num_agents)
        ])
        
        self.render_mode = render_mode
    
    def _destroy(self):
        """销毁所有车辆和赛道"""
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        for car in self.cars:
            car.destroy()
        self.cars = []
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """重置环境，创建多辆车"""
        super().reset(seed=seed)
        self._destroy()
        
        from gymnasium.envs.box2d.car_racing import FrictionDetector
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        
        # 重置奖励和状态
        self.rewards = [0.0] * self.num_agents
        self.prev_rewards = [0.0] * self.num_agents
        self.tile_visited_counts = [0] * self.num_agents
        self.new_laps = [False] * self.num_agents
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        
        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]
            self._reinit_colors(randomize)
        
        # 创建赛道
        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print("retry to generate track")
        
        # 创建多辆车，并排排列在起点
        start_alpha, start_beta, start_x, start_y = self.track[0]
        
        # 计算垂直于赛道的方向（用于并排排列）
        perp_x = -math.sin(start_beta)
        perp_y = math.cos(start_beta)
        
        # 车辆间距
        car_spacing = 50 / SCALE  # 调整这个值来控制车辆间距
        
        for i in range(self.num_agents):
            # 计算每辆车的初始位置（并排）
            offset = (i - (self.num_agents - 1) / 2) * car_spacing
            car_x = start_x + offset * perp_x
            car_y = start_y + offset * perp_y
            
            # 创建车辆，使用不同的颜色
            car = Car(self.world, start_beta, car_x, car_y)
            
            # 为每辆车设置不同颜色以便区分
            colors = [
                (0.8, 0.0, 0.0),  # 红色
                (0.0, 0.0, 0.8),  # 蓝色
                (0.0, 0.8, 0.0),  # 绿色
                (0.8, 0.8, 0.0),  # 黄色
            ]
            car.hull.color = colors[i % len(colors)]
            
            self.cars.append(car)
        
        # 为每辆车创建独立的接触检测器
        # 注意：当前使用共享的 FrictionDetector，每辆车的奖励计算是简化的
        # 实际应该为每辆车单独跟踪 tile_visited_count
        
        if self.render_mode == "human":
            self.render()
        
        # 返回初始观察
        observations = self._get_observations()
        return observations, {}
    
    def _get_observations(self):
        """获取所有车辆的观察（每辆车自己的视角）"""
        observations = []
        for car in self.cars:
            # 为每辆车渲染其视角
            obs = self._render_car_view(car)
            observations.append(obs)
        return tuple(observations)
    
    def _render_car_view(self, car: Car):
        """渲染特定车辆的视角（96x96 RGB图像）"""
        # 创建临时surface用于渲染单车的视角
        temp_surf = pygame.Surface((STATE_W, STATE_H))
        
        # 计算该车的视角变换
        angle = -car.hull.angle
        zoom = ZOOM * SCALE
        scroll_x = -(car.hull.position[0]) * zoom
        scroll_y = -(car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (STATE_W / 2 + trans[0], STATE_H / 2 + trans[1])
        
        # 渲染赛道
        self._render_road_on_surface(temp_surf, zoom, trans, angle)
        
        # 渲染所有车辆
        for c in self.cars:
            c.draw(
                temp_surf,
                zoom,
                trans,
                angle,
                True,  # 绘制所有车辆
            )
        
        # 翻转并转换为numpy数组
        temp_surf = pygame.transform.flip(temp_surf, False, True)
        obs_array = np.transpose(
            np.array(pygame.surfarray.pixels3d(temp_surf)), axes=(1, 0, 2)
        )
        return obs_array
    
    def _render_road_on_surface(self, surface, zoom, translation, angle):
        """在指定surface上渲染赛道"""
        # 临时保存原始surface，使用传入的surface
        original_surf = self.surf
        self.surf = surface
        
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]
        
        # 绘制背景
        self._draw_colored_polygon(
            surface, field, self.bg_color, zoom, translation, angle, clip=False
        )
        
        # 绘制草地
        GRASS_DIM = PLAYFIELD / 20.0
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(
                surface, poly, self.grass_color, zoom, translation, angle
            )
        
        # 绘制道路
        for poly, color in self.road_poly:
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(surface, poly, color, zoom, translation, angle)
        
        # 恢复原始surface
        self.surf = original_surf
    
    def step(self, actions):
        """执行一步，actions是包含每个智能体动作的列表或元组"""
        # 确保actions是列表
        if isinstance(actions, (tuple, np.ndarray)):
            actions = list(actions)
        
        # 应用每个车辆的动作
        for i, car in enumerate(self.cars):
            if i < len(actions) and actions[i] is not None:
                action = actions[i]
                if self.continuous:
                    action = np.array(action, dtype=np.float64)
                    car.steer(-action[0])
                    car.gas(action[1])
                    car.brake(action[2])
                else:
                    # 离散动作
                    if action == 0:  # do nothing
                        pass
                    elif action == 1:  # steer right
                        car.steer(-0.6)
                    elif action == 2:  # steer left
                        car.steer(0.6)
                    elif action == 3:  # gas
                        car.gas(0.2)
                    elif action == 4:  # brake
                        car.brake(0.8)
        
        # 更新所有车辆
        for car in self.cars:
            car.step(1.0 / FPS)
        
        # 更新物理世界
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        
        # 计算奖励和终止条件
        observations = self._get_observations()
        rewards = []
        terminations = []
        truncations = []
        infos = []
        
        for i in range(self.num_agents):
            car = self.cars[i]
            
            # 更新奖励
            # 注意：当前实现是简化版本，每辆车共享相同的奖励逻辑
            # 完整的实现需要为每辆车单独跟踪 tile_visited_count
            # 这里使用简化的奖励：每帧-0.1，完成一圈+1000
            self.rewards[i] -= 0.1  # 每帧惩罚
            
            # 检查是否出界
            x, y = car.hull.position
            terminated = False
            truncated = False
            info = {}
            
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                self.rewards[i] -= 100
            
            # 检查是否完成一圈
            # 注意：当前使用简化的完成检测，实际应该为每辆车单独跟踪
            # 这里暂时使用时间或距离作为完成条件
            if self.t >= 30.0:  # 简化：30秒后认为完成
                terminated = True
                info["lap_finished"] = True
                self.rewards[i] += 1000  # 完成奖励
            
            step_reward = self.rewards[i] - self.prev_rewards[i]
            self.prev_rewards[i] = self.rewards[i]
            
            rewards.append(step_reward)
            terminations.append(terminated)
            truncations.append(truncated)
            infos.append(info)
        
        if self.render_mode == "human":
            self.render()
        
        return observations, rewards, terminations, truncations, infos
    
    def render(self):
        """渲染全局视角（显示所有车辆）"""
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)
    
    def _render(self, mode: str):
        """渲染全局视角"""
        assert mode in self.metadata["render_modes"]
        
        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if "t" not in self.__dict__:
            return  # reset() not called yet
        
        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))
        
        if len(self.cars) == 0:
            return
        
        # 计算相机位置：跟随所有车辆的中心点
        center_x = sum(car.hull.position[0] for car in self.cars) / len(self.cars)
        center_y = sum(car.hull.position[1] for car in self.cars) / len(self.cars)
        
        # 计算缩放：根据车辆之间的距离调整
        if len(self.cars) > 1:
            max_dist = 0
            for i in range(len(self.cars)):
                for j in range(i + 1, len(self.cars)):
                    dx = self.cars[i].hull.position[0] - self.cars[j].hull.position[0]
                    dy = self.cars[i].hull.position[1] - self.cars[j].hull.position[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    max_dist = max(max_dist, dist)
            # 根据最大距离调整缩放
            zoom_factor = max(1.0, max_dist / (200 / SCALE))
            zoom = ZOOM * SCALE * zoom_factor
        else:
            zoom = ZOOM * SCALE
        
        # 使用第一辆车的角度作为相机角度
        angle = -self.cars[0].hull.angle
        
        scroll_x = -center_x * zoom
        scroll_y = -center_y * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])
        
        # 渲染赛道
        self._render_road(zoom, trans, angle)
        
        # 渲染所有车辆
        for car in self.cars:
            car.draw(
                self.surf,
                zoom,
                trans,
                angle,
                mode not in ["state_pixels_list", "state_pixels"],
            )
        
        self.surf = pygame.transform.flip(self.surf, False, True)
        
        # 显示统计信息（第一辆车的）
        if len(self.cars) > 0:
            self._render_indicators(WINDOW_W, WINDOW_H, self.cars[0])
        
        # 显示奖励信息
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        for i, car in enumerate(self.cars):
            text = font.render(f"Car{i}: {self.rewards[i]:.0f}", True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (60 + i * 200, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surf.blit(text, text_rect)
        
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (600, 400))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen
    
    def _render_indicators(self, W, H, car: Car):
        """渲染车辆指标（速度、传感器等）"""
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)
        
        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]
        
        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]
        
        true_speed = np.sqrt(
            np.square(car.hull.linearVelocity[0])
            + np.square(car.hull.linearVelocity[1])
        )
        
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)
        
        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            car.wheels[0].omega,
            vertical_ind(7, 0.01 * car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[1].omega,
            vertical_ind(8, 0.01 * car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[2].omega,
            vertical_ind(9, 0.01 * car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            car.wheels[3].omega,
            vertical_ind(10, 0.01 * car.wheels[3].omega),
            (51, 0, 255),
        )
        
        render_if_min(
            car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            car.hull.angularVelocity,
            horiz_ind(30, -0.8 * car.hull.angularVelocity),
            (255, 0, 0),
        )
    
    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

