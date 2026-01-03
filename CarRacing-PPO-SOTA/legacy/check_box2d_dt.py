"""
检查 Box2D 物理引擎的 dt 参数设置
"""
import gymnasium as gym
from utils_env_sota import make_sota_env

def check_box2d_params(physics_fps=50):
    """检查 Box2D 参数"""
    print(f"\n检查 physics_fps={physics_fps} 时的 Box2D 参数:")
    print("-" * 60)
    
    env = make_sota_env(render_mode=None, frame_stack=4, frame_skip=1, physics_fps=physics_fps)
    
    if hasattr(env, 'unwrapped'):
        unwrapped = env.unwrapped
        
        # 检查 fps
        fps = getattr(unwrapped, 'fps', None)
        print(f"env.unwrapped.fps: {fps}")
        
        # 检查 world
        if hasattr(unwrapped, 'world'):
            world = unwrapped.world
            print(f"world 类型: {type(world)}")
            
            # 检查 dt
            if hasattr(world, 'dt'):
                print(f"world.dt: {world.dt}")
                print(f"物理频率 (1/dt): {1.0/world.dt:.1f} Hz")
            else:
                print("world 没有 dt 属性")
                # 列出所有属性
                print(f"world 属性: {[attr for attr in dir(world) if not attr.startswith('_')]}")
        
        # 检查其他可能的物理参数
        for attr in ['dt', 'timeStep', 'time_step', 'physics_dt']:
            if hasattr(unwrapped, attr):
                val = getattr(unwrapped, attr)
                print(f"env.unwrapped.{attr}: {val}")
    
    env.close()

if __name__ == "__main__":
    print("=" * 60)
    print("Box2D 物理引擎参数检查")
    print("=" * 60)
    
    check_box2d_params(physics_fps=50)
    check_box2d_params(physics_fps=180)

