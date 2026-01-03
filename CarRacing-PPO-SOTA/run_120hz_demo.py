"""
120Hz 超高频控制演示脚本包装器
"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 导入并运行实际脚本
from scripts.visualization.run_120hz_demo import main

if __name__ == "__main__":
    main()

