import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

class MazeEnv:
    """迷宫环境类：封装MDP的状态、动作、转移、奖励等核心逻辑"""
    def __init__(self, maze, start, goal):
        self.maze = maze          # 迷宫矩阵：0=可走，1=墙壁
        self.rows = len(maze)     # 迷宫行数
        self.cols = len(maze[0])  # 迷宫列数
        self.start = start        # 起点坐标 (row, col)
        self.goal = goal          # 终点坐标 (row, col)
        self.actions = ['N', 'E', 'S', 'W']  # 动作：北、东、南、西
        # 动作对应的坐标偏移量（行变化, 列变化）
        self.action_deltas = {'N': (-1, 0), 'E': (0, 1), 'S': (1, 0), 'W': (0, -1)}
        
        # 提取所有非墙壁状态（用于价值迭代）
        self.states = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == 0:
                    self.states.append((i, j))
        # 终止状态集合（仅终点）
        self.terminals = {goal}
    
    def get_reward(self, state, action):
        """计算执行动作后的即时奖励"""
        next_state = self.get_next_state(state, action)
        return 0 if next_state == self.goal else -1  # 终点奖励0，否则-1
    
    def get_next_state(self, state, action):
        """执行动作后转移到的下一个状态（处理墙壁和边界）"""
        i, j = state
        di, dj = self.action_deltas[action]
        ni, nj = i + di, j + dj
        # 检查是否越界或碰到墙壁
        if 0 <= ni < self.rows and 0 <= nj < self.cols and self.maze[ni][nj] == 0:
            return (ni, nj)
        else:
            return state  # 碰到墙壁/边界，停留在当前状态
    
    def is_terminal(self, state):
        """判断是否为终止状态（到达终点）"""
        return state == self.goal


def value_iteration(env, gamma=0.9, epsilon=1e-6):
    """价值迭代算法：求解最优价值函数与策略"""
    # 初始化价值函数（所有状态价值为0）
    V = {s: 0 for s in env.states}
    
    while True:
        delta = 0  # 记录价值函数的最大变化量
        V_new = V.copy()  # 新价值函数
        
        for s in env.states:
            if env.is_terminal(s):
                V_new[s] = 0  # 终止状态价值固定为0
                continue
            
            # 对每个动作，计算「即时奖励 + 折扣后未来价值」
            action_values = []
            for a in env.actions:
                reward = env.get_reward(s, a)
                next_s = env.get_next_state(s, a)
                action_value = reward + gamma * V[next_s]
                action_values.append(action_value)
            
            # 取最大价值作为当前状态的新价值
            V_new[s] = max(action_values)
            # 更新价值变化量
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new  # 迭代价值函数
        if delta < epsilon:  # 收敛时停止
            break
    
    # 从最优价值函数中提取策略（每个状态选价值最大的动作）
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = None  # 终止状态无动作
            continue
        
        action_values = []
        for a in env.actions:
            reward = env.get_reward(s, a)
            next_s = env.get_next_state(s, a)
            action_value = reward + gamma * V[next_s]
            action_values.append(action_value)
        
        best_action_idx = np.argmax(action_values)  # 最大价值对应的动作索引
        best_action = env.actions[best_action_idx]
        policy[s] = best_action
    
    return V, policy


def generate_path(env, policy, start):
    """根据最优策略生成从起点到终点的路径"""
    path = [start]
    current = start
    
    while not env.is_terminal(current):
        action = policy[current]
        current = env.get_next_state(current, action)
        path.append(current)
        # 安全限制：防止无限循环（理论上价值迭代不会陷入循环）
        if len(path) > 100:
            break
    
    return path


def visualize_maze(env, V, policy, path):
    """可视化迷宫、价值函数、策略和路径"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建颜色映射：墙壁、路径、起点、终点
    cmap = ListedColormap(['white', 'black', 'green', 'red', 'lightblue'])
    
    # 创建可视化矩阵
    # 0=可走区域, 1=墙壁, 2=起点, 3=终点, 4=路径
    viz_matrix = np.array(env.maze).astype(float)
    
    # 标记起点和终点
    viz_matrix[env.start] = 2
    viz_matrix[env.goal] = 3
    
    # 标记路径（排除起点和终点）
    for cell in path[1:-1]:
        viz_matrix[cell] = 4
    
    # 绘制迷宫基础
    ax.imshow(viz_matrix, cmap=cmap, vmin=0, vmax=4)
    
    # 添加价值函数标签（统一放在左下角）
    for (i, j) in env.states:
        # 跳过终点（终点有特殊标记）
        if (i, j) == env.goal:
            continue
            
        # 在格子左下角显示价值函数
        ax.text(j - 0.4, i + 0.4, f"{V[(i, j)]:.2f}", 
                ha='left', va='top', 
                fontsize=8, color='darkblue')
    
    # 添加策略方向符号（不显示箭头，改用符号）
    action_symbols = {'N': '↑', 'E': '→', 'S': '↓', 'W': '←'}
    
    for (i, j) in env.states:
        if env.is_terminal((i, j)):
            continue
            
        action = policy[(i, j)]
        # 如果这个状态在路径上，跳过（避免与路径箭头重叠）
        if (i, j) in path:
            continue
            
        symbol = action_symbols[action]
        ax.text(j, i, symbol, 
                ha='center', va='center', 
                fontsize=12, color='darkred', weight='bold')
    
    # 添加路径箭头（使用更明显的蓝色箭头）
    for k in range(1, len(path)):
        prev = path[k-1]
        curr = path[k]
        dx = curr[1] - prev[1]
        dy = curr[0] - prev[0]
        
        # 计算箭头起点（稍微偏移，避免重叠）
        start_x = prev[1] + dx * 0.1
        start_y = prev[0] + dy * 0.1
        
        # 计算箭头长度（稍微缩短）
        arrow_length_x = dx * 0.8
        arrow_length_y = dy * 0.8
        
        ax.arrow(start_x, start_y, arrow_length_x, arrow_length_y, 
                 facecolor='blue', edgecolor='blue', 
                 width=0.3, head_width=0.5, head_length=0.3,
                 length_includes_head=True)
    
    # 在路径点上添加步骤编号（放在格子右上角）
    for k, (i, j) in enumerate(path):
        if k == 0:  # 起点
            ax.text(j, i, "起点", 
                    ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        elif k == len(path) - 1:  # 终点
            ax.text(j, i, "终点", 
                    ha='center', va='center', 
                    fontsize=10, color='white', weight='bold')
        else:  # 路径中间点
            # 在路径点上添加步骤编号（放在格子右上角）
            ax.text(j + 0.4, i - 0.4, str(k), 
                    ha='right', va='top', 
                    fontsize=10, color='blue', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, fc='white', edgecolor='black'),
        plt.Rectangle((0,0), 1, 1, fc='black'),
        plt.Rectangle((0,0), 1, 1, fc='green'),
        plt.Rectangle((0,0), 1, 1, fc='red'),
        plt.Rectangle((0,0), 1, 1, fc='lightblue'),
        plt.Line2D([0], [0], color='blue', lw=2),
        plt.Text(0, 0, '↑', color='darkred', fontsize=12, ha='center')
    ]
    
    ax.legend(legend_elements, 
              ['可走区域', '墙壁', '起点', '终点', '路径', '路径方向', '策略方向'],
              loc='upper right', bbox_to_anchor=(1.25, 1))
    
    ax.set_title('迷宫求解可视化')
    ax.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)
    plt.tight_layout()
    plt.show()
    
    # 打印路径坐标
    print("\n最优路径坐标序列：")
    for i, pos in enumerate(path):
        print(f"步骤 {i}: ({pos[0]}, {pos[1]})")


def analyze_results(env, V, policy, path):
    """分析并打印结果"""
    print("\n===== 结果分析 =====")
    print(f"起点: {env.start}, 终点: {env.goal}")
    print(f"路径长度: {len(path)-1} 步")
    
    # 计算路径总奖励
    total_reward = 0
    for i in range(1, len(path)):
        prev = path[i-1]
        # 找出从prev到path[i]的动作
        for action, delta in env.action_deltas.items():
            next_state = (prev[0] + delta[0], prev[1] + delta[1])
            if next_state == path[i]:
                total_reward += env.get_reward(prev, action)
                break
    
    print(f"路径总奖励: {total_reward}")
    
    # 价值函数统计
    values = [V[s] for s in env.states if not env.is_terminal(s)]
    print(f"\n价值函数统计:")
    print(f"  最大值: {max(values):.2f}, 最小值: {min(values):.2f}")
    print(f"  平均值: {np.mean(values):.2f}, 标准差: {np.std(values):.2f}")
    
    # 策略分布
    action_counts = {a: 0 for a in env.actions}
    for s in policy:
        if policy[s] is not None:
            action_counts[policy[s]] += 1
    
    print("\n策略分布:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} 个状态")


if __name__ == "__main__":
    # 1. 定义迷宫（0=可走，1=墙壁）
    maze = [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
    start = (2,0)  # 起点坐标
    goal = (6,7)   # 终点坐标
    
    # 2. 初始化环境
    env = MazeEnv(maze, start, goal)
    gamma = 0.9  # 折扣因子
    
    # 3. 运行价值迭代
    print("正在运行价值迭代算法...")
    V, policy = value_iteration(env, gamma)
    print("价值函数已收敛！")
    
    # 4. 生成最优路径
    path = generate_path(env, policy, start)
    
    # 5. 可视化与分析
    visualize_maze(env, V, policy, path)
    analyze_results(env, V, policy, path)