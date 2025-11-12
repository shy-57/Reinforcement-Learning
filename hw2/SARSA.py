import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors

class CliffWalkEnvironment:
    def __init__(self, width=12, height=4):
        self.width = width
        self.height = height
        self.start_pos = (height-1, 0)  # 左下角
        self.goal_pos = (height-1, width-1)  # 右下角
        self.cliff_positions = [(height-1, j) for j in range(1, width-1)]  # 悬崖位置
        self.reset()
    
    def reset(self):
        self.current_pos = self.start_pos
        return self.current_pos
    
    def step(self, action):
        """执行动作并返回(next_state, reward, done)"""
        row, col = self.current_pos
        
        # 动作映射: 0=上, 1=右, 2=下, 3=左
        if action == 0:  # 上
            row = max(0, row-1)
        elif action == 1:  # 右
            col = min(self.width-1, col+1)
        elif action == 2:  # 下
            row = min(self.height-1, row+1)
        elif action == 3:  # 左
            col = max(0, col-1)
        
        new_pos = (row, col)
        self.current_pos = new_pos
        
        # 检查是否到达目标
        if new_pos == self.goal_pos:
            return new_pos, 0, True
        
        # 检查是否掉入悬崖
        if new_pos in self.cliff_positions:
            self.current_pos = self.start_pos
            return self.start_pos, -100, False
        
        # 普通移动
        return new_pos, -1, False
    
    def get_all_states(self):
        """返回所有可能的状态"""
        return [(i, j) for i in range(self.height) for j in range(self.width)]

class SARSAAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        
        # 初始化Q表
        self.q_table = {}
        for state in env.get_all_states():
            self.q_table[state] = np.zeros(4)  # 4个动作
    
    def choose_action(self, state):
        """ε-greedy策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # 随机探索
        else:
            return np.argmax(self.q_table[state])  # 贪婪选择
    
    def update(self, state, action, reward, next_state, next_action, done):
        """SARSA更新Q值"""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state][next_action]
        
        self.q_table[state][action] += self.alpha * (target - self.q_table[state][action])
    
    def get_optimal_path(self):
        """获取最优路径"""
        path = []
        state = self.env.start_pos
        visited = set()
        
        while state != self.env.goal_pos and state not in visited:
            visited.add(state)
            path.append(state)
            action = np.argmax(self.q_table[state])
            
            # 模拟移动
            row, col = state
            if action == 0:  # 上
                row = max(0, row-1)
            elif action == 1:  # 右
                col = min(self.env.width-1, col+1)
            elif action == 2:  # 下
                row = min(self.env.height-1, row+1)
            elif action == 3:  # 左
                col = max(0, col-1)
            
            state = (row, col)
            
            # 防止无限循环
            if len(path) > 100:
                break
        
        if state == self.env.goal_pos:
            path.append(state)
        
        return path

def train_sarsa(env, agent, episodes=500):
    """训练SARSA智能体"""
    rewards_per_episode = []
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.choose_action(state)
        total_reward = 0
        steps = 0
        
        while True:
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            state = next_state
            action = next_action
            total_reward += reward
            steps += 1
            
            if done or steps > 1000:
                break
        
        rewards_per_episode.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
    
    return rewards_per_episode

def visualize_results(env, agent, rewards):
    """可视化训练结果和路径 - 修改为分开显示两个图像"""
    # 1. 第一个图：绘制奖励曲线
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Progress - Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.show()
    
    # 2. 第二个图：绘制网格环境和最优路径
    plt.figure(figsize=(10, 6))
    cmap = colors.ListedColormap(['white', 'red', 'green', 'blue'])
    grid = np.zeros((env.height, env.width))
    
    # 标记不同区域
    for i in range(env.height):
        for j in range(env.width):
            pos = (i, j)
            if pos == env.start_pos:
                grid[i, j] = 2  # 起点-绿色
            elif pos == env.goal_pos:
                grid[i, j] = 3  # 终点-蓝色
            elif pos in env.cliff_positions:
                grid[i, j] = 1  # 悬崖-红色
    
    plt.imshow(grid, cmap=cmap, aspect='auto')
    
    # 绘制最优路径
    path = agent.get_optimal_path()
    if len(path) > 1:
        path_y, path_x = zip(*path)
        plt.plot(path_x, path_y, 'o-', color='orange', linewidth=3, markersize=8, label='Optimal Path')
    
    # 添加网格线和标签
    plt.gca().set_xticks(np.arange(-0.5, env.width, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, env.height, 1), minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.title('Cliff Walking Environment with Optimal Path')
    plt.legend()
    plt.show()
    
    # 打印路径信息
    print(f"Optimal path length: {len(path)}")
    print(f"Optimal path: {path}")

# 主程序
if __name__ == "__main__":
    # 创建环境和智能体
    env = CliffWalkEnvironment()
    agent = SARSAAgent(env, alpha=0.1, gamma=0.9, epsilon=0.1)
    
    # 训练SARSA算法
    print("Training SARSA agent...")
    rewards = train_sarsa(env, agent, episodes=500)
    
    # 可视化结果
    visualize_results(env, agent, rewards)
    
    # 测试最终策略
    print("\nTesting final policy:")
    state = env.reset()
    total_reward = 0
    steps = 0
    path = [state]
    
    while True:
        action = np.argmax(agent.q_table[state])
        state, reward, done = env.step(action)
        path.append(state)
        total_reward += reward
        steps += 1
        
        if done or steps > 100:
            break
    
    print(f"Test - Total reward: {total_reward}, Steps: {steps}")
