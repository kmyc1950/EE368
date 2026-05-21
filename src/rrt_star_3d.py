import numpy as np
import matplotlib.pyplot as plt

# === 加上下面这一行，专门显式注册 3D 绘图引擎 ===
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None
        self.cost = 0.0 # 从起点到当前节点的路径代价（长度）

class RRTStar3D:
    def __init__(self, start, goal, obstacle_list, bounds, step_size=0.05, max_iter=1000, search_radius=0.15):
        """
        初始化 RRT* 规划器
        :param start: 起点 (x, y, z)
        :param goal: 终点 (x, y, z)
        :param obstacle_list: 障碍物列表，格式为 [(x, y, z, radius), ...]
        :param bounds: 空间边界 [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
        :param step_size: 树每次生长的最大步长
        :param max_iter: 最大迭代次数
        :param search_radius: RRT* 独有的重连搜索半径
        """
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.obstacle_list = obstacle_list
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius
        self.node_list = [self.start]

    def plan(self):
        for i in range(self.max_iter):
            # 1. 随机采样一个点 (有 10% 的概率直接采样终点，加速收敛)
            if np.random.rand() > 0.1:
                rnd_point = [np.random.uniform(self.bounds[0][0], self.bounds[0][1]),
                             np.random.uniform(self.bounds[1][0], self.bounds[1][1]),
                             np.random.uniform(self.bounds[2][0], self.bounds[2][1])]
            else:
                rnd_point = [self.goal.x, self.goal.y, self.goal.z]

            # 2. 找到树中距离随机点最近的节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_point)
            nearest_node = self.node_list[nearest_ind]

            # 3. 朝着随机点方向生长一步，生成新节点
            new_node = self.steer(nearest_node, rnd_point, self.step_size)

            # 4. 碰撞检测
            if self.check_collision(new_node, self.obstacle_list):
                # 5. RRT* 核心：寻找搜索半径内的所有邻居节点
                near_inds = self.find_near_nodes(new_node)
                
                # 6. 为新节点选择代价最小的父节点
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    # 7. RRT* 核心：重连（Rewire）树，优化现有路径
                    self.rewire(new_node, near_inds)

        # 尝试生成最终路径
        return self.generate_final_course()

    def steer(self, from_node, to_point, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, from_node.z)
        d, theta, phi = self.calc_distance_and_angle(new_node, to_point)
        
        if extend_length > d:
            extend_length = d
            
        new_node.x += extend_length * np.sin(phi) * np.cos(theta)
        new_node.y += extend_length * np.sin(phi) * np.sin(theta)
        new_node.z += extend_length * np.cos(phi)
        
        new_node.cost = from_node.cost + extend_length
        new_node.parent = from_node
        return new_node

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
            
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            d, _, _ = self.calc_distance_and_angle(near_node, [new_node.x, new_node.y, new_node.z])
            if self.check_collision_line(near_node, new_node, self.obstacle_list):
                costs.append(near_node.cost + d)
            else:
                costs.append(float("inf"))
                
        min_cost = min(costs)
        if min_cost == float("inf"):
            return None
            
        min_ind = near_inds[costs.index(min_cost)]
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            d, _, _ = self.calc_distance_and_angle(new_node, [near_node.x, near_node.y, near_node.z])
            edge_node = self.steer(new_node, [near_node.x, near_node.y, near_node.z])
            
            if not edge_node:
                continue
            edge_node.cost = new_node.cost + d
            
            # 如果经过新节点到达邻居的代价更小，则重新连接
            if near_node.cost > edge_node.cost:
                if self.check_collision_line(new_node, near_node, self.obstacle_list):
                    near_node.parent = new_node
                    near_node.cost = edge_node.cost
                    self.propagate_cost_to_leaves(near_node)

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                d, _, _ = self.calc_distance_and_angle(parent_node, [node.x, node.y, node.z])
                node.cost = parent_node.cost + d
                self.propagate_cost_to_leaves(node)

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.search_radius * np.sqrt((np.log(nnode) / nnode))
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 + (node.z - new_node.z)**2 for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r**2]
        return near_inds

    def check_collision(self, node, obstacle_list):
        for (ox, oy, oz, size) in obstacle_list:
            dx = ox - node.x
            dy = oy - node.y
            dz = oz - node.z
            d = np.sqrt(dx**2 + dy**2 + dz**2)
            if d <= size + 0.02: # 0.02 是机械臂的安全膨胀距离
                return False # 发生碰撞
        return True # 安全

    def check_collision_line(self, node1, node2, obstacle_list):
        # 简单实现：将两点之间的连线离散化进行碰撞检测
        steps = 10
        for i in range(steps + 1):
            x = node1.x + (node2.x - node1.x) * i / steps
            y = node1.y + (node2.y - node1.y) * i / steps
            z = node1.z + (node2.z - node1.z) * i / steps
            if not self.check_collision(Node(x, y, z), obstacle_list):
                return False
        return True

    def get_nearest_node_index(self, node_list, rnd_point):
        dlist = [(node.x - rnd_point[0])**2 + (node.y - rnd_point[1])**2 + (node.z - rnd_point[2])**2 for node in node_list]
        return dlist.index(min(dlist))

    def calc_distance_and_angle(self, from_node, to_point):
        dx = to_point[0] - from_node.x
        dy = to_point[1] - from_node.y
        dz = to_point[2] - from_node.z
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        theta = np.arctan2(dy, dx)
        phi = np.arccos(dz / d) if d != 0 else 0
        return d, theta, phi

    def generate_final_course(self):
        dlist = [(node.x - self.goal.x)**2 + (node.y - self.goal.y)**2 + (node.z - self.goal.z)**2 for node in self.node_list]
        best_goal_ind = dlist.index(min(dlist))
        
        # 提取路径
        path = []
        node = self.node_list[best_goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([self.start.x, self.start.y, self.start.z])
        
        # 反转路径，使其从起点到终点
        path = path[::-1]
        
        # 确保终点精确被加入
        if min(dlist) <= self.step_size**2:
            path.append([self.goal.x, self.goal.y, self.goal.z])
            
        return np.array(path)

# ================= 测试与可视化 =================
if __name__ == '__main__':
    # 1. 定义参数 (参考你上一版的坐标系)
    start_point = [0.3, -0.2, 0.1]
    goal_point = [0.5, 0.2, 0.3]
    
    # 障碍物列表: (x, y, z, 半径) -> 我们在中间放一个球形障碍物
    obstacles = [
        (0.4, 0.0, 0.15, 0.1) 
    ]
    
    # 空间边界 [X范围, Y范围, Z范围]
    bounds = [[0.2, 0.6], [-0.4, 0.4], [0.0, 0.5]]

    print("开始 RRT* 3D 路径规划...")
    rrt_star = RRTStar3D(start_point, goal_point, obstacles, bounds, max_iter=2000)
    path = rrt_star.plan()

    print(f"规划成功！生成了包含 {len(path)} 个节点的路径。")
    print("路径矩阵 (Waypoints) :\n", path)

    # 2. 三维绘图可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制起点和终点
    ax.scatter(start_point[0], start_point[1], start_point[2], c='g', marker='o', s=100, label='Start')
    ax.scatter(goal_point[0], goal_point[1], goal_point[2], c='r', marker='x', s=100, label='Goal')
    
    # 绘制障碍物
    for (ox, oy, oz, size) in obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = ox + size * np.cos(u) * np.sin(v)
        y = oy + size * np.sin(u) * np.sin(v)
        z = oz + size * np.cos(v)
        ax.plot_surface(x, y, z, color='gray', alpha=0.5)

    # 绘制生成的路径
    ax.plot(path[:, 0], path[:, 1], path[:, 2], '-b', linewidth=2, label='RRT* Path')

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.legend()
    plt.show()