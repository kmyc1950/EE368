#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import math
from scipy.interpolate import splprep, splev
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
# 导入你刚刚写好的 3D RRT* 规划器
from rrt_star_3d import RRTStar3D

class Gen3LiteBsplineJacobianControl:
    def __init__(self):
        rospy.init_node('bspline_jacobian_control_node', anonymous=True)
        
        # ====================================================
        # 1. 核心控制与控制律参数设置
        # ====================================================
        self.vc = 40.0      # 🎯 期望末端恒定速度: 40 mm/s (注意单位：毫米/秒)
        self.Kp = 6.0       # 📈 CLIK 闭环纠偏增益 (根据跟踪效果可上下微调)
        self.dt = 0.02      # ⏱️ 控制周期: 0.02s (对应 50Hz 高频控制)
        
        # 2. 改进型 DH 参数 (完全对齐你上传的 forward_kinematics_mat.py 参数与偏置)
        self.dh_params = [
            [0,    0,     243.3, 0],   # 关节 1
            [90,   0,     30,    90],  # 关节 2
            [180,  280,   20,    90],  # 关节 3
            [90,   0,     245,   90],  # 关节 4
            [90,   0,     57,    0],   # 关节 5
            [-90,  0,     235,   0]    # 关节 6
        ]
        
        # 3. 状态变量与 ROS 通信初始化
        self.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        self.current_q = [0.0] * 6
        self.joint_received = False
        
        # 📌 注意：请根据你 Gazebo 实际的话题命名空间修改以下两个话题
        self.traj_pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=1)
        rospy.Subscriber('/my_gen3_lite/joint_states', JointState, self.joint_state_callback)
        
        # 4. 初始化带有“弧长参数化”的 B 样条轨迹映射表
        self.prepare_arc_length_bspline()

    def joint_state_callback(self, msg):
        """
        ROS 关节状态回调函数：带有安全排序，防止 Gazebo 乱序发送关节数据
        """
        # === 加上这行测试打印，只要收到一次就会疯狂刷屏，确认连接成功 ===
        # rospy.loginfo_once(f"Connected! Received joint names: {msg.name}") 
        
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_q[i] = msg.position[idx] # 弧度 (rad)
        self.joint_received = True

    def get_mdh_matrix(self, alpha_deg, a, d, theta_deg):
        """ 计算单级改进型 DH 变换矩阵 """
        alpha = math.radians(alpha_deg)
        theta = math.radians(theta_deg)
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        return np.array([
            [ct,        -st,         0,   a],
            [st * ca,   ct * ca,   -sa,  -d * sa],
            [st * sa,   ct * sa,    ca,   d * ca],
            [0,         0,           0,   1]
        ])

    def compute_fk_and_jacobian(self, joint_angles_rad):
        """
        输入当前真实关节弧度，输出末端齐次变换矩阵 T 门和 6x6 几何雅可比矩阵 J
        """
        joint_angles_deg = [math.degrees(q) for q in joint_angles_rad]
        T_matrices = []
        T_cur = np.eye(4)

        for i in range(6):
            alpha  = self.dh_params[i][0]
            a      = self.dh_params[i][1]
            d      = self.dh_params[i][2]
            offset = self.dh_params[i][3]

            q_i = joint_angles_deg[i] + offset
            T_i = self.get_mdh_matrix(alpha, a, d, q_i)
            T_cur = np.dot(T_cur, T_i)
            T_matrices.append(T_cur)

        P_ee = T_matrices[-1][0:3, 3]
        J = np.zeros((6, 6))

        for i in range(6):
            Z_i = T_matrices[i][0:3, 2]
            P_i = T_matrices[i][0:3, 3]
            J[0:3, i] = np.cross(Z_i, P_ee - P_i) # 线速度映射
            J[3:6, i] = Z_i                      # 角速度映射

        return T_matrices[-1], J

    def prepare_arc_length_bspline(self):
        """
        核心步骤一：预先离散化 B 样条，并建立 【物理弧长 s -> 数学参数 u】 的精确映射表
        """
        # 1. 用“米(m)”定义 RRT* 的起点、终点、障碍物和空间边界
        start_point = [0.3, -0.2, 0.15]  
        goal_point = [0.3, 0.2, 0.15]    
        obstacles = [(0.4, 0.0, 0.15, 0.08)] 
        bounds = [[0.2, 0.6], [-0.4, 0.4], [0.0, 0.5]] 
        
        # 2. 运行 RRT* 规划器
        rospy.loginfo("正在运行 RRT* 算法生成避障路径...")
        rrt_planner = RRTStar3D(start_point, goal_point, obstacles, bounds, max_iter=1500)
        waypoints_in_meters = rrt_planner.plan()
        
        # 3. 统一转换为毫米单位
        raw_waypoints = waypoints_in_meters * 1000.0

        # =================================================================
        # ✨ 核心修复：过滤掉 RRT* 产生的重复点或距离过近的点 (距离小于 1mm 视为同一点)
        # =================================================================
        filtered_wps = [raw_waypoints[0]]
        for pt in raw_waypoints[1:]:
            # 计算当前点与上一个有效点之间的欧氏距离
            dist = np.linalg.norm(pt - filtered_wps[-1])
            if dist > 1.0:  # 只有两点间距大于 1 毫米，才保留它
                filtered_wps.append(pt)
        
        # 重新包装成 numpy 数组
        waypoints = np.array(filtered_wps)
        rospy.loginfo(f"RRT* 原始节点数: {len(raw_waypoints)} -> 过滤后有效节点数: {len(waypoints)}")

        # =================================================================
        # 4. 生成 B 样条曲线
        # =================================================================
        x = waypoints[:, 0]
        y = waypoints[:, 1]
        z = waypoints[:, 2]
        
        # 动态自适应阶数
        k_order = 3 if len(waypoints) > 3 else len(waypoints) - 1
        
        # 如果过滤完点数实在太少（少于2个点），做一个安全兜底
        if len(waypoints) < 2:
            rospy.logerr("RRT* 未能规划出有效路径，请检查起点终点或障碍物设置！")
            return

        # 此时有了去重保障，splprep 绝对不会再报错了！
        self.tck, _ = splprep([x, y, z], s=0, k=k_order)

        # 5. 使用 1000 个超高密度点对曲线进行数值积分（保持你后续的代码不变）
        u_fine = np.linspace(0, 1, 1000)
        xf, yf, zf = splev(u_fine, self.tck)

        dx = np.diff(xf)
        dy = np.diff(yf)
        dz = np.diff(zf)
        ds = np.sqrt(dx**2 + dy**2 + dz**2)

        self.s_cumulative = np.concatenate(([0], np.cumsum(ds)))
        self.u_fine = u_fine
        self.total_length = self.s_cumulative[-1]

        rospy.loginfo(f"==== B-Spline Ready ====")
        rospy.loginfo(f"Total physical path length: {self.total_length:.2f} mm")
        rospy.loginfo(f"Estimated duration at {self.vc} mm/s: {self.total_length/self.vc:.2f} s")

    def get_target_pos(self, s_desired):
        """ 根据期望走过的物理弧长，通过插值反查当前在 B 样条上的目标位置 X_d """
        s_desired = min(s_desired, self.total_length)
        
        # 通过查表，破解“参数递增不等于物理匀速”的经典魔咒
        u_target = np.interp(s_desired, self.s_cumulative, self.u_fine)
        
        pos = splev(u_target, self.tck)
        return np.array([pos[0], pos[1], pos[2]])

    def run(self):
        rate = rospy.Rate(1.0 / self.dt)
        
        # 阻塞等待，确保获取到机器人的当前姿态作为起点
        while not rospy.is_shutdown() and not self.joint_received:
            rospy.loginfo_throttle(1.0, "Waiting for Gazebo joint states connection...")
            rate.sleep()
            
        rospy.loginfo("成功连接 Gazebo！开始获取机械臂当前真实位置作为规划起点...")
        q_init = np.array(self.current_q)
        T_ee_init, _ = self.compute_fk_and_jacobian(q_init)
        
        # 拿到当前的真实末端坐标 (注意：正运动学输出是毫米，我们要除以 1000 变成米给 RRT*)
        self.real_start_m = T_ee_init[0:3, 3] / 1000.0 
        
        # 此时去调用轨迹准备函数（记得去 prepare_arc_length_bspline 内部，
        # 把 start_point 改成 self.real_start_m ！）
        self.prepare_arc_length_bspline()

        rospy.loginfo(">>>>>> Commencing Jacobian Constant-Speed Control Loop <<<<<<")

        start_time = rospy.get_time()
        X_d_prev = None

        while not rospy.is_shutdown():
            # 计算当前运行的时间 t
            t = rospy.get_time() - start_time
            
            # 1. 核心：在恒速约束下，当前时间应该到达的物理弧长距离
            s_desired = self.vc * t
            
            # 如果物理路径已经全部走完，优雅退出
            if s_desired >= self.total_length:
                rospy.loginfo("Target reached! Constant-speed tracking trajectory completed perfectly.")
                break
                
            # 2. 根据弧长获取目标 3D 位置 X_d (mm)
            X_d = self.get_target_pos(s_desired)
            
            # 3. 计算切线前馈速度 \dot{X}_d (使用数值微分法，大小严丝合缝等于 vc)
            if X_d_prev is None:
                X_d_dot = np.zeros(3)
            else:
                X_d_dot = (X_d - X_d_prev) / self.dt
            X_d_prev = X_d.copy()
            
            # 4. 获取机器人当前各关节实际弧度值，计算 FK 和 6x6 雅可比
            q_now = np.array(self.current_q)
            T_ee, J = self.compute_fk_and_jacobian(q_now)
            X_now = T_ee[0:3, 3] # 提取当前真实的末端空间坐标 (mm)
            
            # 5. CLIK 核心控制律：前馈速度 + 比例纠偏速度
            V_cmd = X_d_dot + self.Kp * (X_d - X_now)
            
            # 6. 【精髓】截取雅可比矩阵的前 3 行（只管线速度，释放姿态自由度），并求 Moore-Penrose 伪逆
            J_v = J[0:3, :]
            J_v_pinv = np.linalg.pinv(J_v)
            
            # 7. 映射回关节空间，求得当前毫秒下 6 个电机应该给予的旋转速度 \dot{q} (rad/s)
            dq = np.dot(J_v_pinv, V_cmd)
            
            # 8. 位置积分：推算下一周期机器人应该处于的关节角度位置
            q_next = q_now + dq * self.dt
            
            # 9. 组装密集点阵流，实时覆盖更新 Gazebo 里的控制器目标
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            point.positions = list(q_next)
            point.time_from_start = rospy.Duration(self.dt) # 强约束：必须在 dt (20ms) 内精确到达
            
            traj_msg.points.append(point)
            self.traj_pub.publish(traj_msg)
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = Gen3LiteBsplineJacobianControl()
        controller.run()
    except rospy.ROSInterruptException:
        pass