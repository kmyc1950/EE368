#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import math
import sys

# 导入 ROS 官方的控制消息类型
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

def move_robot(target_angles_deg):
    """
    控制机械臂移动到指定关节角度
    :param target_angles_deg: 包含 6 个关节角度的列表 (单位: 度)
    """
    # 1. 初始化 ROS 节点
    rospy.init_node('kinova_joint_tester', anonymous=True)

    # 2. 连接到 Kinova 的轨迹控制 Action Server
    # 注意：根据你之前代码的命名空间，前缀应该是 /my_gen3_lite
    # 具体的 controller 名字在不同的 launch 文件中可能略有不同，通常是 gen3_lite_joint_trajectory_controller
    action_topic = '/my_gen3_lite/gen3_lite_joint_trajectory_controller/follow_joint_trajectory'
    
    rospy.loginfo(f"正在连接到 Action Server: {action_topic} ...")
    client = actionlib.SimpleActionClient(action_topic, FollowJointTrajectoryAction)
    
    # 等待服务器响应 (最多等 10 秒)
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("无法连接到机械臂的控制器，请检查机械臂驱动或 Gazebo 是否已启动！")
        return

    rospy.loginfo("连接成功！准备发送运动指令...")

    # 3. 创建目标 (Goal)
    goal = FollowJointTrajectoryGoal()
    
    # 设置关节名称 (Kinova gen3_lite 的标准关节名称)
    goal.trajectory.joint_names = [
        'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'
    ]

    # 4. 创建一个轨迹点 (Trajectory Point)
    point = JointTrajectoryPoint()
    
    # 将输入的角度 (度) 转换为弧度 (Radian)，因为 ROS 底层只认弧度
    point.positions = [math.radians(angle) for angle in target_angles_deg]
    
    # 【核心安全设置】：告诉机械臂花费多长时间到达这个点（例如 5 秒）
    # 时间设置得越长，机械臂动得越慢、越安全
    point.time_from_start = rospy.Duration(5.0)

    # 将轨迹点加入到轨迹中
    goal.trajectory.points.append(point)

    # 5. 发送指令并等待完成
    rospy.loginfo(f"目标角度 (度): {target_angles_deg}")
    client.send_goal(goal)
    
    rospy.loginfo("机械臂正在移动中...")
    client.wait_for_result()
    
    rospy.loginfo("移动完成！到达目标位置。")


if __name__ == '__main__':
    print("====================================================")
    print("      Kinova Gen3-lite ROS 1 关节控制测试工具")
    print("====================================================")
    print("提示：请输入 6 个数字（如：0 -30 75 0 -45 0），按回车确认")
    
    try:
        raw_input = input("请输入目标关节角度: ").strip()
        angles = [float(x) for x in raw_input.split() if x]

        if len(angles) != 6:
            print(f"输入错误：需要 6 个角度，但你输入了 {len(angles)} 个。")
            sys.exit(1)
            
        # 调用控制函数
        move_robot(angles)

    except ValueError:
        print("错误：请输入有效的数字。")
    except rospy.ROSInterruptException:
        print("程序被用户中断。")