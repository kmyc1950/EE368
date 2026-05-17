import numpy as np

def get_modified_dh_matrix(alpha_deg, a, d, theta_deg):
    """
    计算改进型 DH (Modified DH) 变换矩阵
    """
    alpha = np.radians(alpha_deg)
    theta = np.radians(theta_deg)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    # 改进型 DH 公式
    T = np.array([
        [ct,        -st,         0,   a],
        [st * ca,   ct * ca,   -sa,  -d * sa],
        [st * sa,   ct * sa,    ca,   d * ca],
        [0,         0,           0,   1]
    ])
    return T

def forward_kinematics(theta1, theta2, theta3, theta4, theta5, theta6):
    """
    计算正运动学并返回完整的 4x4 变换矩阵
    :param joint_input: 列表 [q1, q2, q3, q4, q5, q6] (单位：度)
    """
    # 1. 应用关节偏移角 (Joint Offsets)
    # 为了使结果与参考矩阵 对齐
    q = np.array([theta1, theta2, theta3, theta4, theta5, theta6], dtype=float)
    q[1] += 90.0   # theta2 偏移
    q[2] += 90.0  # theta3 偏移
    q[3] += 90.0  # theta4 偏移

    # 2. 修正后的 DH 参数 [alpha_{i-1}, a_{i-1}, d_i]
    dh_params = [
        [0,    0,     243.3],  # 关节 1
        [90,  0,     30],     # 关节 2
        [180,  280,   20],     # 关节 3
        [90,  0,     245],    # 关节 4
        [90,  0,     57],     # 关节 5
        [-90,   0,     235]     # 关节 6
    ]

    T_final = np.eye(4)

    # 3. 计算 0-6 变换
    for i in range(6):
        alpha, a, d = dh_params[i]
        T_i = get_modified_dh_matrix(alpha, a, d, q[i])
        T_final = np.dot(T_final, T_i)
    
    return T_final

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)

    print("====================================================")
    print("      机器人正运动学手动输入测试工具")
    print("====================================================")
    print("提示：请输入 6 个数字（如：0 345 75 0 300 0），按回车确认")
    
    try:
        raw_input = input("请输入角度: ").strip()
        # 自动处理多个空格的情况
        angles = [float(x) for x in raw_input.split() if x]

        if len(angles) != 6:
            print(f"输入错误：需要 6 个角度，但你输入了 {len(angles)} 个。")
        else:
            # 解包调用：将 angles 里的 6 个数分别传给函数的 6 个参数
            T_res = forward_kinematics(*angles)
            
            print("\n" + "-"*20 + " 计算结果 " + "-"*20)
            print("4x4 齐次变换矩阵 T:")
            print(T_res)
            

    except ValueError:
        print("错误：请输入有效的数字，不要包含字母或特殊符号。")