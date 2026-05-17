import numpy as np
import math

class Gen3LiteKinematics:
    def __init__(self):
        # 记录改进型 DH 参数：[alpha_{i-1}, a_{i-1}, d_i, theta_offset]
        # 角度均为度数，长度单位为毫米 (mm)
        self.dh_params = [
            [0,    0,     243.3, 0],   # 关节 1
            [90,   0,     30,    90],  # 关节 2
            [180,  280,   20,    90],  # 关节 3
            [90,   0,     245,   90],  # 关节 4
            [90,   0,     57,    0],   # 关节 5
            [-90,  0,     235,   0]    # 关节 6
        ]

    def get_mdh_matrix(self, alpha_deg, a, d, theta_deg):
        """
        计算单个关节的 4x4 改进型 DH 变换矩阵 (Modified DH)
        """
        alpha = math.radians(alpha_deg)
        theta = math.radians(theta_deg)
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(theta), math.sin(theta)

        T = np.array([
            [ct,        -st,         0,   a],
            [st * ca,   ct * ca,   -sa,  -d * sa],
            [st * sa,   ct * sa,    ca,   d * ca],
            [0,         0,           0,   1]
        ])
        return T

    def compute_fk_and_jacobian(self, joint_angles_deg):
        """
        输入：6 个关节的角度 (度)
        输出：末端齐次变换矩阵 T_ee (4x4), 几何雅可比矩阵 J (6x6)
        """
        T_matrices = []
        T_cur = np.eye(4)

        # 1. 逐个关节连乘，求出所有的 T_0^i 矩阵
        for i in range(6):
            alpha = self.dh_params[i][0]
            a     = self.dh_params[i][1]
            d     = self.dh_params[i][2]
            offset= self.dh_params[i][3]

            q_i = joint_angles_deg[i] + offset
            T_i = self.get_mdh_matrix(alpha, a, d, q_i)
            
            T_cur = np.dot(T_cur, T_i)
            T_matrices.append(T_cur)

        # 2. 提取末端执行器的位置 (第 6 个坐标系的原点)
        P_ee = T_matrices[-1][0:3, 3]

        # 3. 初始化 6x6 的雅可比矩阵
        J = np.zeros((6, 6))

        # 4. 根据改进型 DH (Modified DH) 构造雅可比矩阵
        for i in range(6):
            # 在 MDH 中，关节 i 的旋转轴正是 T_0^i 的 Z 轴
            Z_i = T_matrices[i][0:3, 2]  # 提取第 i 个坐标系的 Z 轴方向向量
            P_i = T_matrices[i][0:3, 3]  # 提取第 i 个坐标系的原点坐标

            # 线速度部分 (J_v = Z_i × (P_ee - P_i))
            J[0:3, i] = np.cross(Z_i, P_ee - P_i)
            
            # 角速度部分 (J_w = Z_i)
            J[3:6, i] = Z_i

        return T_matrices[-1], J

if __name__ == "__main__":
    # 设置打印格式，避免科学计数法，保留4位小数
    np.set_printoptions(precision=4, suppress=True)

    print("====================================================")
    print("      Kinova Gen3-lite 雅可比矩阵计算测试")
    print("====================================================")
    
    kinematics = Gen3LiteKinematics()

    try:
        raw_input = input("请输入 6 个关节角度 (空格分隔，如 0 0 0 0 0 0): ").strip()
        angles = [float(x) for x in raw_input.split() if x]

        if len(angles) != 6:
            print(f"输入错误：需要 6 个角度，但收到了 {len(angles)} 个。")
        else:
            T_ee, J_matrix = kinematics.compute_fk_and_jacobian(angles)
            
            print("\n" + "-"*20 + " 1. 末端位姿 (T_ee) " + "-"*20)
            print(T_ee)
            
            print("\n" + "-"*20 + " 2. 几何雅可比矩阵 (J) 6x6 " + "-"*20)
            print(J_matrix)
            
            # 物理意义测试：求伪逆
            print("\n" + "-"*20 + " 3. 雅可比伪逆矩阵 (J_pinv) " + "-"*20)
            J_pinv = np.linalg.pinv(J_matrix)
            print(J_pinv)
            print("\n💡 提示：这说明当你需要末端 XYZ 移动时，可以用 [角速度] = J_pinv @ [笛卡尔速度] 算出来啦！")

    except ValueError:
        print("错误：请输入有效的数字。")