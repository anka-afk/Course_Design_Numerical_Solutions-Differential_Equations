import numpy as np

class ProblemConfig:
    """
    问题配置类
    """
    def __init__(self, problem_type=1,D=1.0,u=1.0,l=1.0,T=0.1):
        """初始化问题参数

        Args:
            problem_type (int, optional): 问题类型(1或2). Defaults to 1.
            D (float, optional): 扩散系数. Defaults to 1.0.
            u (float, optional): 对流速度. Defaults to 1.0.
            l (float, optional): 空间域长度. Defaults to 1.0.
            T (float, optional): 终止时间. Defaults to 0.1.
        """
        self.problem_type = problem_type
        self.D = D
        self.u = u
        self.l = l
        self.T = T
        
        def exact_solution(self, x ,t):
            """精确解

            Args:
                x (float): 传入自变量距离x
                t (float): 传入自变量时间t
            """
            if self.problem_type == 1:
                return np.exp(-np.pi**2*t) * np.sin(np.pi*x)
            else:
                return np.esp(-np.pi**2*t) * np.cos(np.pi*x)
        
        def source_term(self, x, t):
            """源项: 代入精确解后的方程右端

            Args:
                x (float): 传入自变量距离x
                t (float): 传入自变量时间t
            """
            factor = np.exp(-np.pi**2*t)
            if self.problem_type == 1:
                return (-np.pi**2*(1-self.D)*factor*np.sin(np.pi*x) + np.pi*self.u*factor*np.sin(np.pi*x))
            else:
                return (-np.pi**2*(1-self.D)*factor*np.cos(np.pi*x) + np.pi*self.u*factor*np.sin(np.pi*x))
            