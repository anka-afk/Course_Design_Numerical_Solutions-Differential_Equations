import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


class ProblemConfig:
    """问题配置类"""

    def __init__(self, problem_type=1, D=1.0, u=1.0, l=1.0, T=0.1):
        """
        初始化问题参数

        Args:
            problem_type: 问题类型（1或2）
            D: 扩散系数
            u: 对流速度
            l: 空间域长度
            T: 终止时间
        """
        self.problem_type = problem_type
        self.D = D
        self.u = u
        self.l = l
        self.T = T

    def exact_solution(self, x, t):
        """精确解"""
        if self.problem_type == 1:
            return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
        else:
            return np.exp(-np.pi**2 * t) * np.cos(np.pi * x)

    def source_term(self, x, t):
        """源项f(x,t)"""
        factor = np.exp(-np.pi**2 * t)
        if self.problem_type == 1:
            return -np.pi**2 * (1 - self.D) * factor * np.sin(
                np.pi * x
            ) + np.pi * self.u * factor * np.cos(np.pi * x)
        else:
            return -np.pi**2 * (1 - self.D) * factor * np.cos(
                np.pi * x
            ) - np.pi * self.u * factor * np.sin(np.pi * x)


def explicit_solver(config, h, tau, boundary_handler):
    """
    显式格式求解器

    Args:
        config: 问题配置对象
        h: 空间步长
        tau: 时间步长
        boundary_handler: 边界处理函数

    Returns:
        C: 数值解矩阵 [空间节点数 x 时间步数+1]
        x: 空间网格
        t: 时间网格
    """
    N = int(config.l / h)
    M = int(config.T / tau)
    x = np.linspace(0, config.l, N + 1)
    t = np.linspace(0, config.T, M + 1)

    C = np.zeros((N + 1, M + 1))
    C[:, 0] = config.exact_solution(x, 0)  # 初始条件

    for n in range(M):
        # 内部节点更新
        for i in range(1, N):
            conv = config.u * (C[i + 1, n] - C[i - 1, n]) / (2 * h)
            diff = config.D * (C[i + 1, n] - 2 * C[i, n] + C[i - 1, n]) / h**2
            C[i, n + 1] = C[i, n] + tau * (
                -conv + diff + config.source_term(x[i], t[n])
            )

        # 边界处理
        C = boundary_handler(C, n + 1, config, h)

    return C, x, t


def implicit_solver(config, h, tau, boundary_handler):
    """
    隐式格式求解器

    Args:
        config: 问题配置对象
        h: 空间步长
        tau: 时间步长
        boundary_handler: 边界处理函数

    Returns:
        C: 数值解矩阵
        x: 空间网格
        t: 时间网格
    """
    N = int(config.l / h)
    M = int(config.T / tau)
    x = np.linspace(0, config.l, N + 1)
    t = np.linspace(0, config.T, M + 1)

    # 构造系数矩阵
    main_diag = np.ones(N + 1) * (1 + 2 * config.D * tau / h**2)
    lower_diag = np.ones(N) * (-config.u * tau / (2 * h) - config.D * tau / h**2)
    upper_diag = np.ones(N) * (config.u * tau / (2 * h) - config.D * tau / h**2)
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr")

    C = np.zeros((N + 1, M + 1))
    C[:, 0] = config.exact_solution(x, 0)

    for n in range(M):
        b = C[:, n] + tau * config.source_term(x, t[n])
        C[:, n + 1] = spsolve(A, b)
        C = boundary_handler(C, n + 1, config, h)

    return C, x, t


def crank_nicolson_solver(config, h, tau, boundary_handler):
    """
    Crank-Nicolson格式求解器

    Args:
        config: 问题配置对象
        h: 空间步长
        tau: 时间步长
        boundary_handler: 边界处理函数

    Returns:
        C: 数值解矩阵
        x: 空间网格
        t: 时间网格
    """
    N = int(config.l / h)
    M = int(config.T / tau)
    x = np.linspace(0, config.l, N + 1)
    t = np.linspace(0, config.T, M + 1)

    # 构造系数矩阵
    alpha = config.D * tau / (2 * h**2)
    beta = config.u * tau / (4 * h)
    main_diag = np.ones(N + 1) * (1 + 2 * alpha)
    lower_diag = np.ones(N) * (-beta - alpha)
    upper_diag = np.ones(N) * (beta - alpha)
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format="csr")

    C = np.zeros((N + 1, M + 1))
    C[:, 0] = config.exact_solution(x, 0)

    for n in range(M):
        rhs = C[:, n] + 0.5 * tau * (
            -config.u * (np.roll(C[:, n], -1) - np.roll(C[:, n], 1)) / (2 * h)
            + config.D
            * (np.roll(C[:, n], -1) - 2 * C[:, n] + np.roll(C[:, n], 1))
            / h**2
            + config.source_term(x, t[n])
            + config.source_term(x, t[n + 1])
        )
        C[:, n + 1] = spsolve(A, rhs)
        C = boundary_handler(C, n + 1, config, h)

    return C, x, t


# %% 边界条件处理函数
def dirichlet_boundary(C, n, config, h):
    """Dirichlet边界处理"""
    C[0, n] = 0
    C[-1, n] = 0
    return C


def neumann_simple(C, n, config, h):
    """Neumann边界单侧差分处理"""
    # 左边界: ∂C/∂x = 0
    C[0, n] = C[1, n]
    # 右边界: Dirichlet
    C[-1, n] = 0
    return C


def neumann_ghost(C, n, config, h):
    """Neumann边界虚网格处理"""
    # 左边界虚网格法
    C[0, n] = C[1, n]  # 通过中心差分推导得到
    # 右边界保持Dirichlet
    C[-1, n] = 0
    return C


# %% 误差分析和可视化
def calculate_errors(config, solver, h_list, tau, boundary_handler):
    """
    计算不同空间步长的误差

    Args:
        config: 问题配置
        solver: 求解器函数
        h_list: 空间步长列表
        tau: 固定时间步长
        boundary_handler: 边界处理函数

    Returns:
        errors: L2误差列表
    """
    errors = []
    for h in h_list:
        C, x, t = solver(config, h, tau, boundary_handler)
        exact = config.exact_solution(x, config.T)
        error = np.sqrt(h) * np.linalg.norm(C[:, -1] - exact)
        errors.append(error)
    return errors


def plot_convergence(h_list, errors, title):
    """绘制收敛阶图"""
    plt.loglog(h_list, errors, "o-", label="Numerical")
    fit = np.polyfit(np.log(h_list), np.log(errors), 1)
    plt.loglog(
        h_list, np.exp(fit[1]) * h_list ** fit[0], "--", label=f"Slope={fit[0]:.2f}"
    )
    plt.xlabel("Spatial step h")
    plt.ylabel("L2 Error")
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.show()


# %% 稳定性测试函数
def stability_test(config, solver, h_values, tau_values, boundary_handler):
    """
    稳定性测试

    Args:
        config: 问题配置
        solver: 求解器函数
        h_values: 空间步长列表
        tau_values: 时间步长列表
        boundary_handler: 边界处理函数

    Returns:
        stability_matrix: 稳定性结果矩阵 (0-稳定, 1-不稳定)
    """
    stability = np.zeros((len(h_values), len(tau_values)), dtype=int)

    for i, h in enumerate(h_values):
        for j, tau in enumerate(tau_values):
            try:
                C, _, _ = solver(config, h, tau, boundary_handler)
                if np.any(np.isnan(C)) or np.max(C) > 1e6:
                    stability[i, j] = 1
            except:
                stability[i, j] = 1

    return stability


# %% 主测试程序
if __name__ == "__main__":
    # 测试配置
    config1 = ProblemConfig(problem_type=1, D=1.0, u=1.0)
    config2 = ProblemConfig(problem_type=2, D=1.0, u=1.0)

    # 收敛性测试
    h_list = [1 / 10, 1 / 20, 1 / 40, 1 / 80]
    tau_fixed = 1e-5  # 固定小时间步长

    # 测试问题1显式格式
    errors_exp1 = calculate_errors(
        config1, explicit_solver, h_list, tau_fixed, dirichlet_boundary
    )
    plot_convergence(h_list, errors_exp1, "Problem1 Explicit Spatial Convergence")

    # 测试问题2隐式格式不同边界处理
    errors_imp_simple = calculate_errors(
        config2, implicit_solver, h_list, tau_fixed, neumann_simple
    )
    errors_imp_ghost = calculate_errors(
        config2, implicit_solver, h_list, tau_fixed, neumann_ghost
    )

    plt.loglog(h_list, errors_imp_simple, "s-", label="Simple Neumann")
    plt.loglog(h_list, errors_imp_ghost, "o--", label="Ghost Point")
    plt.xlabel("Spatial step h")
    plt.ylabel("L2 Error")
    plt.title("Problem2 Boundary Treatment Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 稳定性测试
    h_values = [0.1, 0.05, 0.025]
    tau_values = [1e-3, 5e-4, 2e-4]
    stability = stability_test(
        config1, explicit_solver, h_values, tau_values, dirichlet_boundary
    )
    print("Stability Matrix (0=stable, 1=unstable):")
    print(stability)
