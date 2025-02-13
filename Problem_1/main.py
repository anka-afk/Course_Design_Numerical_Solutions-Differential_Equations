from Problem_1.ProblemConfig import ProblemConfig
from Problem_1.Solver import *
from Problem_1.Boundary import *
from Problem_1.Test_Utils import *

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
