import matplotlib.pyplot as plt
import numpy as np

#%% 误差分析和可视化
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
        error = np.sqrt(h)*np.linalg.norm(C[:,-1] - exact)
        errors.append(error)
    return errors

def plot_convergence(h_list, errors, title):
    """绘制收敛阶图"""
    plt.loglog(h_list, errors, 'o-', label='Numerical')
    fit = np.polyfit(np.log(h_list), np.log(errors), 1)
    plt.loglog(h_list, np.exp(fit[1])*h_list**fit[0], '--', 
               label=f'Slope={fit[0]:.2f}')
    plt.xlabel('Spatial step h')
    plt.ylabel('L2 Error')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.show()


#%% 稳定性测试函数
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
                    stability[i,j] = 1
            except:
                stability[i,j] = 1
                
    return stability