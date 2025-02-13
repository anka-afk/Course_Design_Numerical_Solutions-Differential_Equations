import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

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
    x = np.linspace(0, config.l, N+1)
    t = np.linspace(0, config.T, M+1)
    
    C = np.zeros((N+1, M+1))
    C[:,0] = config.exact_solution(x, 0)  # 初始条件
    
    for n in range(M):
        # 内部节点更新
        for i in range(1, N):
            conv = config.u * (C[i+1,n] - C[i-1,n]) / (2*h)
            diff = config.D * (C[i+1,n] - 2*C[i,n] + C[i-1,n]) / h**2
            C[i,n+1] = C[i,n] + tau*(-conv + diff + config.source_term(x[i], t[n]))
        
        # 边界处理
        C = boundary_handler(C, n+1, config, h)
        
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
    x = np.linspace(0, config.l, N+1)
    t = np.linspace(0, config.T, M+1)
    
    # 构造系数矩阵
    main_diag = np.ones(N+1) * (1 + 2*config.D*tau/h**2)
    lower_diag = np.ones(N) * (-config.u*tau/(2*h) - config.D*tau/h**2)
    upper_diag = np.ones(N) * (config.u*tau/(2*h) - config.D*tau/h**2)
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    
    C = np.zeros((N+1, M+1))
    C[:,0] = config.exact_solution(x, 0)
    
    for n in range(M):
        b = C[:,n] + tau*config.source_term(x, t[n])
        C[:,n+1] = spsolve(A, b)
        C = boundary_handler(C, n+1, config, h)
        
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
    x = np.linspace(0, config.l, N+1)
    t = np.linspace(0, config.T, M+1)
    
    # 构造系数矩阵
    alpha = config.D * tau / (2*h**2)
    beta = config.u * tau / (4*h)
    main_diag = np.ones(N+1) * (1 + 2*alpha)
    lower_diag = np.ones(N) * (-beta - alpha)
    upper_diag = np.ones(N) * (beta - alpha)
    A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')
    
    C = np.zeros((N+1, M+1))
    C[:,0] = config.exact_solution(x, 0)
    
    for n in range(M):
        rhs = C[:,n] + 0.5*tau*(
            -config.u*(np.roll(C[:,n], -1) - np.roll(C[:,n], 1))/(2*h) +
            config.D*(np.roll(C[:,n], -1) - 2*C[:,n] + np.roll(C[:,n], 1))/h**2 +
            config.source_term(x, t[n]) + config.source_term(x, t[n+1])
        )
        C[:,n+1] = spsolve(A, rhs)
        C = boundary_handler(C, n+1, config, h)
        
    return C, x, t