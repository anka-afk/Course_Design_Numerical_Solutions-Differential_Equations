#%% 边界条件处理函数
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