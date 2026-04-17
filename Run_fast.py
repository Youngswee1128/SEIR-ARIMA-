import numpy as np
import numba
# ==========================================
# 1. Numba JIT 加速的龙格库塔 (RK4) 求解器
# ==========================================

@numba.jit(nopython=True)
def rk4_step(ode_func, y, t, dt, params):
    """
    执行单步 RK4 积分
    ode_func: 用户定义的微分方程函数 (必须也是 @jit 编译过的)
    """
    k1 = ode_func(y, t, params)
    k2 = ode_func(y + 0.5 * dt * k1, t + 0.5 * dt, params)
    k3 = ode_func(y + 0.5 * dt * k2, t + 0.5 * dt, params)
    k4 = ode_func(y + dt * k3, t + dt, params)
    
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@numba.jit(nopython=True)
def solve_ode_numba(ode_func, y0, t_eval, params):
    """
    高性能 ODE 求解器
    ode_func: JIT 编译的方程函数
    y0: 初始条件数组
    t_eval: 时间点数组
    params: 参数数组
    """
    n_steps = len(t_eval)
    n_vars = len(y0)
    result = np.zeros((n_steps, n_vars))
    
    result[0] = y0
    curr_y = y0
    
    for i in range(1, n_steps):
        t_curr = t_eval[i-1]
        t_next = t_eval[i]
        dt = t_next - t_curr
        
        # 为了提高精度，可以将大步长切分为小步长（这里为了演示简单，直接用一步）
        # 实际应用中建议在这里加一个子循环，例如把 dt 切分成 10 份
        sub_steps = 10
        dt_sub = dt / sub_steps
        for _ in range(sub_steps):
            curr_y = rk4_step(ode_func, curr_y, t_curr, dt_sub, params)
            t_curr += dt_sub
            
        result[i] = curr_y
        
    return result