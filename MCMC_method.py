import numpy as np
import numba
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from copy import deepcopy
import time
from tqdm import tqdm
from Run_fast import solve_ode_numba
# ==========================================
# 2. 贝叶斯参数估计主类
# ==========================================

class BayesianODEFitter:
    def __init__(self, ode_func, time_points, observed_data):
        """
        初始化拟合器
        :param ode_func: 被 @numba.jit 装饰的 ODE 函数
        :param time_points: 时间序列 (numpy array)
        :param observed_data: 观测数据字典 {'CompartmentName': data_array}
        """
        self.ode_func = ode_func
        self.time_points = np.array(time_points, dtype=np.float64)
        self.observed_data = observed_data
        
        # 配置存储
        self.compartment_names = []
        self.fit_targets = {}  # 映射: {'I': 'InfectedData'}
        self.likelihood_type = 'NegativeBinomial' # 默认
        
        # 参数存储结构
        # param_config = {
        #   'name': {'type': 'fixed'/'estimated', 'value': val, 'prior': dict}
        # }
        self.parameters_config = {}
        self.initial_conditions_config = {}
        
        # MCMC 运行时状态
        self.chain_history = []  # 存放多条链的结果
        self.param_names_estimated = [] # 待估参数名称列表（有序）
        
    def set_compartments(self, names):
        """设置仓室名称列表，顺序必须与 ode_func 返回的顺序一致"""
        self.compartment_names = names
        
    def set_fit_targets(self, targets_dict, target_types=None):
        """
        设置拟合目标
        :param targets_dict: 例如 {'I': observed_infected_array}
        :param target_types: 可选，字典 {'I': 'val'} 或 {'I': 'diff'}
                             'val': 直接拟合数值 (默认)
                             'diff': 拟合差分 (np.diff) - 通常用于拟合每日新增
        """
        self.fit_targets = targets_dict
        if target_types is None:
            self.fit_target_types = {k: 'val' for k in targets_dict.keys()}
        else:
            self.fit_target_types = target_types
        
    def add_parameter(self, name, value=None, type='fixed', prior=None):
        """
        添加参数
        :param type: 'fixed' 或 'estimated'
        :param value: 固定值 或 初始猜测值
        :param prior: 字典, e.g. {'dist': 'normal', 'mu': 0.5, 'sigma': 0.1}
        """
        self.parameters_config[name] = {
            'type': type,
            'value': value,
            'prior': prior
        }
        
    def set_initial_conditions(self, ic_dict):
        """
        设置初值条件
        格式: {'S': 990, 'I': {'type': 'estimated', 'value': 10, 'prior': ...}}
        或者简单格式: {'S': 990, 'I': 10}
        """
        for name, config in ic_dict.items():
            if isinstance(config, dict):
                # 如果初值也是待估参数
                self.initial_conditions_config[name] = config
            else:
                # 固定初值
                self.initial_conditions_config[name] = {'type': 'fixed', 'value': config}

    def set_likelihood(self, distribution='NegativeBinomial', **kwargs):
        """
        设置似然函数类型
        支持: 'NegativeBinomial', 'Poisson', 'Normal', 'ZeroInflatedNegativeBinomial'
        """
        self.likelihood_type = distribution
        self.likelihood_kwargs = kwargs
        
        # 如果是负二项分布，通常需要估计离散度参数 (dispersion/r)
        # 我们自动添加一个名为 'dispersion' 的待估参数，除非用户已经在 add_parameter 中添加了
        if distribution in ['NegativeBinomial', 'ZeroInflatedNegativeBinomial']:
            if 'dispersion' not in self.parameters_config:
                print("Hint: Adding default 'dispersion' parameter for NB distribution.")
                self.add_parameter('dispersion', value=1.0, type='estimated', 
                                   prior={'dist': 'gamma', 'alpha': 2.0, 'beta': 0.1})

    def _prepare_vectors(self):
        """
        (内部方法) 准备 MCMC 所需的向量映射
        """
        self.param_names_estimated = []
        self.param_names_fixed = []
        self.param_values_fixed = []
        
        # 处理普通参数
        for name, cfg in self.parameters_config.items():
            if cfg['type'] == 'estimated':
                self.param_names_estimated.append(name)
            else:
                self.param_names_fixed.append(name)
                self.param_values_fixed.append(cfg['value'])
                
        # 处理初值参数 (如果有待估初值)
        self.ic_names_estimated = []
        self.ic_fixed_map = {} # {'S': 990, ...}
        
        for name, cfg in self.initial_conditions_config.items():
            if cfg['type'] == 'estimated':
                # 将初值参数名标记为 "IC_Name" 防止重名
                p_name = f"IC_{name}"
                self.param_names_estimated.append(p_name)
                self.ic_names_estimated.append(name)
            else:
                self.ic_fixed_map[name] = cfg['value']
                
        self.n_dim = len(self.param_names_estimated)
        self.param_values_fixed = np.array(self.param_values_fixed, dtype=np.float64)

    def _assemble_params_and_ic(self, theta_estimated):
        """
        (内部方法) 将 MCMC 采样的 theta 向量拼装成 ODE 需要的完整参数和初值
        """
        # 1. 拆解 theta
        # 字典映射当前采样值
        current_vals = {name: val for name, val in zip(self.param_names_estimated, theta_estimated)}
        
        # 2. 组装 ODE 参数数组 (需要严格按照 add_parameter 的顺序或者 ode_func 内部约定的顺序?)
        # 假设 ode_func 接收一个数组，顺序取决于 self.parameters_config 的 key 顺序
        # 为了安全，我们需要根据 keys 的顺序重建数组
        
        full_params = []
        # 注意：这里假设 parameters_config 的顺序就是 ode_func 期望的参数顺序
        # 排除掉 'dispersion' 这种统计参数，它不进 ODE
        ode_param_names = [k for k in self.parameters_config.keys() if k != 'dispersion']
        
        for name in ode_param_names:
            if self.parameters_config[name]['type'] == 'estimated':
                full_params.append(current_vals[name])
            else:
                full_params.append(self.parameters_config[name]['value'])
                
        # 3. 组装 Initial Conditions
        y0 = []
        for name in self.compartment_names:
            if name in self.ic_names_estimated:
                y0.append(current_vals[f"IC_{name}"])
            else:
                y0.append(self.ic_fixed_map[name])
                
        return np.array(y0, dtype=np.float64), np.array(full_params, dtype=np.float64), current_vals

    def _log_prior(self, theta):
        """计算先验概率对数和"""
        log_p = 0.0
        for i, name in enumerate(self.param_names_estimated):
            val = theta[i]
            
            # 查找该参数的配置（注意区分 IC 和普通参数）
            if name.startswith("IC_"):
                orig_name = name.split("IC_")[1]
                config = self.initial_conditions_config[orig_name]
            else:
                config = self.parameters_config[name]
            
            prior = config['prior']
            
            # 分布计算
            if prior['dist'] == 'uniform':
                if not (prior['lower'] <= val <= prior['upper']):
                    return -np.inf
                # uniform 的 logpdf 是常数，可以忽略
            elif prior['dist'] == 'normal':
                log_p += stats.norm.logpdf(val, loc=prior['mu'], scale=prior['sigma'])
            elif prior['dist'] == 'truncated_normal':
                if not (prior['lower'] <= val <= prior['upper']):
                    return -np.inf
                log_p += stats.truncnorm.logpdf(val, a=(prior['lower']-prior['mu'])/prior['sigma'],
                                                b=(prior['upper']-prior['mu'])/prior['sigma'],
                                                loc=prior['mu'], scale=prior['sigma'])
            elif prior['dist'] == 'gamma':
                if val <= 0: return -np.inf
                log_p += stats.gamma.logpdf(val, a=prior['alpha'], scale=1/prior['beta'])
            elif prior['dist'] == 'lognormal':
                # 参数: mu (对数均值), sigma (对数标准差)
                if val <= 0: return -np.inf
                log_p += stats.lognorm.logpdf(val, s=prior['sigma'], scale=np.exp(prior['mu']))
            elif prior['dist'] == 'beta':
                # 参数: alpha, beta, lower (可选,默认0), upper (可选,默认1)
                lower = prior.get('lower', 0)
                upper = prior.get('upper', 1)
                if not (lower < val < upper): return -np.inf
                # 标准化到 [0,1]
                val_std = (val - lower) / (upper - lower)
                log_p += stats.beta.logpdf(val_std, a=prior['alpha'], b=prior['beta'])
                log_p -= np.log(upper - lower)  # Jacobian
            elif prior['dist'] == 'exponential':
                # 参数: rate (lambda)
                if val < 0: return -np.inf
                log_p += stats.expon.logpdf(val, scale=1/prior['rate'])
            else:
                raise ValueError(f"Unknown prior distribution: {prior['dist']}")
                
        return log_p

    def _log_likelihood(self, y_model, current_vals):
        """计算似然函数"""
        log_l = 0.0
        
        # 获取离散度参数 (如果是 NB 分布)
        dispersion = current_vals.get('dispersion', 1.0)
        
        for comp_name, obs_data in self.fit_targets.items():
            # 找到对应仓室在结果矩阵中的列索引
            idx = self.compartment_names.index(comp_name)
            model_data = y_model[:, idx]
            
            # 处理差分拟合 (Incidence)
            if self.fit_target_types.get(comp_name, 'val') == 'diff':
                # 对 model_data 进行差分，得到每个时间步的增量
                # 注意: obs_data 的长度应该匹配 time_points 的长度
                # 假设 obs_data[i] 对应 t[i] 发生的事件数，通常 ODE 算出来的是累积量
                # model_inc[i] = model_cum[i] - model_cum[i-1]
                # 这里为了简单，我们假定 obs_data[0] 对应 t[0]~t[1] 还是 t[-1]~t[0]?
                # 通常的做法: prepend=model_data[0] 使得第一个点是0或者保持初始值
                # 或者 diff 之后长度少 1，需要根据 obs_data 对齐
                
                # 策略: 使用 np.diff(model_data, prepend=model_data[0]) 保持长度一致
                # 物理含义: 第一个点的增量是 model_data[0] - model_data[0] = 0 (如果是累积量)
                # 这种方式可能导致第一个点拟合不上。
                
                # 另一种常见策略: obs_data 对应的是区间 [t_i-1, t_i] 的增量
                # model_inc[i] = model_data[i] - model_data[i-1] (i>0)
                # i=0 时设为 0 或 model_data[0]
                
                model_diff = np.diff(model_data, prepend=model_data[0])
                model_data = model_diff

            # 避免数值错误 (模型产生负数或零)
            model_data = np.maximum(model_data, 1e-6)
            
            if self.likelihood_type == 'NegativeBinomial':
                # 参数化: n=dispersion, p = n / (n + mu)
                n = dispersion
                p = n / (n + model_data)
                log_l += np.sum(stats.nbinom.logpmf(obs_data, n, p))
                
            elif self.likelihood_type == 'Normal':
                sigma = current_vals.get('sigma', 1.0) # 假设有个 sigma 参数
                log_l += np.sum(stats.norm.logpdf(obs_data, loc=model_data, scale=sigma))
                
            elif self.likelihood_type == 'Poisson':
                log_l += np.sum(stats.poisson.logpmf(obs_data, mu=model_data))
                
        return log_l

    def _posterior(self, theta):
        """计算后验概率 (Log Posterior)"""
        # 1. Prior
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        
        # 2. Solve ODE
        y0, ode_params, current_vals = self._assemble_params_and_ic(theta)
        
        # 调用 Numba 求解器
        try:
            y_model = solve_ode_numba(self.ode_func, y0, self.time_points, ode_params)
        except Exception as e:
            print(f"Error in ODE solver: {e}")
            return -np.inf # 求解失败（例如参数导致发散）
            
        # 3. Likelihood
        ll = self._log_likelihood(y_model, current_vals)
        
        return lp + ll
    
    def _validate_config(self):
        """
        验证配置是否完整
        在 run_mcmc 之前调用，提前发现配置错误
        """
        errors = []
        
        # 检查仓室是否已设置
        if not self.compartment_names:
            errors.append("Compartments not set. Call set_compartments() first.")
        
        # 检查拟合目标是否已设置
        if not self.fit_targets:
            errors.append("Fit targets not set. Call set_fit_targets() first.")
        
        # 检查是否有待估参数 (从 config 中检查)
        has_estimated = False
        for name, cfg in self.parameters_config.items():
            if cfg.get('type') == 'estimated':
                has_estimated = True
                if not cfg.get('prior'):
                    errors.append(f"Parameter '{name}' is estimated but has no prior distribution.")
        
        for comp, cfg in self.initial_conditions_config.items():
            if isinstance(cfg, dict) and cfg.get('type') == 'estimated':
                has_estimated = True
                if not cfg.get('prior'):
                    errors.append(f"Initial condition 'IC_{comp}' is estimated but has no prior distribution.")
        
        if not has_estimated:
            errors.append("No parameters to estimate. Use type='estimated' when adding parameters.")
        
        # 检查初值条件是否覆盖所有仓室
        if self.compartment_names:
            for comp in self.compartment_names:
                if comp not in self.initial_conditions_config:
                    errors.append(f"Initial condition for compartment '{comp}' not set.")
        
        if errors:
            error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        return True
    
    def _sample_from_prior(self, name, prior):
        """
        从先验分布中采样一个值
        用于初始化 MCMC 链
        """
        dist = prior['dist']
        
        if dist == 'uniform':
            return np.random.uniform(prior['lower'], prior['upper'])
        elif dist == 'normal':
            return np.random.normal(prior['mu'], prior['sigma'])
        elif dist == 'truncated_normal':
            a = (prior['lower'] - prior['mu']) / prior['sigma']
            b = (prior['upper'] - prior['mu']) / prior['sigma']
            return stats.truncnorm.rvs(a, b, loc=prior['mu'], scale=prior['sigma'])
        elif dist == 'gamma':
            return np.random.gamma(prior['alpha'], 1/prior['beta'])
        elif dist == 'lognormal':
            return np.random.lognormal(prior['mu'], prior['sigma'])
        elif dist == 'beta':
            lower = prior.get('lower', 0)
            upper = prior.get('upper', 1)
            val_std = np.random.beta(prior['alpha'], prior['beta'])
            return lower + val_std * (upper - lower)
        elif dist == 'exponential':
            return np.random.exponential(1/prior['rate'])
        else:
            # 默认回退到初始值
            return None

    def run_mcmc(self, n_iter=10000, n_chains=1, burn_in=2000, adapt_step=True, use_prior_init=True):
        """
        执行 MCMC (Metropolis-Hastings with Adaptive Step)
        :param n_iter: 迭代次数
        :param n_chains: 链数
        :param burn_in: 燃烧期
        :param adapt_step: 是否自适应步长
        :param use_prior_init: 是否从先验分布采样初始化 (默认 True)
        """
        # 验证配置
        self._validate_config()
        
        self._prepare_vectors()
        print(f"Starting MCMC: {n_chains} chains, {n_iter} iterations.")
        print(f"Estimated Parameters: {self.param_names_estimated}")
        
        self.chain_history = []
        self.use_prior_init = use_prior_init
        
        # 为简单起见，这里使用循环跑多链（实际项目可用 multiprocessing）
        for chain_idx in range(n_chains):
            print(f"Running Chain {chain_idx+1}...")
            chain_samples, chain_log_probs = self._run_single_chain(n_iter, adapt_step)
            
            # 去除 Burn-in
            self.chain_history.append({
                'samples': chain_samples[burn_in:],
                'log_probs': chain_log_probs[burn_in:]
            })
            
        print("MCMC Completed.")

    def _run_single_chain(self, n_iter, adapt_step):
        """单链运行逻辑"""
        
        def nearest_positive_definite(A):
            """
            Higham (1988) 算法: 找到最近的对称正定矩阵
            理论依据: N.J. Higham, "Computing a nearest symmetric positive semidefinite matrix"
            Linear Algebra and its Applications, 103:103-118, 1988
            """
            B = (A + A.T) / 2  # 对称化
            _, s, V = np.linalg.svd(B)
            H = V.T @ np.diag(s) @ V
            A2 = (B + H) / 2
            A3 = (A2 + A2.T) / 2  # 确保对称
            
            if np.all(np.linalg.eigvalsh(A3) > 0):
                return A3
            
            # 如果仍不正定，添加小扰动 (Tikhonov 正则化)
            spacing = np.spacing(np.linalg.norm(A))
            I = np.eye(A.shape[0])
            k = 1
            while k <= 100:
                mineig = np.min(np.real(np.linalg.eigvalsh(A3)))
                if mineig > 0:
                    break
                A3 += I * (-mineig * k**2 + spacing)
                k += 1
            return A3
        
        # 初始化
        n_dim = self.n_dim
        
        # 初始化起始点
        current_theta = []
        for name in self.param_names_estimated:
            if name.startswith('IC_'):
                orig = name.split('IC_')[1]
                config = self.initial_conditions_config[orig]
            else:
                config = self.parameters_config[name]
            
            init_val = config['value']
            
            # 如果启用先验采样初始化
            if self.use_prior_init and config.get('prior'):
                sampled = self._sample_from_prior(name, config['prior'])
                if sampled is not None:
                    init_val = sampled
            else:
                # 从初始值附近扰动
                init_val = init_val * np.random.uniform(0.9, 1.1)
            
            current_theta.append(init_val)
        
        current_theta = np.array(current_theta)
        current_log_post = self._posterior(current_theta)
        
        # 存储
        samples = np.zeros((n_iter, n_dim))
        log_probs = np.zeros(n_iter)
        
        # 自适应步长配置
        # 使用参数值比例设置初始步长 (每个参数步长约为其初始值的 1%)
        init_stds = np.array(current_theta) * 0.01
        init_stds = np.maximum(init_stds, 1e-6)  # 防止零值
        cov = np.diag(init_stds ** 2)  # 初始协方差矩阵
        adapt_interval = 50       # 每隔多少步更新一次协方差
        
        accepted = 0
        
        # 目标接受率 (对于高维 Metropolis，理论最优约为 0.234)
        target_accept_rate = 0.25
        # 全局缩放因子，用于动态调整步长
        global_scale = 1.0
        
        # Cholesky 缓存 (仅在协方差更新时重新计算)
        L_cached = None
        cov_needs_update = True
        
        for i in tqdm(range(n_iter), desc="Sampling", leave=False):
            # 提议分布 (Proposal)
            # 使用缓存的 Cholesky 分解，减少计算量
            if cov_needs_update:
                scaled_cov = nearest_positive_definite(cov * global_scale)
                try:
                    L_cached = np.linalg.cholesky(scaled_cov)
                except np.linalg.LinAlgError:
                    # Cholesky 失败时，使用对角线
                    L_cached = np.diag(np.sqrt(np.diag(scaled_cov)))
                cov_needs_update = False
            
            proposal = current_theta + L_cached @ np.random.randn(n_dim)
            
            proposal_log_post = self._posterior(proposal)
            
            # 接受率
            if np.isfinite(proposal_log_post):
                alpha = proposal_log_post - current_log_post
                if np.log(np.random.rand()) < alpha:
                    current_theta = proposal
                    current_log_post = proposal_log_post
                    accepted += 1
            
            samples[i] = current_theta
            log_probs[i] = current_log_post
            
            # 自适应更新 (Robbins-Monro 风格)
            if adapt_step and i > 100 and i % adapt_interval == 0:
                # 当前接受率
                current_rate = accepted / (i + 1)
                
                # 使用 Robbins-Monro 更新: 根据与目标接受率的差异调整
                # 学习率随迭代次数衰减
                learn_rate = 1.0 / (1 + i / 1000)
                
                # 如果当前接受率 < 目标，缩小步长; 如果 > 目标，增大步长
                adjustment = np.exp(learn_rate * (current_rate - target_accept_rate) / target_accept_rate)
                global_scale *= adjustment
                
                # 限制缩放因子范围，防止过大或过小
                global_scale = np.clip(global_scale, 0.01, 100.0)
                
                # 同时更新协方差矩阵结构 (学习参数间相关性)
                param_hist = samples[:i]
                new_cov = np.cov(param_hist, rowvar=False)
                
                # 确保协方差矩阵对称正定
                new_cov = (new_cov + new_cov.T) / 2  # 确保对称
                # 使用特征值分解确保正定
                eigvals, eigvecs = np.linalg.eigh(new_cov)
                eigvals = np.maximum(eigvals, 1e-8)  # 确保所有特征值为正
                new_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                scale_factor = (2.38 ** 2) / n_dim
                cov = new_cov * scale_factor
                
                # 标记需要重新计算 Cholesky
                cov_needs_update = True
                
        print(f"Chain finished. Acceptance rate: {accepted/n_iter:.2f}")
        return samples, log_probs

    # ==========================
    # 诊断与绘图模块
    # ==========================
    
    def summary_statistics(self, credible_interval=0.95):
        """
        计算并输出参数估计的统计量
        :param credible_interval: 可信区间水平 (默认 95%)
        :return: DataFrame 包含统计量
        """
        # 合并所有链的样本
        all_samples = np.concatenate([c['samples'] for c in self.chain_history], axis=0)
        
        alpha = 1 - credible_interval
        lower_q = alpha / 2 * 100
        upper_q = (1 - alpha / 2) * 100
        
        results = []
        
        for i, name in enumerate(self.param_names_estimated):
            samples = all_samples[:, i]
            
            mean_val = np.mean(samples)
            median_val = np.median(samples)
            std_val = np.std(samples)
            var_val = np.var(samples)
            ci_lower = np.percentile(samples, lower_q)
            ci_upper = np.percentile(samples, upper_q)
            
            results.append({
                'Parameter': name,
                'Mean': mean_val,
                'Median': median_val,
                'Std': std_val,
                'Variance': var_val,
                f'CI_{credible_interval*100:.0f}%_Lower': ci_lower,
                f'CI_{credible_interval*100:.0f}%_Upper': ci_upper
            })
        
        # 创建 DataFrame
        df = pd.DataFrame(results)
        
        # 打印结果
        print("\n" + "="*80)
        print(f"PARAMETER ESTIMATES (n_samples = {len(all_samples)})")
        print("="*80)
        
        for r in results:
            print(f"\n{r['Parameter']}:")
            print(f"  Mean:     {r['Mean']:.6f}")
            print(f"  Median:   {r['Median']:.6f}")
            print(f"  Std:      {r['Std']:.6f}")
            print(f"  Variance: {r['Variance']:.6f}")
            print(f"  {credible_interval*100:.0f}% CI:  [{r[f'CI_{credible_interval*100:.0f}%_Lower']:.6f}, {r[f'CI_{credible_interval*100:.0f}%_Upper']:.6f}]")
        
        print("\n" + "="*80 + "\n")
        
        return df
    
    def compute_ess(self):
        """
        计算有效样本量 (Effective Sample Size)
        ESS 衡量采样效率，值越接近总样本数越好
        使用自相关时间估计方法
        """
        all_samples = np.concatenate([c['samples'] for c in self.chain_history], axis=0)
        n_samples = len(all_samples)
        
        ess_results = {}
        
        print("\n=== Effective Sample Size (ESS) ===")
        
        for i, name in enumerate(self.param_names_estimated):
            samples = all_samples[:, i]
            
            # 计算自相关函数
            n = len(samples)
            mean = np.mean(samples)
            var = np.var(samples)
            
            if var < 1e-10:
                ess_results[name] = n
                continue
            
            # 自相关估计 (使用 FFT 加速)
            x = samples - mean
            fft_x = np.fft.fft(x, n=2*n)
            acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real / (var * n)
            
            # 计算自相关时间 (Sokal 方法)
            # 找到第一个负自相关或截断点
            tau = 1.0
            for k in range(1, n):
                if acf[k] < 0:
                    break
                tau += 2 * acf[k]
            
            ess = n / tau
            ess_results[name] = ess
            
            ratio = ess / n_samples * 100
            status = "✓" if ratio > 10 else "⚠"  # ESS > 10% 通常可接受
            print(f"  {name}: ESS = {ess:.1f} ({ratio:.1f}% of samples) {status}")
        
        print("===================================\n")
        
        return ess_results

    def compute_rhat(self):
        """
        计算 Gelman-Rubin R-hat 诊断统计量
        R-hat < 1.1 通常表示链已收敛
        返回: 字典 {参数名: R-hat值}
        """
        if len(self.chain_history) < 2:
            print("Warning: R-hat requires at least 2 chains. Skipping.")
            return {}
        
        n_chains = len(self.chain_history)
        n_samples = len(self.chain_history[0]['samples'])
        n_params = self.n_dim
        
        rhat_results = {}
        
        for p in range(n_params):
            param_name = self.param_names_estimated[p]
            
            # 收集各链该参数的样本
            chain_samples = [self.chain_history[c]['samples'][:, p] for c in range(n_chains)]
            
            # 各链均值
            chain_means = np.array([np.mean(cs) for cs in chain_samples])
            
            # 各链方差
            chain_vars = np.array([np.var(cs, ddof=1) for cs in chain_samples])
            
            # 总体均值
            grand_mean = np.mean(chain_means)
            
            # 链间方差 B (between-chain variance)
            B = n_samples * np.var(chain_means, ddof=1)
            
            # 链内方差 W (within-chain variance)
            W = np.mean(chain_vars)
            
            # 后验方差估计
            var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B
            
            # R-hat
            if W > 0:
                rhat = np.sqrt(var_hat / W)
            else:
                rhat = np.nan
                
            rhat_results[param_name] = rhat
        
        # 打印结果
        print("\n=== R-hat Diagnostics ===")
        all_converged = True
        for name, rhat in rhat_results.items():
            status = "✓" if rhat < 1.1 else "✗"
            if rhat >= 1.1:
                all_converged = False
            print(f"  {name}: R-hat = {rhat:.4f} {status}")
        
        if all_converged:
            print("All parameters converged (R-hat < 1.1)")
        else:
            print("WARNING: Some parameters may not have converged (R-hat >= 1.1)")
        print("=========================\n")
        
        return rhat_results
    
    # ==========================
    # 绘图与预测功能已移至独立模块
    # ==========================
    # 绘图: from MCMC_plotting import MCMCPlotting
    # 预测: from MCMC_prediction import MCMCPrediction
    # 
    # 使用示例:
    #   plotter = MCMCPlotting(fitter)
    #   plotter.plot_chains()
    #   plotter.plot_posterior()
    #   plotter.plot_fit()
    #   plotter.plot_corner()
    #
    #   predictor = MCMCPrediction(fitter)
    #   pred = predictor.predict(t_new)
    #   R0 = predictor.compute_R0()
