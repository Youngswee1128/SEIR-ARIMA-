import io
import contextlib
from pathlib import Path
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numba
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

from MCMC_method import BayesianODEFitter
from Run_fast import solve_ode_numba


N = 1.3e9
SIGMA = 1.0 / 2.5
GAMMA = 1.0 / 3.5
START_DATE = "2009-09-01"
TEST_HORIZONS = (7, 14, 21, 28)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_mcmc_phase2"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
OBSERVED_DATA_PATH = BASE_DIR / "observed_cases_smoothed.csv"
PLOT_DPI = 300
PLOT_FONT_SIZE = 15.0
PLOT_LABEL_SIZE = 16.0
PLOT_TITLE_SIZE = 17.0
PLOT_TICK_LABEL_SIZE = 14.0
PLOT_LEGEND_SIZE = 13.0
PLOT_SMALL_LEGEND_SIZE = 12.0
PLOT_INSET_TICK_LABEL_SIZE = 12.5

MCMC_ITER = 20000
MCMC_BURN_IN = 6000
MCMC_CHAINS = 3
MCMC_USE_PRIOR_INIT = False
PREDICTIVE_INTERVAL_MAX_SAMPLES = 500
RANDOM_SEED = 84
JUDGE1_EPS = 1e-12
JUDGE1_METRIC_NAMES = (
	"MAE",
	"MSE",
	"MSLE",
	"Normalized MAE",
	"Normalized MSE",
	"Maximum deviation",
)
JUDGE2_METRIC_NAMES = (
	"MAE",
	"RMSE",
	"WAPE(%)",
)


def apply_plot_style():
	plt.rcParams.update(
		{
			"font.family": "serif",
			"font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
			"font.size": PLOT_FONT_SIZE,
			"axes.labelsize": PLOT_LABEL_SIZE,
			"axes.titlesize": PLOT_TITLE_SIZE,
			"xtick.labelsize": PLOT_TICK_LABEL_SIZE,
			"ytick.labelsize": PLOT_TICK_LABEL_SIZE,
			"legend.fontsize": PLOT_LEGEND_SIZE,
			"figure.titlesize": PLOT_TITLE_SIZE,
			"svg.fonttype": "none",
			"axes.unicode_minus": False,
		}
	)


apply_plot_style()


def style_plot_legend(ax, loc="upper left", fontsize=None, ncol=None, bbox_to_anchor=None):
	kwargs = {
		"loc": loc,
		"frameon": True,
		"fancybox": False,
		"facecolor": "white",
		"edgecolor": "#4d4d4d",
		"framealpha": 1.0,
		"handlelength": 2.4,
		"borderpad": 0.35,
		"labelspacing": 0.35,
	}
	if fontsize is not None:
		kwargs["fontsize"] = fontsize
	if ncol is not None:
		kwargs["ncol"] = ncol
	if bbox_to_anchor is not None:
		kwargs["bbox_to_anchor"] = bbox_to_anchor
	ax.legend(**kwargs)


@numba.jit(nopython=True)
def seirc_ode_decay(y, t, params):
	beta0, decay_rate, sigma, gamma, n_pop = params
	s, e, i, r, c = y
	beta_t = beta0 * np.exp(-decay_rate * t)
	infection = beta_t * s * i / n_pop
	ds = -infection
	de = infection - sigma * e
	di = sigma * e - gamma * i
	dr = gamma * i
	dc = sigma * e
	return np.array((ds, de, di, dr, dc), dtype=np.float64)


def simulate_seirc_decay(beta0, decay_rate, e0, i0, r0, c0, num_days):
	y0 = np.array([N - e0 - i0 - r0, e0, i0, r0, c0], dtype=np.float64)
	params = np.array([beta0, decay_rate, SIGMA, GAMMA, N], dtype=np.float64)
	t_eval = np.arange(num_days + 1, dtype=np.float64)
	return solve_ode_numba(seirc_ode_decay, y0, t_eval, params)


def to_population_percent(values):
	return 100.0 * np.asarray(values, dtype=np.float64) / N


def fit_best_arima(resid_train):
	best_model = None
	best_order = None
	best_aic = np.inf
	for d in (0, 1):
		for p in range(0, 4):
			for q in range(0, 4):
				if p == 0 and d == 0 and q == 0:
					continue
				try:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						trend = "n" if d > 0 else "c"
						model = ARIMA(resid_train, order=(p, d, q), trend=trend).fit()
				except Exception:
					continue
				aic = float(model.aic)
				if np.isfinite(aic) and aic < best_aic:
					best_aic = aic
					best_model = model
					best_order = (p, d, q)
	if best_model is None:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			best_model = ARIMA(resid_train, order=(1, 0, 0), trend="c").fit()
		best_order = (1, 0, 0)
	return best_model, best_order


def compute_judge1_metrics(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
	y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
	if y_true.shape != y_pred.shape:
		raise ValueError("y_true and y_pred must have the same shape")
	if y_true.size == 0:
		raise ValueError("y_true and y_pred cannot be empty")

	valid = np.isfinite(y_true) & np.isfinite(y_pred)
	if not np.any(valid):
		return {name: np.nan for name in JUDGE1_METRIC_NAMES}

	y = to_population_percent(y_true[valid])
	pred = to_population_percent(y_pred[valid])
	abs_err = np.abs(y - pred)
	sq_err = (y - pred) ** 2
	mae = float(np.mean(abs_err))
	mse = float(np.mean(sq_err))
	msle = float(np.mean((np.log1p(np.maximum(y, 0.0)) - np.log1p(np.maximum(pred, 0.0))) ** 2))
	mean_y = float(np.mean(y))
	mean_pred = float(np.mean(pred))
	norm_mse_denom = mean_y * mean_pred
	relative_mask = np.abs(y) > JUDGE1_EPS

	return {
		"MAE": mae,
		"MSE": mse,
		"MSLE": msle,
		"Normalized MAE": np.nan if abs(mean_y) <= JUDGE1_EPS else float(mae / mean_y),
		"Normalized MSE": np.nan if abs(norm_mse_denom) <= JUDGE1_EPS else float(mse / norm_mse_denom),
		"Maximum deviation": np.nan if not np.any(relative_mask) else float(np.max(abs_err[relative_mask] / np.abs(y[relative_mask]))),
	}


def compute_judge2_metrics(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
	y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
	if y_true.shape != y_pred.shape:
		raise ValueError("y_true and y_pred must have the same shape")
	if y_true.size == 0:
		raise ValueError("y_true and y_pred cannot be empty")

	valid = np.isfinite(y_true) & np.isfinite(y_pred)
	if not np.any(valid):
		return {name: np.nan for name in JUDGE2_METRIC_NAMES}

	y = y_true[valid]
	pred = y_pred[valid]
	err = y - pred
	abs_err = np.abs(err)
	mae = float(np.mean(abs_err))
	rmse = float(np.sqrt(np.mean(err**2)))
	observed_total = float(np.sum(np.abs(y)))
	wape = np.nan if observed_total <= 1e-12 else float(100.0 * np.sum(abs_err) / observed_total)

	return {
		"MAE": mae,
		"RMSE": rmse,
		"WAPE(%)": wape,
	}


def get_arima_fitted_values(arima_model, n_train):
	try:
		fitted = np.asarray(arima_model.fittedvalues, dtype=np.float64)
	except Exception:
		fitted = np.empty(0, dtype=np.float64)
	if fitted.size != n_train:
		fitted = np.asarray(arima_model.predict(start=0, end=n_train - 1), dtype=np.float64)
	return np.where(np.isfinite(fitted), fitted, 0.0)


def get_arima_forecast_with_interval(arima_model, horizon):
	if arima_model is None:
		zeros = np.zeros(int(horizon), dtype=np.float64)
		return zeros, zeros, zeros
	forecast = arima_model.get_forecast(steps=horizon)
	mean = np.asarray(forecast.predicted_mean, dtype=np.float64)
	ci = np.asarray(forecast.conf_int(alpha=0.05), dtype=np.float64)
	return mean, ci[:, 0], ci[:, 1]


def cumulative_segment_to_daily(segment_cumulative, previous_cumulative):
	prev = max(float(previous_cumulative), 0.0)
	seg = np.asarray(segment_cumulative, dtype=np.float64)
	return np.maximum(np.diff(np.concatenate(([prev], seg))), 0.0)


def enforce_monotonic_cumulative_segment(segment_cumulative, previous_cumulative):
	prev = max(float(previous_cumulative), 0.0)
	seg = np.maximum(np.asarray(segment_cumulative, dtype=np.float64), 0.0)
	return np.maximum.accumulate(np.concatenate(([prev], seg)))[1:]


def cumulative_samples_to_daily_samples(cumulative_samples, previous_cumulative):
	cumulative_samples = np.asarray(cumulative_samples, dtype=np.float64)
	previous = np.full((cumulative_samples.shape[0], 1), max(float(previous_cumulative), 0.0))
	return np.maximum(np.diff(np.concatenate([previous, cumulative_samples], axis=1), axis=1), 0.0)


def predictive_quantiles_from_samples(samples):
	lower, upper = np.quantile(np.asarray(samples, dtype=np.float64), [0.025, 0.975], axis=0)
	return np.maximum(lower, 0.0), np.maximum(upper, 0.0)


def compute_instantaneous_reproduction_number(beta0, decay_rate, n_days):
	t = np.arange(int(n_days), dtype=np.float64)
	return (float(beta0) * np.exp(-float(decay_rate) * t)) / GAMMA


def build_instantaneous_reproduction_interval(posterior_samples, posterior_param_names, n_days, point_curve):
	samples = select_posterior_samples(posterior_samples)
	if len(samples) == 0:
		return point_curve, point_curve

	name_to_idx = {name: idx for idx, name in enumerate(posterior_param_names)}
	required = ("beta0", "decay_rate")
	if any(name not in name_to_idx for name in required):
		return point_curve, point_curve

	t = np.arange(int(n_days), dtype=np.float64)
	beta0_samples = samples[:, name_to_idx["beta0"]]
	decay_samples = samples[:, name_to_idx["decay_rate"]]
	rt_samples = (beta0_samples[:, None] * np.exp(-decay_samples[:, None] * t[None, :])) / GAMMA
	rt_low, rt_high = predictive_quantiles_from_samples(rt_samples)
	return include_point_in_interval(point_curve, rt_low, rt_high)


def include_point_in_interval(point, lower, upper):
	point = np.asarray(point, dtype=np.float64)
	lower = np.asarray(lower, dtype=np.float64)
	upper = np.asarray(upper, dtype=np.float64)
	return np.minimum(lower, point), np.maximum(upper, point)


def select_posterior_samples(samples, max_samples=PREDICTIVE_INTERVAL_MAX_SAMPLES):
	samples = np.asarray(samples, dtype=np.float64)
	if samples.ndim != 2 or samples.size == 0:
		return np.empty((0, 0), dtype=np.float64)
	samples = samples[np.all(np.isfinite(samples), axis=1)]
	if len(samples) <= max_samples:
		return samples
	sample_idx = np.linspace(0, len(samples) - 1, max_samples, dtype=int)
	return samples[sample_idx]


def arima_forecast_draws_from_interval(arima_model, horizon, n_samples, seed):
	mean, lower, upper = get_arima_forecast_with_interval(arima_model, horizon)
	std = np.maximum((upper - lower) / (2.0 * 1.96), 1e-9)
	rng = np.random.default_rng(seed)
	return rng.normal(mean[None, :], std[None, :], size=(n_samples, horizon))


def collect_posterior_samples(fitter):
	if not fitter.chain_history:
		return np.empty((0, 0), dtype=np.float64)
	return np.concatenate([chain["samples"] for chain in fitter.chain_history], axis=0)


def build_joint_hybrid_interval_for_forecast_segment(posterior_samples, posterior_param_names, n_total_days, train_len, cumulative_anchor, arima_model, prev_cumulative, point_daily, point_cumulative, seed):
	samples = select_posterior_samples(posterior_samples)
	horizon = n_total_days - train_len
	if len(samples) == 0:
		return point_daily, point_daily, point_cumulative, point_cumulative

	name_to_idx = {name: idx for idx, name in enumerate(posterior_param_names)}
	required = ("beta0", "decay_rate", "IC_E", "IC_I", "IC_R")
	if any(name not in name_to_idx for name in required):
		return point_daily, point_daily, point_cumulative, point_cumulative

	residual_draws = arima_forecast_draws_from_interval(arima_model, horizon, len(samples), seed)
	cumulative_draws = []
	for sample, residual_path in zip(samples, residual_draws):
		try:
			sim = simulate_seirc_decay(
				sample[name_to_idx["beta0"]],
				sample[name_to_idx["decay_rate"]],
				sample[name_to_idx["IC_E"]],
				sample[name_to_idx["IC_I"]],
				sample[name_to_idx["IC_R"]],
				cumulative_anchor,
				n_total_days,
			)
			base_cum = sim[1:, 4]
			segment = enforce_monotonic_cumulative_segment(base_cum[train_len:] + residual_path, prev_cumulative)
		except Exception:
			continue
		if len(segment) == horizon and np.all(np.isfinite(segment)):
			cumulative_draws.append(segment)

	if not cumulative_draws:
		return point_daily, point_daily, point_cumulative, point_cumulative

	cumulative_samples = np.vstack(cumulative_draws)
	daily_samples = cumulative_samples_to_daily_samples(cumulative_samples, prev_cumulative)
	daily_low, daily_high = predictive_quantiles_from_samples(daily_samples)
	cum_low, cum_high = predictive_quantiles_from_samples(cumulative_samples)
	daily_low, daily_high = include_point_in_interval(point_daily, daily_low, daily_high)
	cum_low, cum_high = include_point_in_interval(point_cumulative, cum_low, cum_high)
	return daily_low, daily_high, cum_low, cum_high


def build_arima_only_hybrid_interval(base_cumulative_test, arima_model, horizon, prev_cumulative, point_daily, point_cumulative):
	_, resid_lower, resid_upper = get_arima_forecast_with_interval(arima_model, horizon)
	base_cumulative_test = np.asarray(base_cumulative_test, dtype=np.float64)

	cum_lower_raw = enforce_monotonic_cumulative_segment(base_cumulative_test + resid_lower, prev_cumulative)
	cum_upper_raw = enforce_monotonic_cumulative_segment(base_cumulative_test + resid_upper, prev_cumulative)
	cum_low = np.minimum(cum_lower_raw, cum_upper_raw)
	cum_high = np.maximum(cum_lower_raw, cum_upper_raw)

	daily_lower_raw = cumulative_segment_to_daily(cum_low, prev_cumulative)
	daily_upper_raw = cumulative_segment_to_daily(cum_high, prev_cumulative)
	daily_low = np.minimum(daily_lower_raw, daily_upper_raw)
	daily_high = np.maximum(daily_lower_raw, daily_upper_raw)

	daily_low, daily_high = include_point_in_interval(point_daily, daily_low, daily_high)
	cum_low, cum_high = include_point_in_interval(point_cumulative, cum_low, cum_high)
	return daily_low, daily_high, cum_low, cum_high


def build_joint_daily_residual_hybrid_interval_for_forecast_segment(posterior_samples, posterior_param_names, n_total_days, train_len, cumulative_anchor, arima_model, prev_cumulative, point_daily, point_cumulative, seed):
	samples = select_posterior_samples(posterior_samples)
	horizon = n_total_days - train_len
	if len(samples) == 0:
		return point_daily, point_daily, point_cumulative, point_cumulative

	name_to_idx = {name: idx for idx, name in enumerate(posterior_param_names)}
	required = ("beta0", "decay_rate", "IC_E", "IC_I", "IC_R")
	if any(name not in name_to_idx for name in required):
		return point_daily, point_daily, point_cumulative, point_cumulative

	residual_draws = arima_forecast_draws_from_interval(arima_model, horizon, len(samples), seed)
	daily_draws = []
	for sample, residual_path in zip(samples, residual_draws):
		try:
			sim = simulate_seirc_decay(
				sample[name_to_idx["beta0"]],
				sample[name_to_idx["decay_rate"]],
				sample[name_to_idx["IC_E"]],
				sample[name_to_idx["IC_I"]],
				sample[name_to_idx["IC_R"]],
				cumulative_anchor,
				n_total_days,
			)
			base_daily = np.diff(sim[:, 4])[train_len:]
			segment = np.maximum(base_daily + residual_path, 0.0)
		except Exception:
			continue
		if len(segment) == horizon and np.all(np.isfinite(segment)):
			daily_draws.append(segment)

	if not daily_draws:
		return point_daily, point_daily, point_cumulative, point_cumulative

	daily_samples = np.vstack(daily_draws)
	cumulative_samples = max(float(prev_cumulative), 0.0) + np.cumsum(daily_samples, axis=1)
	daily_low, daily_high = predictive_quantiles_from_samples(daily_samples)
	cum_low, cum_high = predictive_quantiles_from_samples(cumulative_samples)
	daily_low, daily_high = include_point_in_interval(point_daily, daily_low, daily_high)
	cum_low, cum_high = include_point_in_interval(point_cumulative, cum_low, cum_high)
	return daily_low, daily_high, cum_low, cum_high


def build_arima_only_daily_residual_hybrid_interval(base_daily_test, arima_model, horizon, prev_cumulative, point_daily, point_cumulative):
	_, resid_lower, resid_upper = get_arima_forecast_with_interval(arima_model, horizon)
	base_daily_test = np.asarray(base_daily_test, dtype=np.float64)

	daily_lower_raw = np.maximum(base_daily_test + resid_lower, 0.0)
	daily_upper_raw = np.maximum(base_daily_test + resid_upper, 0.0)
	daily_low = np.minimum(daily_lower_raw, daily_upper_raw)
	daily_high = np.maximum(daily_lower_raw, daily_upper_raw)
	daily_low, daily_high = include_point_in_interval(point_daily, daily_low, daily_high)

	cum_low = max(float(prev_cumulative), 0.0) + np.cumsum(daily_low)
	cum_high = max(float(prev_cumulative), 0.0) + np.cumsum(daily_high)
	cum_low, cum_high = include_point_in_interval(point_cumulative, cum_low, cum_high)
	return daily_low, daily_high, cum_low, cum_high


def estimate_ic_priors(train_daily):
	total = max(float(np.sum(train_daily[:7])), 50.0)
	return {
		"E": {"value": total * 0.25, "mu": total * 0.25, "sigma": max(total * 0.15, 20.0), "lower": 1e-3, "upper": max(total * 10.0, 300.0)},
		"I": {"value": total * 0.45, "mu": total * 0.45, "sigma": max(total * 0.22, 30.0), "lower": 1e-3, "upper": max(total * 12.0, 500.0)},
		"R": {"value": total * 0.15, "mu": total * 0.15, "sigma": max(total * 0.10, 10.0), "lower": 0.0, "upper": max(total * 6.0, 200.0)},
	}


def plot_chain_diagnostics(fitter, horizon):
	apply_plot_style()
	prefix = FIGURES_DIR / f"mcmc_h{horizon}"
	n_params = len(fitter.param_names_estimated)
	if n_params == 0 or not fitter.chain_history:
		return
	fig, axes = plt.subplots(n_params, 1, figsize=(13, 2.5 * n_params), sharex=True)
	if n_params == 1:
		axes = [axes]
	for p_idx, p_name in enumerate(fitter.param_names_estimated):
		ax = axes[p_idx]
		for c_idx, chain in enumerate(fitter.chain_history):
			ax.plot(chain["samples"][:, p_idx], lw=0.8, alpha=0.75, label=f"Chain {c_idx + 1}")
		ax.set_ylabel(p_name)
		ax.grid(True, linestyle=":", alpha=0.4)
		if p_idx == 0:
			style_plot_legend(ax, loc="upper right", fontsize=PLOT_SMALL_LEGEND_SIZE, ncol=min(len(fitter.chain_history), 3))
	axes[-1].set_xlabel("Iteration (post burn-in)")
	fig.tight_layout()
	fig.savefig(prefix.with_name(prefix.name + "_trace.png"), dpi=PLOT_DPI, bbox_inches="tight")
	plt.close(fig)

	fig, axes = plt.subplots(n_params, 1, figsize=(13, 2.5 * n_params))
	if n_params == 1:
		axes = [axes]
	for p_idx, p_name in enumerate(fitter.param_names_estimated):
		ax = axes[p_idx]
		samples_all = np.concatenate([chain["samples"][:, p_idx] for chain in fitter.chain_history])
		ax.hist(samples_all, bins=30, color="#4c78a8", alpha=0.8, edgecolor="white")
		ax.set_ylabel(p_name)
		ax.grid(True, linestyle=":", alpha=0.4)
	axes[-1].set_xlabel("Value")
	fig.tight_layout()
	fig.savefig(prefix.with_name(prefix.name + "_posterior.png"), dpi=PLOT_DPI, bbox_inches="tight")
	plt.close(fig)


def summarize_samples(parameter, samples, credible_interval=0.95):
	samples = np.asarray(samples, dtype=np.float64)
	alpha = 1.0 - credible_interval
	return {
		"Parameter": parameter,
		"Mean": float(np.mean(samples)),
		"Median": float(np.median(samples)),
		"Std": float(np.std(samples)),
		"Variance": float(np.var(samples)),
		f"CI_{credible_interval * 100:.0f}%_Lower": float(np.percentile(samples, 100.0 * alpha / 2.0)),
		f"CI_{credible_interval * 100:.0f}%_Upper": float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0))),
	}


def get_parameter_samples(fitter, parameter):
	if parameter not in fitter.param_names_estimated:
		raise ValueError(f"Parameter {parameter!r} is not estimated.")
	param_idx = fitter.param_names_estimated.index(parameter)
	return np.concatenate([chain["samples"][:, param_idx] for chain in fitter.chain_history])


def plot_initial_reproduction_posterior(r0_samples, horizon):
	apply_plot_style()
	mean_val = float(np.mean(r0_samples))
	ci_low, ci_high = np.percentile(r0_samples, [2.5, 97.5])
	fig, ax = plt.subplots(figsize=(7.2, 4.4))
	ax.hist(r0_samples, bins=36, density=True, color="#4c78a8", alpha=0.82, edgecolor="white")
	ax.axvline(mean_val, color="#d62728", linewidth=1.8, label=f"Mean = {mean_val:.3f}")
	ax.axvline(ci_low, color="#333333", linestyle="--", linewidth=1.2, label=f"95% CI = [{ci_low:.3f}, {ci_high:.3f}]")
	ax.axvline(ci_high, color="#333333", linestyle="--", linewidth=1.2)
	ax.set_title(f"Posterior distribution of initial reproduction number (H={horizon})")
	ax.set_xlabel(r"$\mathcal{R}_0 = \beta_0 / \gamma$")
	ax.set_ylabel("Density")
	ax.grid(True, linestyle=":", alpha=0.35)
	style_plot_legend(ax, loc="upper right", fontsize=PLOT_SMALL_LEGEND_SIZE)
	fig.tight_layout()
	fig.savefig(FIGURES_DIR / f"mcmc_h{horizon}_initial_reproduction_posterior.png", dpi=PLOT_DPI, bbox_inches="tight")
	plt.close(fig)


def save_mcmc_diagnostics(fitter, horizon):
	with contextlib.redirect_stdout(io.StringIO()):
		summary_df = fitter.summary_statistics()
	with contextlib.redirect_stdout(io.StringIO()):
		ess_dict = fitter.compute_ess()
	with contextlib.redirect_stdout(io.StringIO()):
		rhat_dict = fitter.compute_rhat()
	diag_df = pd.DataFrame([{"Parameter": name, "ESS": ess_dict.get(name, np.nan), "R_hat": rhat_dict.get(name, np.nan)} for name in fitter.param_names_estimated])
	beta0_samples = get_parameter_samples(fitter, "beta0")
	initial_r0_samples = beta0_samples / GAMMA
	summary_df = pd.concat(
		[summary_df, pd.DataFrame([summarize_samples("initial_reproduction_number", initial_r0_samples)])],
		ignore_index=True,
	)
	summary_df.to_csv(DATA_DIR / f"mcmc_summary_h{horizon}.csv", index=False)
	diag_df.to_csv(DATA_DIR / f"mcmc_chain_diagnostics_h{horizon}.csv", index=False)
	plot_chain_diagnostics(fitter, horizon)
	plot_initial_reproduction_posterior(initial_r0_samples, horizon)
	return summary_df


def fit_seirc_decay_mcmc(y_train_daily, cumulative_anchor, horizon):
	np.random.seed(RANDOM_SEED + horizon)
	priors = estimate_ic_priors(y_train_daily)
	time_points = np.arange(len(y_train_daily) + 1, dtype=np.float64)
	obs_daily = np.insert(y_train_daily.astype(np.float64), 0, 0.0)
	fitter = BayesianODEFitter(seirc_ode_decay, time_points, {})
	fitter.set_compartments(["S", "E", "I", "R", "C"])
	fitter.set_fit_targets({"C": obs_daily}, target_types={"C": "diff"})
	fitter.add_parameter("beta0", value=0.45, type="estimated", prior={"dist": "uniform", "lower": 0.2, "upper": 0.9})
	fitter.add_parameter("decay_rate", value=0.006, type="estimated", prior={"dist": "uniform", "lower": 0.0, "upper": 0.03})
	fitter.add_parameter("dispersion", value=6.0, type="estimated", prior={"dist": "gamma", "alpha": 10.0, "beta": 1.5})
	fitter.add_parameter("sigma", value=SIGMA, type="fixed")
	fitter.add_parameter("gamma", value=GAMMA, type="fixed")
	fitter.add_parameter("n_pop", value=N, type="fixed")
	fitter.set_initial_conditions(
		{
			"S": N,
			"E": {"type": "estimated", "value": priors["E"]["value"], "prior": {"dist": "truncated_normal", "mu": priors["E"]["mu"], "sigma": priors["E"]["sigma"], "lower": priors["E"]["lower"], "upper": priors["E"]["upper"]}},
			"I": {"type": "estimated", "value": priors["I"]["value"], "prior": {"dist": "truncated_normal", "mu": priors["I"]["mu"], "sigma": priors["I"]["sigma"], "lower": priors["I"]["lower"], "upper": priors["I"]["upper"]}},
			"R": {"type": "estimated", "value": priors["R"]["value"], "prior": {"dist": "truncated_normal", "mu": priors["R"]["mu"], "sigma": priors["R"]["sigma"], "lower": priors["R"]["lower"], "upper": priors["R"]["upper"]}},
			"C": float(cumulative_anchor),
		}
	)
	fitter.set_likelihood("NegativeBinomial")
	fitter.run_mcmc(n_iter=MCMC_ITER, n_chains=MCMC_CHAINS, burn_in=MCMC_BURN_IN, adapt_step=True, use_prior_init=MCMC_USE_PRIOR_INIT)
	summary_df = save_mcmc_diagnostics(fitter, horizon)
	param_means = summary_df.set_index("Parameter")["Mean"].to_dict()
	posterior_samples = collect_posterior_samples(fitter)
	return {
		"beta0": float(param_means["beta0"]),
		"initial_reproduction_number": float(param_means["initial_reproduction_number"]),
		"decay_rate": float(param_means["decay_rate"]),
		"E0": float(param_means["IC_E"]),
		"I0": float(param_means["IC_I"]),
		"R0": float(param_means["IC_R"]),
		"C0": float(cumulative_anchor),
		"dispersion": float(param_means["dispersion"]),
		"posterior_param_names": list(fitter.param_names_estimated),
		"posterior_samples": posterior_samples,
	}


def summarize_residuals(residuals, run_ljung_box=False, run_normality_tests=False):
	residuals = np.asarray(residuals, dtype=np.float64)
	residuals = residuals[np.isfinite(residuals)]
	n = residuals.size
	if n == 0:
		return {"sample_size": 0, "residual_mean": np.nan, "residual_std": np.nan, "residual_mae": np.nan, "residual_rmse": np.nan, "ljung_box_pvalue": np.nan, "jarque_bera_pvalue": np.nan, "shapiro_pvalue": np.nan}
	ljung = np.nan
	if run_ljung_box and n >= 20:
		try:
			ljung_lag = min(10, max(1, n // 5))
			ljung = float(acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)["lb_pvalue"].iloc[0])
		except Exception:
			pass
	jb = np.nan
	if run_normality_tests and n > 7:
		try:
			jb = float(stats.jarque_bera(residuals).pvalue)
		except Exception:
			pass
	shapiro = np.nan
	if run_normality_tests and 3 <= n <= 5000:
		try:
			shapiro = float(stats.shapiro(residuals).pvalue)
		except Exception:
			pass
	return {
		"sample_size": int(n),
		"residual_mean": float(np.mean(residuals)),
		"residual_std": float(np.std(residuals, ddof=1)) if n > 1 else 0.0,
		"residual_mae": float(np.mean(np.abs(residuals))),
		"residual_rmse": float(np.sqrt(np.mean(residuals**2))),
		"ljung_box_pvalue": ljung,
		"jarque_bera_pvalue": jb,
		"shapiro_pvalue": shapiro,
	}


def evaluate_single_horizon(data_window, horizon):
	n_all = len(data_window)
	n_train = n_all - horizon
	y_all_cum = data_window["cumulative_cases"].to_numpy(dtype=np.float64)
	y_all_daily = data_window["daily_cases"].to_numpy(dtype=np.float64)
	dates_all = data_window["date"].to_numpy()
	dates_test = dates_all[n_train:]
	cumulative_anchor = float(data_window["cumulative_anchor"].iloc[0])
	y_train_daily = y_all_daily[:n_train]
	y_test_daily = y_all_daily[n_train:]
	y_test_cum = y_all_cum[n_train:]

	fit = fit_seirc_decay_mcmc(y_train_daily, cumulative_anchor, horizon)
	sim = simulate_seirc_decay(fit["beta0"], fit["decay_rate"], fit["E0"], fit["I0"], fit["R0"], fit["C0"], n_all)
	c_seirc_all = sim[1:, 4]
	daily_seirc_all = np.diff(sim[:, 4])
	rt_all = compute_instantaneous_reproduction_number(fit["beta0"], fit["decay_rate"], n_all)
	rt_low_all, rt_high_all = build_instantaneous_reproduction_interval(
		fit["posterior_samples"],
		fit["posterior_param_names"],
		n_all,
		rt_all,
	)

	cumulative_resid_train = y_all_cum[:n_train] - c_seirc_all[:n_train]
	arima_model, arima_order = fit_best_arima(cumulative_resid_train)
	resid_fc, _, _ = get_arima_forecast_with_interval(arima_model, horizon)

	arima_fit_train = get_arima_fitted_values(arima_model, n_train)

	hybrid_cum_train = enforce_monotonic_cumulative_segment(c_seirc_all[:n_train] + arima_fit_train, cumulative_anchor)
	if hybrid_cum_train.size > 0:
		hybrid_cum_train[0] = max(float(y_all_cum[0]), 0.0)
		hybrid_cum_train = np.maximum.accumulate(hybrid_cum_train)
	last_train_cumulative = float(hybrid_cum_train[-1])
	hybrid_cum_test = enforce_monotonic_cumulative_segment(c_seirc_all[n_train:] + resid_fc, last_train_cumulative)
	hybrid_daily_train = cumulative_segment_to_daily(hybrid_cum_train, cumulative_anchor)
	hybrid_daily_test = cumulative_segment_to_daily(hybrid_cum_test, last_train_cumulative)
	arima_only_daily_low_test, arima_only_daily_high_test, arima_only_low_test, arima_only_high_test = build_arima_only_hybrid_interval(
		c_seirc_all[n_train:],
		arima_model,
		horizon,
		last_train_cumulative,
		hybrid_daily_test,
		hybrid_cum_test,
	)
	hybrid_daily_low_test, hybrid_daily_high_test, hybrid_low_test, hybrid_high_test = build_joint_hybrid_interval_for_forecast_segment(
		fit["posterior_samples"],
		fit["posterior_param_names"],
		n_all,
		n_train,
		cumulative_anchor,
		arima_model,
		last_train_cumulative,
		hybrid_daily_test,
		hybrid_cum_test,
		RANDOM_SEED + 1000 + horizon,
	)
	hybrid_daily_all = np.concatenate([hybrid_daily_train, hybrid_daily_test])
	hybrid_all = np.concatenate([hybrid_cum_train, hybrid_cum_test])

	daily_resid_train = y_all_daily[:n_train] - daily_seirc_all[:n_train]
	try:
		arima_model_daily, arima_order_daily = fit_best_arima(daily_resid_train)
		resid_daily_fc, _, _ = get_arima_forecast_with_interval(arima_model_daily, horizon)
		arima_fit_daily_train = get_arima_fitted_values(arima_model_daily, n_train)
	except Exception:
		arima_model_daily, arima_order_daily = None, None
		resid_daily_fc = np.zeros(horizon, dtype=np.float64)
		arima_fit_daily_train = np.zeros(n_train, dtype=np.float64)

	hybrid_daily_train_from_daily_arima = np.maximum(daily_seirc_all[:n_train] + arima_fit_daily_train, 0.0)
	hybrid_daily_test_from_daily_arima = np.maximum(daily_seirc_all[n_train:] + resid_daily_fc, 0.0)
	if hybrid_daily_train_from_daily_arima.size > 0:
		hybrid_cum_train_from_daily_arima = max(float(cumulative_anchor), 0.0) + np.cumsum(hybrid_daily_train_from_daily_arima)
		hybrid_cum_train_from_daily_arima = np.maximum.accumulate(hybrid_cum_train_from_daily_arima)
		last_train_cumulative_daily = float(hybrid_cum_train_from_daily_arima[-1])
	else:
		hybrid_cum_train_from_daily_arima = np.array([], dtype=np.float64)
		last_train_cumulative_daily = last_train_cumulative
	hybrid_cum_test_from_daily_arima = last_train_cumulative_daily + np.cumsum(hybrid_daily_test_from_daily_arima)

	arima_daily_only_daily_low_test, arima_daily_only_daily_high_test, arima_daily_only_low_test, arima_daily_only_high_test = build_arima_only_daily_residual_hybrid_interval(
		daily_seirc_all[n_train:],
		arima_model_daily,
		horizon,
		last_train_cumulative_daily,
		hybrid_daily_test_from_daily_arima,
		hybrid_cum_test_from_daily_arima,
	)
	hybrid_daily_low_test_from_daily_arima, hybrid_daily_high_test_from_daily_arima, hybrid_low_test_from_daily_arima, hybrid_high_test_from_daily_arima = build_joint_daily_residual_hybrid_interval_for_forecast_segment(
		fit["posterior_samples"],
		fit["posterior_param_names"],
		n_all,
		n_train,
		cumulative_anchor,
		arima_model_daily,
		last_train_cumulative_daily,
		hybrid_daily_test_from_daily_arima,
		hybrid_cum_test_from_daily_arima,
		RANDOM_SEED + 2000 + horizon,
	)

	return {
		"horizon": horizon,
		"n_train": n_train,
		"dates_all": dates_all,
		"dates_test": dates_test,
		"y_all": y_all_cum,
		"y_all_daily": y_all_daily,
		"y_test_daily": y_test_daily,
		"y_test_cum": y_test_cum,
		"daily_seirc_all": daily_seirc_all,
		"instantaneous_reproduction_number_all": rt_all,
		"instantaneous_reproduction_number_low_all": rt_low_all,
		"instantaneous_reproduction_number_high_all": rt_high_all,
		"hybrid_daily_all": hybrid_daily_all,
		"arima_only_daily_low_test": arima_only_daily_low_test,
		"arima_only_daily_high_test": arima_only_daily_high_test,
		"hybrid_daily_low_test": hybrid_daily_low_test,
		"hybrid_daily_high_test": hybrid_daily_high_test,
		"seirc_all": c_seirc_all,
		"hybrid_all": hybrid_all,
		"arima_only_low_test": arima_only_low_test,
		"arima_only_high_test": arima_only_high_test,
		"hybrid_low_test": hybrid_low_test,
		"hybrid_high_test": hybrid_high_test,
		"seirc_resid_train": cumulative_resid_train,
		"seirc_resid_test": y_test_cum - c_seirc_all[n_train:],
		"seirc_resid_all": y_all_cum - c_seirc_all,
		"hybrid_resid_train": y_all_cum[:n_train] - hybrid_cum_train,
		"hybrid_resid_test": y_test_cum - hybrid_cum_test,
		"hybrid_resid_all": y_all_cum - hybrid_all,
		"hybrid_daily_train_from_daily_arima": hybrid_daily_train_from_daily_arima,
		"hybrid_daily_test_from_daily_arima": hybrid_daily_test_from_daily_arima,
		"hybrid_cum_train_from_daily_arima": hybrid_cum_train_from_daily_arima,
		"hybrid_cum_test_from_daily_arima": hybrid_cum_test_from_daily_arima,
		"seirc_daily_resid_train": daily_resid_train,
		"seirc_daily_resid_test": y_test_daily - daily_seirc_all[n_train:],
		"arima_daily_order": arima_order_daily,
		"arima_daily_model": arima_model_daily,
		"arima_daily_only_daily_low_test": arima_daily_only_daily_low_test,
		"arima_daily_only_daily_high_test": arima_daily_only_daily_high_test,
		"arima_daily_only_low_test": arima_daily_only_low_test,
		"arima_daily_only_high_test": arima_daily_only_high_test,
		"hybrid_daily_low_test_from_daily_arima": hybrid_daily_low_test_from_daily_arima,
		"hybrid_daily_high_test_from_daily_arima": hybrid_daily_high_test_from_daily_arima,
		"hybrid_low_test_from_daily_arima": hybrid_low_test_from_daily_arima,
		"hybrid_high_test_from_daily_arima": hybrid_high_test_from_daily_arima,
		"params": {
			"beta0": fit["beta0"],
			"initial_reproduction_number": fit["initial_reproduction_number"],
			"decay_rate": fit["decay_rate"],
			"E0": fit["E0"],
			"I0": fit["I0"],
			"R0": fit["R0"],
			"C0": fit["C0"],
			"dispersion": fit["dispersion"],
			"ARIMA_order": arima_order,
			"ARIMA_daily_order": arima_order_daily,
			"mcmc_iter": MCMC_ITER,
			"mcmc_burn_in": MCMC_BURN_IN,
			"mcmc_chains": MCMC_CHAINS,
		},
	}


def build_parameter_table(results):
	rows = []
	for result in results:
		params = result["params"]
		rows.append(
			{
				"test_horizon_days": result["horizon"],
				"beta0": params["beta0"],
				"initial_reproduction_number": params["initial_reproduction_number"],
				"decay_rate": params["decay_rate"],
				"E0": params["E0"],
				"I0": params["I0"],
				"R0": params["R0"],
				"C0": params["C0"],
				"dispersion": params["dispersion"],
				"ARIMA_order": str(params["ARIMA_order"]),
				"ARIMA_daily_order": str(params.get("ARIMA_daily_order")),
				"mcmc_iter": params["mcmc_iter"],
				"mcmc_burn_in": params["mcmc_burn_in"],
				"mcmc_chains": params["mcmc_chains"],
			}
		)
	return pd.DataFrame(rows)


def error_reduction_pct(baseline_value, hybrid_value):
	baseline_value = float(baseline_value)
	hybrid_value = float(hybrid_value)
	if not np.isfinite(baseline_value) or abs(baseline_value) <= 1e-12:
		return np.nan
	return float(100.0 * (baseline_value - hybrid_value) / baseline_value)


def build_judge_table(results):
	rows = []
	target_specs = [
		("cases_daily", "y_test_daily", "daily_seirc_all", "hybrid_daily_all", True),
		("cases_cumulative", "y_test_cum", "seirc_all", "hybrid_all", True),
	]
	for result in results:
		horizon = result["horizon"]
		n_train = result["n_train"]
		for target, observed_key, baseline_key, hybrid_key, needs_slice in target_specs:
			observed = np.asarray(result[observed_key], dtype=np.float64)
			baseline = np.asarray(result[baseline_key][n_train:] if needs_slice else result[baseline_key], dtype=np.float64)
			hybrid = np.asarray(result[hybrid_key][n_train:] if needs_slice else result[hybrid_key], dtype=np.float64)
			if target == "cases_daily":
				hybrid_daily_from_daily_pred = np.asarray(result.get("hybrid_daily_test_from_daily_arima", np.array([])), dtype=np.float64)
			else:
				hybrid_daily_from_daily_pred = np.asarray(result.get("hybrid_cum_test_from_daily_arima", np.array([])), dtype=np.float64)

			metric_by_model = {}
			models_to_eval = [("SEIRC-decay", baseline), ("SEIRC-decay+ARIMA", hybrid)]
			if hybrid_daily_from_daily_pred.size == observed.size:
				models_to_eval.append(("SEIRC-decay+ARIMA_daily", hybrid_daily_from_daily_pred))

			for model, predicted in models_to_eval:
				judge1_metrics = compute_judge1_metrics(observed, predicted)
				judge2_metrics = compute_judge2_metrics(observed, predicted)
				metric_by_model[model] = {"judge1": judge1_metrics, "judge2": judge2_metrics}
				row = {
					"output_name": "judge",
					"test_horizon_days": horizon,
					"target": target,
					"model": model,
					"judge1_metric_scale": "population_percent",
					"judge1_population_base": N,
				}
				row.update({f"judge1_{name}": judge1_metrics[name] for name in JUDGE1_METRIC_NAMES})
				row["judge2_metric_scale"] = "count"
				row.update({f"judge2_{name}": judge2_metrics[name] for name in JUDGE2_METRIC_NAMES})
				rows.append(row)

			for hybrid_model in ("SEIRC-decay+ARIMA", "SEIRC-decay+ARIMA_daily"):
				if hybrid_model not in metric_by_model:
					continue
				row = {
					"output_name": "judge",
					"test_horizon_days": horizon,
					"target": target,
					"model": f"{hybrid_model} reduction (%)",
					"judge1_metric_scale": "percent_reduction",
					"judge1_population_base": N,
				}
				for metric_name in JUDGE1_METRIC_NAMES:
					row[f"judge1_{metric_name}"] = error_reduction_pct(
						metric_by_model["SEIRC-decay"]["judge1"][metric_name],
						metric_by_model[hybrid_model]["judge1"][metric_name],
					)
				row["judge2_metric_scale"] = "percent_reduction"
				for metric_name in JUDGE2_METRIC_NAMES:
					row[f"judge2_{metric_name}"] = error_reduction_pct(
						metric_by_model["SEIRC-decay"]["judge2"][metric_name],
						metric_by_model[hybrid_model]["judge2"][metric_name],
					)
				rows.append(row)
	return pd.DataFrame(rows)


def build_residual_diagnostics_table(results):
	rows = []
	for result in results:
		n_train = result["n_train"]
		for model_name, prefix in (("SEIRC-decay", "seirc"), ("SEIRC-decay+ARIMA", "hybrid")):
			for segment in ("train", "test", "all"):
				run_tests = segment == "train"
				row = {"test_horizon_days": result["horizon"], "model": model_name, "segment": segment, "residual_target": "cumulative_cases"}
				row.update(summarize_residuals(result[f"{prefix}_resid_{segment}"], run_ljung_box=run_tests, run_normality_tests=run_tests))
				rows.append(row)

		observed_cumulative = np.asarray(result.get("y_all", np.array([])), dtype=np.float64)
		hybrid_cum_train_from_daily_arima = np.asarray(result.get("hybrid_cum_train_from_daily_arima", np.array([])), dtype=np.float64)
		hybrid_cum_test_from_daily_arima = np.asarray(result.get("hybrid_cum_test_from_daily_arima", np.array([])), dtype=np.float64)
		hybrid_cum_all_from_daily_arima = (
			np.concatenate([hybrid_cum_train_from_daily_arima, hybrid_cum_test_from_daily_arima])
			if hybrid_cum_train_from_daily_arima.size + hybrid_cum_test_from_daily_arima.size > 0
			else np.array([], dtype=np.float64)
		)
		if observed_cumulative.size == hybrid_cum_all_from_daily_arima.size and hybrid_cum_all_from_daily_arima.size > 0:
			hybrid_daily_arima_cum_resid_all = observed_cumulative - hybrid_cum_all_from_daily_arima
			for segment, series_segment in (
				("train", hybrid_daily_arima_cum_resid_all[:n_train]),
				("test", hybrid_daily_arima_cum_resid_all[n_train:]),
				("all", hybrid_daily_arima_cum_resid_all),
			):
				run_tests = segment == "train"
				row = {"test_horizon_days": result["horizon"], "model": "SEIRC-decay+ARIMA_daily", "segment": segment, "residual_target": "cumulative_cases"}
				row.update(summarize_residuals(series_segment, run_ljung_box=run_tests, run_normality_tests=run_tests))
				rows.append(row)

		observed_daily = np.asarray(result.get("y_all_daily", np.array([])), dtype=np.float64)
		seirc_daily_train = np.asarray(result.get("seirc_daily_resid_train", np.array([])), dtype=np.float64)
		seirc_daily_test = np.asarray(result.get("seirc_daily_resid_test", np.array([])), dtype=np.float64)
		seirc_daily_resid_all = (
			np.concatenate([seirc_daily_train, seirc_daily_test])
			if seirc_daily_train.size + seirc_daily_test.size > 0
			else np.array([], dtype=np.float64)
		)

		hybrid_daily_all = np.asarray(result.get("hybrid_daily_all", np.array([])), dtype=np.float64)
		hybrid_daily_train = hybrid_daily_all[:n_train] if hybrid_daily_all.size > 0 else np.array([], dtype=np.float64)
		hybrid_daily_test = hybrid_daily_all[n_train:] if hybrid_daily_all.size > 0 else np.array([], dtype=np.float64)
		hybrid_daily_resid_train = observed_daily[:n_train] - hybrid_daily_train if observed_daily.size >= n_train and hybrid_daily_train.size > 0 else np.array([], dtype=np.float64)
		hybrid_daily_resid_test = observed_daily[n_train:] - hybrid_daily_test if observed_daily.size > n_train and hybrid_daily_test.size > 0 else np.array([], dtype=np.float64)
		hybrid_daily_resid_all = (
			np.concatenate([hybrid_daily_resid_train, hybrid_daily_resid_test])
			if hybrid_daily_resid_train.size + hybrid_daily_resid_test.size > 0
			else np.array([], dtype=np.float64)
		)

		hybrid_daily_train_from_daily_arima = np.asarray(result.get("hybrid_daily_train_from_daily_arima", np.array([])), dtype=np.float64)
		hybrid_daily_test_from_daily_arima = np.asarray(result.get("hybrid_daily_test_from_daily_arima", np.array([])), dtype=np.float64)
		hybrid_daily_all_from_daily_arima = (
			np.concatenate([hybrid_daily_train_from_daily_arima, hybrid_daily_test_from_daily_arima])
			if hybrid_daily_train_from_daily_arima.size + hybrid_daily_test_from_daily_arima.size > 0
			else np.array([], dtype=np.float64)
		)
		hybrid_daily_arima_resid_all = (
			observed_daily - hybrid_daily_all_from_daily_arima
			if observed_daily.size == hybrid_daily_all_from_daily_arima.size and hybrid_daily_all_from_daily_arima.size > 0
			else np.array([], dtype=np.float64)
		)

		for model_name, series in (
			("SEIRC-decay", seirc_daily_resid_all),
			("SEIRC-decay+ARIMA", hybrid_daily_resid_all),
			("SEIRC-decay+ARIMA_daily", hybrid_daily_arima_resid_all),
		):
			for segment, series_segment in (
				("train", series[:n_train] if series.size > 0 else np.array([], dtype=np.float64)),
				("test", series[n_train:] if series.size > n_train else np.array([], dtype=np.float64)),
				("all", series),
			):
				run_tests = segment == "train"
				row = {"test_horizon_days": result["horizon"], "model": model_name, "segment": segment, "residual_target": "daily_cases"}
				row.update(summarize_residuals(series_segment, run_ljung_box=run_tests, run_normality_tests=run_tests))
				rows.append(row)
	return pd.DataFrame(rows)


def plot_comparison(results, output_prefix="seirc_decay_vs_seirc_decay_arima_h"):
	output_files = []
	apply_plot_style()
	observed_color = "#b2182b"
	baseline_color = "black"
	hybrid_color = "#2166ac"
	interval_color = "#92c5de"
	grid_color = "#d9d9d9"

	def format_main_axis(ax, ylabel, locator, log_scale=False, add_legend=True):
		ax.set_ylabel(ylabel)
		if log_scale:
			ax.set_yscale("log")
		ax.grid(True, axis="y", which="major", color=grid_color, linestyle="-", linewidth=0.65, alpha=0.75)
		ax.grid(False, axis="x")
		ax.set_axisbelow(True)
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
		for spine in ax.spines.values():
			spine.set_linewidth(0.8)
		if add_legend:
			style_plot_legend(ax)

	for result in results:
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		dates_test = pd.to_datetime(result["dates_test"])
		n_train = result["n_train"]
		split_date = dates[n_train]

		scale_specs = [
			("percent", to_population_percent, 1e-12, "Daily new cases (% of population)", "Cumulative cases (% of population)"),
			("count", lambda values: np.asarray(values, dtype=np.float64), 1.0, "Daily new cases", "Cumulative cases"),
		]
		for scale_name, transform, cumulative_floor, daily_ylabel, cumulative_ylabel in scale_specs:
			y_all = np.clip(transform(result["y_all"]), cumulative_floor, None)
			y_all_daily = transform(result["y_all_daily"])
			seirc = np.clip(transform(result["seirc_all"]), cumulative_floor, None)
			daily_seirc_all = np.clip(transform(result["daily_seirc_all"]), 0.0, None)
			hybrid_daily_all = np.clip(transform(result["hybrid_daily_all"]), 0.0, None)
			arima_only_daily_low_test = np.clip(transform(result["arima_only_daily_low_test"]), 0.0, None)
			arima_only_daily_high_test = np.clip(transform(result["arima_only_daily_high_test"]), 0.0, None)
			hybrid = np.clip(transform(result["hybrid_all"]), cumulative_floor, None)
			arima_only_low_test = np.clip(transform(result["arima_only_low_test"]), cumulative_floor, None)
			arima_only_high_test = np.clip(transform(result["arima_only_high_test"]), cumulative_floor, None)
			arima_daily_only_daily_low_test = np.clip(transform(result.get("arima_daily_only_daily_low_test", np.array([]))), 0.0, None)
			arima_daily_only_daily_high_test = np.clip(transform(result.get("arima_daily_only_daily_high_test", np.array([]))), 0.0, None)
			arima_daily_only_low_test = np.clip(transform(result.get("arima_daily_only_low_test", np.array([]))), cumulative_floor, None)
			arima_daily_only_high_test = np.clip(transform(result.get("arima_daily_only_high_test", np.array([]))), cumulative_floor, None)

			window_start = max(0, n_train - 1)
			dates_window = dates[window_start:]
			y_window = y_all[window_start:]
			seirc_window = seirc[window_start:]
			daily_plot_start = pd.to_datetime(START_DATE)
			daily_plot_mask = dates >= daily_plot_start
			daily_plot_dates = dates[daily_plot_mask]
			daily_plot_observed = y_all_daily[daily_plot_mask]
			daily_plot_seirc = daily_seirc_all[daily_plot_mask]
			right_pad = pd.Timedelta(days=max(1, int(np.ceil(horizon * 0.12))))

			hybrid_daily_all_from_daily_arima = np.array([], dtype=np.float64)
			if "hybrid_daily_train_from_daily_arima" in result and "hybrid_daily_test_from_daily_arima" in result:
				daily_arima_train = np.asarray(result.get("hybrid_daily_train_from_daily_arima", np.array([])), dtype=np.float64)
				daily_arima_test = np.asarray(result.get("hybrid_daily_test_from_daily_arima", np.array([])), dtype=np.float64)
				if daily_arima_train.size + daily_arima_test.size > 0:
					hybrid_daily_all_from_daily_arima = np.concatenate([daily_arima_train, daily_arima_test])

			hybrid_cum_all_from_daily_arima = np.array([], dtype=np.float64)
			if "hybrid_cum_train_from_daily_arima" in result and "hybrid_cum_test_from_daily_arima" in result:
				cum_arima_train = np.asarray(result.get("hybrid_cum_train_from_daily_arima", np.array([])), dtype=np.float64)
				cum_arima_test = np.asarray(result.get("hybrid_cum_test_from_daily_arima", np.array([])), dtype=np.float64)
				if cum_arima_train.size + cum_arima_test.size > 0:
					hybrid_cum_all_from_daily_arima = np.concatenate([cum_arima_train, cum_arima_test])

			def save_daily_scheme_plot(scheme_suffix, scheme_label, daily_series, interval_low, interval_high):
				series_dates = dates[n_train - 1 :]
				series_values = daily_series[n_train - 1 :]
				series_plot_mask = series_dates >= daily_plot_start
				fig_daily, ax_daily = plt.subplots(figsize=(7.8, 5.2))
				ax_daily.plot(daily_plot_dates, daily_plot_observed, color=observed_color, linestyle="None", marker="*", markersize=5.0, markerfacecolor=observed_color, markeredgewidth=0.45, alpha=0.9, label="Observed")
				ax_daily.plot(daily_plot_dates, daily_plot_seirc, color=baseline_color, linestyle="--", linewidth=1.9, label="SEIRC-decay")
				ax_daily.plot(series_dates[series_plot_mask], series_values[series_plot_mask], color=hybrid_color, linewidth=1.55, alpha=0.98, label=scheme_label)
				if interval_low.size == dates_test.size and interval_high.size == dates_test.size:
					ax_daily.fill_between(dates_test, interval_low, interval_high, color=interval_color, alpha=0.35, label="ARIMA 95% interval")
				ax_daily.axvline(split_date, color="black", linestyle=":", linewidth=1.2, label="Prediction start")
				format_main_axis(ax_daily, daily_ylabel, mdates.AutoDateLocator(minticks=4, maxticks=7), add_legend=False)
				style_plot_legend(ax_daily, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02))
				fig_daily.tight_layout()
				daily_output_path = FIGURES_DIR / f"daily_new_cases_{scale_name}_{scheme_suffix}_h{horizon}.png"
				fig_daily.savefig(daily_output_path, dpi=PLOT_DPI, bbox_inches="tight")
				plt.close(fig_daily)
				output_files.append(str(daily_output_path))

			def save_cumulative_scheme_plot(scheme_suffix, scheme_label, cumulative_series, interval_low, interval_high):
				cumulative_window = cumulative_series[window_start:]
				has_interval = interval_low.size == dates_test.size and interval_high.size == dates_test.size
				fig_cum, ax = plt.subplots(figsize=(8.4, 5.8))
				ax.plot(dates, y_all, color=observed_color, linestyle="None", marker="*", markersize=5.2, markerfacecolor=observed_color, markeredgewidth=0.5, alpha=0.9, label="Observed")
				ax.plot(dates, seirc, color=baseline_color, linestyle="--", linewidth=2.0, label="SEIRC-decay")
				ax.plot(dates[n_train - 1 :], cumulative_series[n_train - 1 :], color=hybrid_color, linewidth=1.6, alpha=0.98, label=scheme_label)
				if has_interval:
					ax.fill_between(dates_test, interval_low, interval_high, color=interval_color, alpha=0.35, label="ARIMA 95% interval")
				ax.axvline(split_date, color="black", linestyle=":", linewidth=1.25, label="Prediction start")
				format_main_axis(ax, cumulative_ylabel, mdates.AutoDateLocator(minticks=5, maxticks=8), log_scale=True, add_legend=False)
				style_plot_legend(ax, loc="lower left", ncol=2, bbox_to_anchor=(0.0, 1.02))

				ax_zoom = ax.inset_axes([0.56, 0.14, 0.38, 0.34])
				ax_zoom.set_facecolor("white")
				ax_zoom.plot(dates_window, y_window, color=observed_color, linestyle="None", marker="*", markersize=4.6, markerfacecolor=observed_color, markeredgewidth=0.45, alpha=0.9)
				ax_zoom.plot(dates_window, seirc_window, color=baseline_color, linestyle="--", linewidth=1.7)
				ax_zoom.plot(dates_window, cumulative_window, color=hybrid_color, linewidth=1.35)
				if has_interval:
					ax_zoom.fill_between(dates_test, interval_low, interval_high, color=interval_color, alpha=0.35)
				ax_zoom.axvline(split_date, color="black", linestyle=":", linewidth=1.15)
				y_candidate_parts = [np.asarray(y_window, dtype=np.float64), seirc_window, cumulative_window]
				if has_interval:
					y_candidate_parts.extend([interval_low, interval_high])
				y_candidates = np.concatenate(y_candidate_parts)
				y_candidates = y_candidates[np.isfinite(y_candidates)]
				y_candidates = y_candidates[y_candidates > 0]
				if y_candidates.size > 0:
					ax_zoom.set_ylim(float(np.min(y_candidates)) * 0.9, float(np.max(y_candidates)) * 1.12)
				ax_zoom.set_xlim(dates_window[0], dates_window[-1] + right_pad)
				ax_zoom.set_yscale("log")
				ax_zoom.grid(True, axis="y", which="major", color=grid_color, linestyle="-", linewidth=0.55, alpha=0.7)
				ax_zoom.grid(False, axis="x")
				zoom_locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
				ax_zoom.xaxis.set_major_locator(zoom_locator)
				ax_zoom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(zoom_locator))
				ax_zoom.tick_params(axis="both", which="major", length=3.0, width=0.7, labelsize=PLOT_INSET_TICK_LABEL_SIZE)
				for spine in ax_zoom.spines.values():
					spine.set_linewidth(0.75)
					spine.set_edgecolor("#4d4d4d")
				try:
					mark_artists = mark_inset(ax, ax_zoom, loc1=2, loc2=4, fc="none", ec="#8a8a8a", lw=0.7)
					for artist in mark_artists:
						artist.set_alpha(0.75)
				except Exception:
					pass

				fig_cum.tight_layout()
				cumulative_output_path = FIGURES_DIR / f"cumulative_cases_with_test_zoom_{scale_name}_{scheme_suffix}_h{horizon}.png"
				fig_cum.savefig(cumulative_output_path, dpi=PLOT_DPI, bbox_inches="tight")
				plt.close(fig_cum)
				output_files.append(str(cumulative_output_path))

			scheme_specs = [
				(
					"cumulative_residual_arima_only_interval",
					"SEIRC-decay+ARIMA",
					hybrid_daily_all,
					hybrid,
					arima_only_daily_low_test,
					arima_only_daily_high_test,
					arima_only_low_test,
					arima_only_high_test,
				),
			]
			if hybrid_daily_all_from_daily_arima.size == dates.size and hybrid_cum_all_from_daily_arima.size == dates.size:
				scheme_specs.append(
					(
						"daily_residual_arima_only_interval",
						"SEIRC-decay+ARIMA_daily",
						np.clip(transform(hybrid_daily_all_from_daily_arima), 0.0, None),
						np.clip(transform(hybrid_cum_all_from_daily_arima), cumulative_floor, None),
						arima_daily_only_daily_low_test,
						arima_daily_only_daily_high_test,
						arima_daily_only_low_test,
						arima_daily_only_high_test,
					)
				)

			for scheme_suffix, scheme_label, daily_series, cumulative_series, daily_low, daily_high, cumulative_low, cumulative_high in scheme_specs:
				save_daily_scheme_plot(scheme_suffix, scheme_label, daily_series, daily_low, daily_high)
				save_cumulative_scheme_plot(scheme_suffix, scheme_label, cumulative_series, cumulative_low, cumulative_high)
	return output_files


def plot_instantaneous_reproduction_numbers(results, output_prefix="time-varying effective reproduction number_h"):
	apply_plot_style()
	output_files = []
	for result in results:
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		n_train = result["n_train"]
		split_date = dates[n_train]
		rt = np.asarray(result["instantaneous_reproduction_number_all"], dtype=np.float64)
		rt_low = np.asarray(result["instantaneous_reproduction_number_low_all"], dtype=np.float64)
		rt_high = np.asarray(result["instantaneous_reproduction_number_high_all"], dtype=np.float64)

		fig, ax = plt.subplots(figsize=(7.8, 5.0))
		ax.fill_between(dates, rt_low, rt_high, color="#92c5de", alpha=0.35, label="MCMC 95% credible interval")
		ax.plot(dates, rt, color="black", linewidth=2.0, label=r"$R_t$")
		ax.axhline(1.0, color="#b2182b", linestyle="--", linewidth=1.25, label=r"$R_t = 1$")
		ax.axvline(split_date, color="#ff7f0e", linestyle=":", linewidth=1.25, label="Prediction start")
		ax.set_ylabel(r"$R_t$")
		ax.grid(True, axis="y", color="#d9d9d9", linestyle="-", linewidth=0.65, alpha=0.75)
		ax.grid(False, axis="x")
		locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		ax.tick_params(axis="both", which="major", length=3.5, width=0.8)
		for spine in ax.spines.values():
			spine.set_linewidth(0.8)
		style_plot_legend(ax, loc="upper right", fontsize=PLOT_SMALL_LEGEND_SIZE)

		fig.tight_layout()
		output_path = FIGURES_DIR / f"{output_prefix}{horizon}.png"
		fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
		plt.close(fig)
		output_files.append(str(output_path))
	return output_files


def plot_residuals_and_diagnostics(results, output_path):
	apply_plot_style()
	n_rows = len(results)
	fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.6 * n_rows), squeeze=False)
	for row_idx, result in enumerate(results):
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		n_train = result["n_train"]
		ax_series = axes[row_idx, 0]
		seirc_resid_all = to_population_percent(result["seirc_resid_all"])
		hybrid_resid_all = to_population_percent(result["hybrid_resid_all"])
		ax_series.plot(dates, seirc_resid_all, color="#7f7f7f", linestyle="--", linewidth=1.8, label="SEIRC-decay cumulative residual")
		ax_series.plot(dates, hybrid_resid_all, color="#1f77b4", linewidth=1.8, label="SEIRC-decay+ARIMA cumulative residual")
		ax_series.axhline(0.0, color="black", linewidth=1.0)
		ax_series.axvline(dates[n_train], color="#ff7f0e", linestyle=":", linewidth=1.4)
		ax_series.set_title(f"H={horizon} Cumulative Residual Series")
		ax_series.set_ylabel("Residual (% of population)")
		ax_series.grid(alpha=0.25)
		if row_idx == 0:
			style_plot_legend(ax_series, loc="upper left")

		train_resid = to_population_percent(result["hybrid_resid_train"])
		ax_hist = axes[row_idx, 1]
		ax_hist.hist(train_resid, bins=14, density=True, color="#9ecae1", alpha=0.85, edgecolor="white")
		mu = float(np.mean(train_resid))
		sigma = float(np.std(train_resid, ddof=1)) if len(train_resid) > 1 else 0.0
		if sigma > 0.0:
			x = np.linspace(mu - 4.0 * sigma, mu + 4.0 * sigma, 220)
			ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), color="#d62728", linewidth=1.8)
		ax_hist.set_title(f"H={horizon} Train Cumulative Residual Histogram")

		ax_acf = axes[row_idx, 2]
		if len(train_resid) > 2:
			plot_acf(train_resid, lags=min(20, len(train_resid) - 1), alpha=0.05, zero=False, ax=ax_acf)
		ax_acf.set_title(f"H={horizon} Train Cumulative Residual ACF")

		ax_qq = axes[row_idx, 3]
		if len(train_resid) > 2:
			stats.probplot(train_resid, dist="norm", plot=ax_qq)
		ax_qq.set_title(f"H={horizon} Train Cumulative Residual Q-Q")

	fig.tight_layout()
	fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
	plt.close(fig)


def prepare_data(csv_path):
	df = pd.read_csv(csv_path)
	df["date"] = pd.to_datetime(df["date"])
	if "cummulative_cases" in df.columns:
		df["cumulative_cases"] = df["cummulative_cases"].astype(np.float64)
	elif "cumulative_cases" in df.columns:
		df["cumulative_cases"] = df["cumulative_cases"].astype(np.float64)
	else:
		raise KeyError("Neither cummulative_cases nor cumulative_cases found in data")
	if "daily_cases" not in df.columns:
		df["daily_cases"] = df["cumulative_cases"].diff().fillna(df["cumulative_cases"]).astype(np.float64)
	df["cumulative_anchor"] = df["cumulative_cases"] - df["daily_cases"]
	end_date = df["date"].max().strftime("%Y-%m-%d")
	mask = (df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(end_date))
	window = df.loc[mask, ["date", "cumulative_cases", "daily_cases", "cumulative_anchor"]].reset_index(drop=True)
	if len(window) <= max(TEST_HORIZONS):
		raise ValueError("Insufficient data points for requested test horizons")
	return window, end_date


def main():
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	data_window, end_date = prepare_data(OBSERVED_DATA_PATH)
	results = [evaluate_single_horizon(data_window, h) for h in TEST_HORIZONS]
	params_df = build_parameter_table(results)
	judge_df = build_judge_table(results)
	residual_diag_df = build_residual_diagnostics_table(results)
	params_path = DATA_DIR / "estimated_parameters.csv"
	judge_path = DATA_DIR / "judge.csv"
	residual_diag_path = DATA_DIR / "residual_diagnostics_table.csv"
	residual_plot_path = FIGURES_DIR / "residual_diagnostics.png"
	metadata_path = DATA_DIR / "mcmc_run_metadata.csv"
	params_df.to_csv(params_path, index=False)
	judge_df.to_csv(judge_path, index=False)
	residual_diag_df.to_csv(residual_diag_path, index=False)
	pd.DataFrame(
		[
			{
				"mcmc_iter": MCMC_ITER,
				"mcmc_burn_in": MCMC_BURN_IN,
				"mcmc_chains": MCMC_CHAINS,
				"likelihood": "NegativeBinomial",
				"fit_target": "daily_cases",
				"arima_target": "cumulative_residual,daily_residual",
				"metric_output_policy": "single_judge_table_with_prefixed_judge1_and_judge2_metrics",
				"judge_file": judge_path.name,
				"judge1_metrics": "MAE,MSE,MSLE,Normalized MAE,Normalized MSE,Maximum deviation",
				"judge1_metric_scale": "population percent = count / N * 100",
				"judge1_population_base": N,
				"judge2_metrics": "MAE,RMSE,WAPE(%)",
				"judge2_metric_scale": "original count scale",
				"judge2_wape_definition": "100 * sum(abs(y_test - y_pred)) / sum(abs(y_test))",
				"prediction_interval_method": "Prediction plots use separate ARIMA-only 95% intervals for cumulative-residual and daily-residual ARIMA targets; each interval uses a fixed SEIRC-decay trajectory plus ARIMA residual forecast uncertainty",
				"arima_daily_support": "daily-residual ARIMA correction and intervals added; returns keys arima_daily_order, arima_daily_model, arima_daily_only_*, hybrid_daily_*_from_daily_arima",
				"residual_diagnostics_target": "cumulative_cases,daily_cases",
				"residual_tests_scope": "train_only",
				"instantaneous_reproduction_number": "Rt = beta0 * exp(-decay_rate * t) / gamma, plotted with MCMC 95% credible interval",
			}
		]
	).to_csv(metadata_path, index=False)
	plot_files = plot_comparison(results)
	reproduction_plot_files = plot_instantaneous_reproduction_numbers(results)
	plot_residuals_and_diagnostics(results, residual_plot_path)
	print("Date window:", START_DATE, "to", end_date)
	print("\nSaved outputs:")
	print(f"- {params_path}")
	print(f"- {judge_path}")
	print(f"- {residual_diag_path}")
	print(f"- {metadata_path}")
	for path in plot_files:
		print(f"- {path}")
	for path in reproduction_plot_files:
		print(f"- {path}")
	print(f"- {residual_plot_path}")


if __name__ == "__main__":
	main()
