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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

from MCMC_method import BayesianODEFitter
from Run_fast import solve_ode_numba


N = 1.3e9
SIGMA = 1.0 / 2.5
GAMMA = 1.0 / 3.5
START_DATE = "2009-05-19"
END_DATE = "2009-08-31"
TEST_HORIZONS = (7, 14, 21, 28)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs_mcmc_phase1"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
OBSERVED_DATA_PATH = BASE_DIR / "observed_cases_smoothed.csv"

MCMC_ITER = 20000
MCMC_BURN_IN = 4000
MCMC_CHAINS = 3
MCMC_USE_PRIOR_INIT = False
RANDOM_SEED = 42


@numba.jit(nopython=True)
def seirc_ode(y, t, params):
	beta, sigma, gamma, n_pop = params
	s, e, i, r, c = y

	infection = beta * s * i / n_pop
	ds = -infection
	de = infection - sigma * e
	di = sigma * e - gamma * i
	dr = gamma * i
	dc = sigma * e
	return np.array((ds, de, di, dr, dc), dtype=np.float64)


def simulate_seirc(beta, e0, i0, r0, num_days):
	y0 = np.array([N - e0 - i0 - r0, e0, i0, r0, 0.0], dtype=np.float64)
	params = np.array([beta, SIGMA, GAMMA, N], dtype=np.float64)
	t_eval = np.arange(num_days + 1, dtype=np.float64)
	return solve_ode_numba(seirc_ode, y0, t_eval, params)


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


def compute_metrics(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=np.float64)
	y_pred = np.asarray(y_pred, dtype=np.float64)
	y_pred_clip = np.clip(y_pred, 0.0, None)

	mae = float(np.mean(np.abs(y_true - y_pred)))
	mse = float(np.mean((y_true - y_pred) ** 2))
	msle = float(np.mean((np.log1p(y_true) - np.log1p(y_pred_clip)) ** 2))
	max_dev = float(np.max(np.abs(y_true - y_pred)))
	value_range = float(np.max(y_true) - np.min(y_true))
	if value_range <= 0.0:
		value_range = max(float(np.mean(np.abs(y_true))), 1.0)

	return {
		"MAE": mae,
		"MSE": mse,
		"MSLE": msle,
		"NORMALIZED MAE": float(mae / value_range),
		"NORMALIZED MSE": float(mse / (value_range**2)),
		"maximum deviation": max_dev,
	}


def estimate_ic_priors(train_daily):
	total = max(float(np.sum(train_daily[:7])), 10.0)
	return {
		"E": {"value": total * 0.35, "mu": total * 0.35, "sigma": max(total * 0.18, 10.0), "lower": 1e-3, "upper": max(total * 8.0, 80.0)},
		"I": {"value": total * 0.60, "mu": total * 0.60, "sigma": max(total * 0.25, 15.0), "lower": 1e-3, "upper": max(total * 10.0, 120.0)},
		"R": {"value": max(train_daily[0], 1.0), "mu": max(train_daily[0], 1.0), "sigma": max(train_daily[0] * 0.8, 2.0), "lower": 0.0, "upper": max(total * 4.0, 40.0)},
	}


def plot_chain_diagnostics(fitter, horizon):
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
			ax.legend(loc="upper right", fontsize=8)
	axes[-1].set_xlabel("Iteration (post burn-in)")
	fig.tight_layout()
	fig.savefig(prefix.with_name(prefix.name + "_trace.svg"), bbox_inches="tight")
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
	fig.savefig(prefix.with_name(prefix.name + "_posterior.svg"), bbox_inches="tight")
	plt.close(fig)


def save_mcmc_diagnostics(fitter, horizon):
	with contextlib.redirect_stdout(io.StringIO()):
		summary_df = fitter.summary_statistics()
	with contextlib.redirect_stdout(io.StringIO()):
		ess_dict = fitter.compute_ess()
	with contextlib.redirect_stdout(io.StringIO()):
		rhat_dict = fitter.compute_rhat()

	diag_df = pd.DataFrame(
		[
			{
				"Parameter": name,
				"ESS": ess_dict.get(name, np.nan),
				"R_hat": rhat_dict.get(name, np.nan),
			}
			for name in fitter.param_names_estimated
		]
	)
	summary_path = DATA_DIR / f"mcmc_summary_h{horizon}.csv"
	diag_path = DATA_DIR / f"mcmc_chain_diagnostics_h{horizon}.csv"
	summary_df.to_csv(summary_path, index=False)
	diag_df.to_csv(diag_path, index=False)
	plot_chain_diagnostics(fitter, horizon)
	return summary_df


def fit_seirc_mcmc(y_train_daily, cumulative_anchor, horizon):
	np.random.seed(RANDOM_SEED + horizon)
	priors = estimate_ic_priors(y_train_daily)
	time_points = np.arange(len(y_train_daily) + 1, dtype=np.float64)
	obs_daily = np.insert(y_train_daily.astype(np.float64), 0, 0.0)

	fitter = BayesianODEFitter(seirc_ode, time_points, {})
	fitter.set_compartments(["S", "E", "I", "R", "C"])
	fitter.set_fit_targets({"C": obs_daily}, target_types={"C": "diff"})
	fitter.add_parameter("beta", value=0.35, type="estimated", prior={"dist": "uniform", "lower": 0.2, "upper": 0.6})
	fitter.add_parameter("dispersion", value=4.0, type="estimated", prior={"dist": "gamma", "alpha": 10.0, "beta": 2.5})
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

	return {
		"beta": float(param_means["beta"]),
		"E0": float(param_means["IC_E"]),
		"I0": float(param_means["IC_I"]),
		"R0": float(param_means["IC_R"]),
		"dispersion": float(param_means["dispersion"]),
	}


def summarize_residuals(residuals):
	residuals = np.asarray(residuals, dtype=np.float64)
	residuals = residuals[np.isfinite(residuals)]
	n = residuals.size
	if n == 0:
		return {"sample_size": 0, "residual_mean": np.nan, "residual_std": np.nan, "residual_mae": np.nan, "residual_rmse": np.nan, "ljung_box_pvalue": np.nan, "jarque_bera_pvalue": np.nan, "shapiro_pvalue": np.nan}

	ljung = np.nan
	if n > 5:
		try:
			ljung = float(acorr_ljungbox(residuals, lags=[min(10, n - 1)], return_df=True)["lb_pvalue"].iloc[0])
		except Exception:
			pass

	jb = np.nan
	if n > 7:
		try:
			jb = float(stats.jarque_bera(residuals).pvalue)
		except Exception:
			pass

	shapiro = np.nan
	if 3 <= n <= 5000:
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
	y_test_cum = y_all_cum[n_train:]

	mcmc_fit = fit_seirc_mcmc(y_train_daily, cumulative_anchor, horizon)
	sim = simulate_seirc(mcmc_fit["beta"], mcmc_fit["E0"], mcmc_fit["I0"], mcmc_fit["R0"], n_all)
	c_seirc_all = sim[1:, 4]
	daily_seirc_all = np.diff(sim[:, 4])

	daily_resid_train = y_train_daily - daily_seirc_all[:n_train]
	arima_model, arima_order = fit_best_arima(daily_resid_train)
	fc = arima_model.get_forecast(steps=horizon)
	resid_fc = np.asarray(fc.predicted_mean, dtype=np.float64)
	ci = np.asarray(fc.conf_int(alpha=0.05), dtype=np.float64)
	resid_low = ci[:, 0]
	resid_high = ci[:, 1]

	try:
		arima_fit_train = np.asarray(arima_model.fittedvalues, dtype=np.float64)
	except Exception:
		arima_fit_train = np.asarray(arima_model.predict(start=0, end=n_train - 1), dtype=np.float64)
	if arima_fit_train.size != n_train:
		arima_fit_train = np.asarray(arima_model.predict(start=0, end=n_train - 1), dtype=np.float64)
	arima_fit_train = np.where(np.isfinite(arima_fit_train), arima_fit_train, 0.0)

	hybrid_daily_train = np.clip(daily_seirc_all[:n_train] + arima_fit_train, 0.0, None)
	hybrid_daily_test = np.clip(daily_seirc_all[n_train:] + resid_fc, 0.0, None)
	hybrid_daily_all = np.concatenate([hybrid_daily_train, hybrid_daily_test])
	hybrid_all = cumulative_anchor + np.cumsum(hybrid_daily_all)

	hybrid_low_test = cumulative_anchor + np.cumsum(np.concatenate([hybrid_daily_train, np.clip(daily_seirc_all[n_train:] + resid_low, 0.0, None)]))[n_train:]
	hybrid_high_test = cumulative_anchor + np.cumsum(np.concatenate([hybrid_daily_train, np.clip(daily_seirc_all[n_train:] + resid_high, 0.0, None)]))[n_train:]

	return {
		"horizon": horizon,
		"n_train": n_train,
		"dates_all": dates_all,
		"dates_test": dates_test,
		"y_all": y_all_cum,
		"y_all_daily": y_all_daily,
		"daily_seirc_all": daily_seirc_all,
		"hybrid_daily_all": hybrid_daily_all,
		"hybrid_daily_test": hybrid_daily_test,
		"hybrid_daily_low_test": np.clip(daily_seirc_all[n_train:] + resid_low, 0.0, None),
		"hybrid_daily_high_test": np.clip(daily_seirc_all[n_train:] + resid_high, 0.0, None),
		"seirc_all": c_seirc_all,
		"hybrid_all": hybrid_all,
		"hybrid_low_test": hybrid_low_test,
		"hybrid_high_test": hybrid_high_test,
		"seirc_resid_train": y_all_cum[:n_train] - c_seirc_all[:n_train],
		"seirc_resid_test": y_test_cum - c_seirc_all[n_train:],
		"seirc_resid_all": y_all_cum - c_seirc_all,
		"hybrid_resid_train": y_all_cum[:n_train] - hybrid_all[:n_train],
		"hybrid_resid_test": y_test_cum - hybrid_all[n_train:],
		"hybrid_resid_all": y_all_cum - hybrid_all,
		"metrics_seirc": compute_metrics(y_test_cum, c_seirc_all[n_train:]),
		"metrics_hybrid": compute_metrics(y_test_cum, hybrid_all[n_train:]),
		"params": {
			"beta": mcmc_fit["beta"],
			"E0": mcmc_fit["E0"],
			"I0": mcmc_fit["I0"],
			"R0": mcmc_fit["R0"],
			"dispersion": mcmc_fit["dispersion"],
			"ARIMA_order": arima_order,
			"mcmc_iter": MCMC_ITER,
			"mcmc_burn_in": MCMC_BURN_IN,
			"mcmc_chains": MCMC_CHAINS,
		},
	}


def build_tables(results):
	metric_rows = []
	param_rows = []
	for result in results:
		params = result["params"]
		horizon = result["horizon"]
		param_rows.append({"test_horizon_days": horizon, "beta": params["beta"], "E0": params["E0"], "I0": params["I0"], "R0": params["R0"], "dispersion": params["dispersion"], "ARIMA_order": str(params["ARIMA_order"]), "mcmc_iter": params["mcmc_iter"], "mcmc_burn_in": params["mcmc_burn_in"], "mcmc_chains": params["mcmc_chains"]})
		row1 = {"test_horizon_days": horizon, "model": "SEIRC"}
		row1.update(result["metrics_seirc"])
		metric_rows.append(row1)
		row2 = {"test_horizon_days": horizon, "model": "SEIRC+ARIMA"}
		row2.update(result["metrics_hybrid"])
		metric_rows.append(row2)
	return pd.DataFrame(param_rows), pd.DataFrame(metric_rows)


def build_residual_diagnostics_table(results):
	rows = []
	for result in results:
		for model_name, prefix in (("SEIRC", "seirc"), ("SEIRC+ARIMA", "hybrid")):
			for segment in ("train", "test", "all"):
				row = {"test_horizon_days": result["horizon"], "model": model_name, "segment": segment}
				row.update(summarize_residuals(result[f"{prefix}_resid_{segment}"]))
				rows.append(row)
	return pd.DataFrame(rows)


def plot_comparison(results, output_prefix="seirc_vs_seirc_arima_h"):
	output_files = []
	for result in results:
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		dates_test = pd.to_datetime(result["dates_test"])
		y_all = result["y_all"]
		y_all_daily = result["y_all_daily"]
		n_train = result["n_train"]
		seirc = np.clip(result["seirc_all"], 1.0, None)
		daily_seirc_all = np.clip(result["daily_seirc_all"], 0.0, None)
		hybrid_daily_all = np.clip(result["hybrid_daily_all"], 0.0, None)
		hybrid_daily_test = np.clip(result["hybrid_daily_test"], 0.0, None)
		hybrid_daily_low_test = np.clip(result["hybrid_daily_low_test"], 0.0, None)
		hybrid_daily_high_test = np.clip(result["hybrid_daily_high_test"], 0.0, None)
		hybrid = np.clip(result["hybrid_all"], 1.0, None)
		hybrid_low_test = np.clip(result["hybrid_low_test"], 1.0, None)
		hybrid_high_test = np.clip(result["hybrid_high_test"], 1.0, None)

		window_start = max(0, n_train - max(7, horizon // 2))
		dates_window = dates[window_start:]
		y_window = y_all[window_start:]
		seirc_window = seirc[window_start:]

		plt.rcParams.update({"font.family": "monospace", "font.size": 10.5, "axes.labelsize": 10.5, "axes.titlesize": 12, "legend.fontsize": 9.5})
		fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), gridspec_kw={"width_ratios": [1.35, 1.45, 1.0]})
		ax_daily, ax, ax_zoom = axes

		ax_daily.plot(dates, y_all_daily, color="#ff4d4d", linestyle="None", marker="*", markersize=5.6, markerfacecolor="#ff4d4d", markeredgewidth=0.5, label="Observed daily cases")
		ax_daily.plot(dates, daily_seirc_all, color="#7a7a7a", linestyle="--", linewidth=1.3, label="SEIR baseline")
		ax_daily.plot(dates[n_train - 1 :], hybrid_daily_all[n_train - 1 :], color="blue", linewidth=1.4, alpha=0.95, label="SEIR+ARIMA")
		ax_daily.fill_between(dates_test, hybrid_daily_low_test, hybrid_daily_high_test, color="#9ecae1", alpha=0.55, label="95% prediction interval")
		split_date = dates[n_train]
		ax_daily.axvline(split_date, color="black", linestyle=":", linewidth=1.4, label="Prediction Start")
		ax_daily.set_title("Daily New Cases")
		ax_daily.set_ylabel("Cases")
		ax_daily.grid(True, which="major", color="#c8ced8", linestyle=":", linewidth=0.8, alpha=0.95)
		ax_daily.set_axisbelow(True)
		daily_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
		ax_daily.xaxis.set_major_locator(daily_locator)
		ax_daily.xaxis.set_major_formatter(mdates.ConciseDateFormatter(daily_locator))
		ax_daily.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cfcfcf")

		ax.plot(dates, y_all, color="#ff4d4d", linestyle="None", marker="*", markersize=6.2, markerfacecolor="#ff4d4d", markeredgewidth=0.6, label="Observed")
		ax.plot(dates, seirc, color="#7a7a7a", linestyle="--", linewidth=1.5, label="SEIR Baseline")
		ax.plot(dates[n_train - 1 :], hybrid[n_train - 1 :], color="blue", linewidth=1.5, alpha=0.95, label="SEIR+ARIMA (daily residual)")
		ax.fill_between(dates_test, hybrid_low_test, hybrid_high_test, color="#9ecae1", alpha=0.55, label="95% prediction interval")
		ax.axvline(split_date, color="black", linestyle=":", linewidth=1.6, label="Prediction Start")
		ax.set_title("Cumulative Cases")
		ax.set_ylabel("Cumulative Cases")
		ax.set_yscale("log")
		ax.grid(True, which="major", color="#c8ced8", linestyle=":", linewidth=0.8, alpha=0.95)
		ax.set_axisbelow(True)
		locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cfcfcf")

		ax_zoom.plot(dates_window, y_window, color="#ff4d4d", linestyle="None", marker="*", markersize=6.4, markerfacecolor="#ff4d4d", markeredgewidth=0.6)
		ax_zoom.plot(dates_window, seirc_window, color="#7a7a7a", linestyle="--", linewidth=1.5)
		ax_zoom.plot(dates[n_train - 1 :], hybrid[n_train - 1 :], color="blue", linewidth=1.5)
		ax_zoom.fill_between(dates_test, hybrid_low_test, hybrid_high_test, color="#9ecae1", alpha=0.55)
		ax_zoom.axvline(split_date, color="black", linestyle=":", linewidth=1.6)
		y_candidates = np.concatenate([np.asarray(y_window, dtype=np.float64), hybrid_low_test, hybrid_high_test, seirc_window, hybrid[n_train - 1 :]])
		y_candidates = y_candidates[np.isfinite(y_candidates)]
		y_candidates = y_candidates[y_candidates > 0]
		if y_candidates.size > 0:
			ax_zoom.set_ylim(float(np.min(y_candidates)) * 0.9, float(np.max(y_candidates)) * 1.12)
		right_pad = pd.Timedelta(days=max(1, int(np.ceil(horizon * 0.12))))
		ax_zoom.set_xlim(dates_window[0], dates_window[-1] + right_pad)
		ax_zoom.set_title(f"Test Segment Cumulative (H = {horizon})")
		ax_zoom.set_yscale("log")
		ax_zoom.grid(True, which="major", color="#c8ced8", linestyle=":", linewidth=0.8, alpha=0.95)
		zoom_locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
		ax_zoom.xaxis.set_major_locator(zoom_locator)
		ax_zoom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(zoom_locator))

		fig.tight_layout()
		output_path = FIGURES_DIR / f"{output_prefix}{horizon}.svg"
		fig.savefig(output_path, bbox_inches="tight")
		plt.close(fig)
		output_files.append(str(output_path))
	return output_files


def plot_residuals_and_diagnostics(results, output_path):
	n_rows = len(results)
	fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.6 * n_rows), squeeze=False)
	for row_idx, result in enumerate(results):
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		n_train = result["n_train"]
		ax_series = axes[row_idx, 0]
		ax_series.plot(dates, result["seirc_resid_all"], color="#7f7f7f", linestyle="--", linewidth=1.8, label="SEIRC residual")
		ax_series.plot(dates, result["hybrid_resid_all"], color="#1f77b4", linewidth=1.8, label="SEIRC+ARIMA residual")
		ax_series.axhline(0.0, color="black", linewidth=1.0)
		ax_series.axvline(dates[n_train], color="#ff7f0e", linestyle=":", linewidth=1.4)
		ax_series.set_title(f"H={horizon} Residual Series")
		ax_series.grid(alpha=0.25)
		if row_idx == 0:
			ax_series.legend(loc="upper left", frameon=False)

		train_resid = result["hybrid_resid_train"]
		ax_hist = axes[row_idx, 1]
		ax_hist.hist(train_resid, bins=14, density=True, color="#9ecae1", alpha=0.85, edgecolor="white")
		mu = float(np.mean(train_resid))
		sigma = float(np.std(train_resid, ddof=1)) if len(train_resid) > 1 else 0.0
		if sigma > 0.0:
			x = np.linspace(mu - 4.0 * sigma, mu + 4.0 * sigma, 220)
			ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), color="#d62728", linewidth=1.8)
		ax_hist.set_title(f"H={horizon} Train Residual Histogram")

		ax_acf = axes[row_idx, 2]
		if len(train_resid) > 2:
			plot_acf(train_resid, lags=min(20, len(train_resid) - 1), alpha=0.05, zero=False, ax=ax_acf)
		ax_acf.set_title(f"H={horizon} Train Residual ACF")

		ax_qq = axes[row_idx, 3]
		if len(train_resid) > 2:
			stats.probplot(train_resid, dist="norm", plot=ax_qq)
		ax_qq.set_title(f"H={horizon} Train Residual Q-Q")

	fig.tight_layout()
	fig.savefig(output_path, dpi=220)
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
	mask = (df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(END_DATE))
	window = df.loc[mask, ["date", "cumulative_cases", "daily_cases", "cumulative_anchor"]].reset_index(drop=True)
	if len(window) <= max(TEST_HORIZONS):
		raise ValueError("Insufficient data points for requested test horizons")
	return window


def main():
	FIGURES_DIR.mkdir(parents=True, exist_ok=True)
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	data_window = prepare_data(OBSERVED_DATA_PATH)
	results = [evaluate_single_horizon(data_window, h) for h in TEST_HORIZONS]
	params_df, metrics_df = build_tables(results)
	residual_diag_df = build_residual_diagnostics_table(results)
	params_path = DATA_DIR / "estimated_parameters.csv"
	metrics_path = DATA_DIR / "metrics_table.csv"
	residual_diag_path = DATA_DIR / "residual_diagnostics_table.csv"
	residual_plot_path = FIGURES_DIR / "residual_diagnostics.png"
	metadata_path = DATA_DIR / "mcmc_run_metadata.csv"

	params_df.to_csv(params_path, index=False)
	metrics_df.to_csv(metrics_path, index=False)
	residual_diag_df.to_csv(residual_diag_path, index=False)
	pd.DataFrame([{"mcmc_iter": MCMC_ITER, "mcmc_burn_in": MCMC_BURN_IN, "mcmc_chains": MCMC_CHAINS, "likelihood": "NegativeBinomial", "fit_target": "daily_cases", "arima_target": "daily_residual"}]).to_csv(metadata_path, index=False)

	plot_files = plot_comparison(results)
	plot_residuals_and_diagnostics(results, residual_plot_path)

	print("Date window:", START_DATE, "to", END_DATE)
	print("\nSaved outputs:")
	print(f"- {params_path}")
	print(f"- {metrics_path}")
	print(f"- {residual_diag_path}")
	print(f"- {metadata_path}")
	for path in plot_files:
		print(f"- {path}")
	print(f"- {residual_plot_path}")


if __name__ == "__main__":
	main()
