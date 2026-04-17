import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numba
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

from Run_fast import solve_ode_numba


N = 1.3e9
SIGMA = 1.0 / 2.5
GAMMA = 1.0 / 3.5
START_DATE = "2009-05-19"
END_DATE = "2009-08-31"
TEST_HORIZONS = (7, 14, 21, 28)
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
OBSERVED_DATA_PATH = BASE_DIR / "observed_cases_smoothed.csv"


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


def simulate_seirc(beta, e0, i0, c0, num_days):
	if num_days <= 0:
		raise ValueError("num_days must be positive")

	s0 = max(N - e0 - i0, 1.0)
	y0 = np.array([s0, e0, i0, 0.0, c0], dtype=np.float64)
	params = np.array([beta, SIGMA, GAMMA, N], dtype=np.float64)
	t_eval = np.arange(num_days, dtype=np.float64)

	return solve_ode_numba(seirc_ode, y0, t_eval, params)


def seirc_residuals(theta, y_obs):
	beta, e0, i0, c0 = theta

	if beta <= 0.0 or e0 <= 0.0 or i0 <= 0.0 or c0 < 1.0:
		return np.full_like(y_obs, 1e12, dtype=np.float64)

	c_pred = simulate_seirc(beta, e0, i0, c0, len(y_obs))[:, 4]
	return c_pred - y_obs


def fit_seirc_least_squares(y_train):
	c0_guess = max(float(y_train[0]), 1.0)
	lower = np.array([1e-8, 1e-6, 1e-6, 1.0], dtype=np.float64)
	upper = np.array([5.0, 1e8, 1e8, 1e7], dtype=np.float64)

	initial_guesses = [
		np.array([0.40, 100.0, 50.0, c0_guess], dtype=np.float64),
		np.array([0.60, 300.0, 100.0, c0_guess], dtype=np.float64),
		np.array([0.25, 800.0, 300.0, c0_guess], dtype=np.float64),
		np.array([1.20, 200.0, 80.0, c0_guess], dtype=np.float64),
	]

	best = None
	for x0 in initial_guesses:
		x0 = np.clip(x0, lower + 1e-8, upper - 1e-8)
		try:
			res = least_squares(
				seirc_residuals,
				x0,
				args=(y_train,),
				bounds=(lower, upper),
				method="trf",
				max_nfev=6000,
			)
		except Exception:
			continue

		if best is None or res.cost < best.cost:
			best = res

	if best is None:
		raise RuntimeError("Least squares fitting failed for all initial guesses")

	beta, e0, i0, c0 = best.x
	pred_train = simulate_seirc(beta, e0, i0, c0, len(y_train))[:, 4]
	train_resid = y_train - pred_train

	return {
		"beta": float(beta),
		"E0": float(e0),
		"I0": float(i0),
		"C0": float(c0),
		"resid_train": train_resid,
	}


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


def forecast_residual_with_ci(arima_model, horizon):
	fc = arima_model.get_forecast(steps=horizon)
	mean = np.asarray(fc.predicted_mean, dtype=np.float64)
	ci = np.asarray(fc.conf_int(alpha=0.05), dtype=np.float64)

	lower = ci[:, 0]
	upper = ci[:, 1]
	return mean, lower, upper


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
		value_range = float(np.mean(np.abs(y_true)))
	if value_range <= 0.0:
		value_range = 1.0

	nmae = float(mae / value_range)
	nmse = float(mse / (value_range**2))

	return {
		"MAE": mae,
		"MSE": mse,
		"MSLE": msle,
		"NORMALIZED MAE": nmae,
		"NORMALIZED MSE": nmse,
		"maximum deviation": max_dev,
	}


def evaluate_single_horizon(data_window, horizon):
	n_all = len(data_window)
	if horizon >= n_all:
		raise ValueError(f"horizon {horizon} is too large for sample size {n_all}")

	y_all = data_window["cumulative_cases"].to_numpy(dtype=np.float64)
	dates_all = data_window["date"].to_numpy()
	n_train = n_all - horizon

	y_train = y_all[:n_train]
	y_test = y_all[n_train:]
	dates_test = dates_all[n_train:]

	seirc_fit = fit_seirc_least_squares(y_train)
	beta = seirc_fit["beta"]
	e0 = seirc_fit["E0"]
	i0 = seirc_fit["I0"]
	c0 = seirc_fit["C0"]

	c_seirc_all = simulate_seirc(beta, e0, i0, c0, n_all)[:, 4]
	c_seirc_train = c_seirc_all[:n_train]
	c_seirc_test = c_seirc_all[n_train:]

	resid_train = seirc_fit["resid_train"]

	arima_model, arima_order = fit_best_arima(resid_train)
	resid_fc, resid_low, resid_high = forecast_residual_with_ci(arima_model, horizon)

	c_hybrid_test = c_seirc_test + resid_fc
	hybrid_low_test = c_seirc_test + resid_low
	hybrid_high_test = c_seirc_test + resid_high

	try:
		arima_fit_train = np.asarray(arima_model.fittedvalues, dtype=np.float64)
	except Exception:
		arima_fit_train = np.asarray(arima_model.predict(start=0, end=n_train - 1), dtype=np.float64)

	if arima_fit_train.size != n_train:
		arima_fit_train = np.asarray(arima_model.predict(start=0, end=n_train - 1), dtype=np.float64)

	arima_fit_train = np.where(np.isfinite(arima_fit_train), arima_fit_train, 0.0)
	c_hybrid_train = c_seirc_train + arima_fit_train

	c_hybrid_all = np.concatenate([c_hybrid_train, c_hybrid_test])

	metrics_seirc = compute_metrics(y_test, c_seirc_test)
	metrics_hybrid = compute_metrics(y_test, c_hybrid_test)

	return {
		"horizon": horizon,
		"n_train": n_train,
		"dates_all": dates_all,
		"dates_test": dates_test,
		"y_all": y_all,
		"seirc_all": c_seirc_all,
		"hybrid_all": c_hybrid_all,
		"hybrid_low_test": hybrid_low_test,
		"hybrid_high_test": hybrid_high_test,
		"seirc_resid_train": y_train - c_seirc_train,
		"seirc_resid_test": y_test - c_seirc_test,
		"seirc_resid_all": y_all - c_seirc_all,
		"hybrid_resid_train": y_train - c_hybrid_train,
		"hybrid_resid_test": y_test - c_hybrid_test,
		"hybrid_resid_all": y_all - c_hybrid_all,
		"metrics_seirc": metrics_seirc,
		"metrics_hybrid": metrics_hybrid,
		"params": {
			"beta": beta,
			"E0": e0,
			"I0": i0,
			"C0": c0,
			"ARIMA_order": arima_order,
		},
	}


def build_tables(results):
	metric_rows = []
	param_rows = []

	for result in results:
		horizon = result["horizon"]
		params = result["params"]

		param_rows.append(
			{
				"test_horizon_days": horizon,
				"beta": params["beta"],
				"E0": params["E0"],
				"I0": params["I0"],
				"C0": params["C0"],
				"ARIMA_order": str(params["ARIMA_order"]),
			}
		)

		row_seirc = {"test_horizon_days": horizon, "model": "SEIRC"}
		row_seirc.update(result["metrics_seirc"])
		metric_rows.append(row_seirc)

		row_hybrid = {"test_horizon_days": horizon, "model": "SEIRC+ARIMA"}
		row_hybrid.update(result["metrics_hybrid"])
		metric_rows.append(row_hybrid)

	metrics_df = pd.DataFrame(metric_rows)
	params_df = pd.DataFrame(param_rows)
	return params_df, metrics_df


def plot_comparison(results, output_prefix="seirc_vs_seirc_arima_h"):
	output_files = []

	for result in results:
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		dates_test = pd.to_datetime(result["dates_test"])
		y_all = result["y_all"]
		n_train = result["n_train"]

		seirc = np.clip(result["seirc_all"], 1.0, None)
		hybrid = np.clip(result["hybrid_all"], 1.0, None)
		hybrid_low_test = np.clip(result["hybrid_low_test"], 1.0, None)
		hybrid_high_test = np.clip(result["hybrid_high_test"], 1.0, None)

		plt.rcParams.update(
			{
				"font.family": "monospace",
				"font.size": 10.5,
				"axes.labelsize": 10.5,
				"axes.titlesize": 12,
				"legend.fontsize": 9.5,
			}
		)

		window_start = max(0, n_train - max(7, horizon // 2))
		dates_window = dates[window_start:]
		y_window = y_all[window_start:]
		seirc_window = seirc[window_start:]
		hybrid_window = hybrid[window_start:]

		fig, axes = plt.subplots(
			1,
			2,
			figsize=(13.2, 5.6),
			gridspec_kw={"width_ratios": [2.5, 1.2]},
		)
		ax = axes[0]
		ax_zoom = axes[1]

		ax.plot(
			dates,
			y_all,
			color="#ff4d4d",
			linestyle="None",
			marker="*",
			markersize=6.2,
			markerfacecolor="#ff4d4d",
			markeredgewidth=0.6,
			label="Observed",
		)
		ax.plot(
			dates,
			seirc,
			color="#7a7a7a",
			linestyle="--",
			linewidth=1.5,
			label="SEIR Baseline",
		)
		ax.plot(
			dates[n_train - 1 :],
			hybrid[n_train - 1 :],
			color="blue",
			linewidth=1.5,
			alpha=0.95,
			label="SEIR+ARIMA (cumsum)",
		)
		ax.fill_between(
			dates_test,
			hybrid_low_test,
			hybrid_high_test,
			color="#9ecae1",
			alpha=0.55,
			label="95% prediction interval",
		)

		if n_train < len(dates):
			split_date = dates[n_train]
			ax.axvline(split_date, color="black", linestyle=":", linewidth=1.6, label="Prediction Start")

		ax.set_title("Cumulative Cases")
		ax.set_xlabel("")
		ax.set_ylabel("Cumulative Cases")
		ax.set_yscale("log")
		ax.grid(True, which="major", color="#c8ced8", linestyle=":", linewidth=0.8, alpha=0.95)
		ax.grid(False, which="minor")
		ax.set_axisbelow(True)
		for spine in ax.spines.values():
			spine.set_linewidth(0.9)
			spine.set_color("#333333")

		locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
		ax.xaxis.set_major_locator(locator)
		ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
		ax.tick_params(axis="x", rotation=0)
		ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cfcfcf")

		ax_zoom.plot(
			dates_window,
			y_window,
			color="#ff4d4d",
			linestyle="None",
			marker="*",
			markersize=6.4,
			markerfacecolor="#ff4d4d",
			markeredgewidth=0.6,
			label="Observed",
		)
		ax_zoom.plot(
			dates_window,
			seirc_window,
			color="#7a7a7a",
			linestyle="--",
			linewidth=1.5,
			label="SEIR Baseline",
		)
		ax_zoom.plot(
			dates[n_train - 1 :],
			hybrid[n_train - 1 :],
			color="blue",
			linewidth=1.5,
			label="SEIR+ARIMA (cumsum)",
		)
		ax_zoom.fill_between(
			dates_test,
			hybrid_low_test,
			hybrid_high_test,
			color="#9ecae1",
			alpha=0.55,
			label="95% prediction interval",
		)
		ax_zoom.axvline(split_date, color="black", linestyle=":", linewidth=1.6)

		y_candidates = np.concatenate(
			[
				np.asarray(y_window, dtype=np.float64),
				hybrid_low_test,
				hybrid_high_test,
				seirc_window,
				hybrid_window,
			]
		)
		y_candidates = y_candidates[np.isfinite(y_candidates)]
		y_candidates = y_candidates[y_candidates > 0]
		if y_candidates.size > 0:
			y_min = float(np.min(y_candidates))
			y_max = float(np.max(y_candidates))
			ax_zoom.set_ylim(y_min * 0.9, y_max * 1.12)

		right_pad = pd.Timedelta(days=max(1, int(np.ceil(horizon * 0.12))))
		ax_zoom.set_xlim(dates_window[0], dates_window[-1] + right_pad)
		ax_zoom.set_title(f"Test Segment (H = {horizon})")
		ax_zoom.set_xlabel("")
		ax_zoom.set_ylabel("Cumulative Cases")
		ax_zoom.grid(True, which="major", color="#c8ced8", linestyle=":", linewidth=0.8, alpha=0.95)
		ax_zoom.set_axisbelow(True)
		for spine in ax_zoom.spines.values():
			spine.set_linewidth(0.9)
			spine.set_color("#333333")

		zoom_locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
		ax_zoom.xaxis.set_major_locator(zoom_locator)
		ax_zoom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(zoom_locator))
		ax_zoom.tick_params(axis="x", rotation=0)

		fig.tight_layout()
		output_path = FIGURES_DIR / f"{output_prefix}{horizon}.svg"
		fig.savefig(output_path, bbox_inches="tight")
		plt.close(fig)
		output_files.append(str(output_path))

	return output_files


def summarize_residuals(residuals):
	residuals = np.asarray(residuals, dtype=np.float64)
	residuals = residuals[np.isfinite(residuals)]
	n = residuals.size

	if n == 0:
		return {
			"sample_size": 0,
			"residual_mean": np.nan,
			"residual_std": np.nan,
			"residual_mae": np.nan,
			"residual_rmse": np.nan,
			"ljung_box_pvalue": np.nan,
			"jarque_bera_pvalue": np.nan,
			"shapiro_pvalue": np.nan,
		}

	mean_val = float(np.mean(residuals))
	std_val = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
	mae_val = float(np.mean(np.abs(residuals)))
	rmse_val = float(np.sqrt(np.mean(residuals**2)))

	ljung_box_pvalue = np.nan
	if n > 5:
		lag = min(10, n - 1)
		try:
			lb_df = acorr_ljungbox(residuals, lags=[lag], return_df=True)
			ljung_box_pvalue = float(lb_df["lb_pvalue"].iloc[0])
		except Exception:
			pass

	jarque_bera_pvalue = np.nan
	if n > 7:
		try:
			jb_result = stats.jarque_bera(residuals)
			jarque_bera_pvalue = float(jb_result.pvalue)
		except Exception:
			pass

	shapiro_pvalue = np.nan
	if 3 <= n <= 5000:
		try:
			shapiro_pvalue = float(stats.shapiro(residuals).pvalue)
		except Exception:
			pass

	return {
		"sample_size": int(n),
		"residual_mean": mean_val,
		"residual_std": std_val,
		"residual_mae": mae_val,
		"residual_rmse": rmse_val,
		"ljung_box_pvalue": ljung_box_pvalue,
		"jarque_bera_pvalue": jarque_bera_pvalue,
		"shapiro_pvalue": shapiro_pvalue,
	}


def build_residual_diagnostics_table(results):
	rows = []

	for result in results:
		horizon = result["horizon"]

		for model_name, key_prefix in (("SEIRC", "seirc"), ("SEIRC+ARIMA", "hybrid")):
			for segment in ("train", "test", "all"):
				residuals = result[f"{key_prefix}_resid_{segment}"]
				summary = summarize_residuals(residuals)
				row = {
					"test_horizon_days": horizon,
					"model": model_name,
					"segment": segment,
				}
				row.update(summary)
				rows.append(row)

	return pd.DataFrame(rows)


def plot_residuals_and_diagnostics(results, output_path):
	n_rows = len(results)
	fig, axes = plt.subplots(n_rows, 4, figsize=(22, 4.6 * n_rows), squeeze=False)

	for row_idx, result in enumerate(results):
		horizon = result["horizon"]
		dates = pd.to_datetime(result["dates_all"])
		n_train = result["n_train"]

		seirc_resid_all = result["seirc_resid_all"]
		hybrid_resid_all = result["hybrid_resid_all"]
		hybrid_resid_train = result["hybrid_resid_train"]

		ax_series = axes[row_idx, 0]
		ax_series.plot(dates, seirc_resid_all, color="#7f7f7f", linestyle="--", linewidth=1.8, label="SEIRC residual")
		ax_series.plot(dates, hybrid_resid_all, color="#1f77b4", linewidth=1.8, label="SEIRC+ARIMA residual")
		ax_series.axhline(0.0, color="black", linewidth=1.0)
		if n_train < len(dates):
			split_date = dates[n_train]
			ax_series.axvline(split_date, color="#ff7f0e", linestyle=":", linewidth=1.4)
			ax_series.axvspan(split_date, dates[-1], color="#f4f4f4", alpha=0.30)
		ax_series.set_title(f"H={horizon} Residual Series (Train+Test)")
		ax_series.set_xlabel("Date")
		ax_series.set_ylabel("Residual")
		ax_series.grid(alpha=0.25)
		ax_series.tick_params(axis="x", rotation=25)
		if row_idx == 0:
			ax_series.legend(loc="upper left", frameon=False)

		ax_hist = axes[row_idx, 1]
		ax_hist.hist(hybrid_resid_train, bins=14, density=True, color="#9ecae1", alpha=0.85, edgecolor="white")
		mu = float(np.mean(hybrid_resid_train))
		sigma = float(np.std(hybrid_resid_train, ddof=1)) if len(hybrid_resid_train) > 1 else 0.0
		if sigma > 0.0:
			x = np.linspace(mu - 4.0 * sigma, mu + 4.0 * sigma, 220)
			ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), color="#d62728", linewidth=1.8, label="Normal PDF")
		if row_idx == 0:
			ax_hist.legend(loc="upper right", frameon=False)
		ax_hist.set_title(f"H={horizon} Train Residual Histogram")
		ax_hist.set_xlabel("Residual")
		ax_hist.set_ylabel("Density")
		ax_hist.grid(alpha=0.20)

		ax_acf = axes[row_idx, 2]
		if len(hybrid_resid_train) > 2:
			lags = min(20, len(hybrid_resid_train) - 1)
			try:
				plot_acf(hybrid_resid_train, lags=lags, alpha=0.05, zero=False, ax=ax_acf)
				ax_acf.set_title(f"H={horizon} Train Residual ACF")
			except Exception:
				ax_acf.text(0.5, 0.5, "ACF unavailable", ha="center", va="center")
				ax_acf.set_title(f"H={horizon} Train Residual ACF")
		else:
			ax_acf.text(0.5, 0.5, "Too few points", ha="center", va="center")
			ax_acf.set_title(f"H={horizon} Train Residual ACF")

		ax_qq = axes[row_idx, 3]
		if len(hybrid_resid_train) > 2:
			stats.probplot(hybrid_resid_train, dist="norm", plot=ax_qq)
			ax_qq.set_title(f"H={horizon} Train Residual Q-Q")
		else:
			ax_qq.text(0.5, 0.5, "Too few points", ha="center", va="center")
			ax_qq.set_title(f"H={horizon} Train Residual Q-Q")

	fig.suptitle("Residual Diagnostics (SEIRC and SEIRC+ARIMA)", fontsize=14, y=0.995)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
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

	mask = (df["date"] >= pd.to_datetime(START_DATE)) & (df["date"] <= pd.to_datetime(END_DATE))
	window = df.loc[mask, ["date", "cumulative_cases"]].reset_index(drop=True)

	if window.empty:
		raise ValueError("No data found in the specified date window")

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

	params_df.to_csv(params_path, index=False)
	metrics_df.to_csv(metrics_path, index=False)
	residual_diag_df.to_csv(residual_diag_path, index=False)

	fullrange_plot_files = plot_comparison(results)
	plot_residuals_and_diagnostics(results, residual_plot_path)

	print("Date window:", START_DATE, "to", END_DATE)
	print("\nEstimated parameters (beta, E0, I0, C0):")
	print(params_df.to_string(index=False))

	print("\nMetrics table:")
	print(metrics_df.to_string(index=False))

	print("\nResidual diagnostics table:")
	print(residual_diag_df.to_string(index=False))

	print("\nSaved outputs:")
	print(f"- {params_path}")
	print(f"- {metrics_path}")
	print(f"- {residual_diag_path}")
	for path in fullrange_plot_files:
		print(f"- {path}")
	print(f"- {residual_plot_path}")


if __name__ == "__main__":
	main()
