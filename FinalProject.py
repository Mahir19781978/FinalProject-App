import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats


# -----------------------------
# Config and distribution list
# -----------------------------
st.set_page_config(
    page_title="Histogram Fitter",
    layout="wide"
)

DISTRIBUTIONS = {
    "Normal (norm)": stats.norm,
    "Gamma": stats.gamma,
    "Weibull (weibull_min)": stats.weibull_min,
    "Exponential (expon)": stats.expon,
    "Lognormal": stats.lognorm,
    "Beta": stats.beta,
    "Chi-squared": stats.chi2,
    "Uniform": stats.uniform,
    "Student t": stats.t,
    "Triangular (triang)": stats.triang,
    "Pareto": stats.pareto,
}


# -----------------------------
# Helper functions
# -----------------------------
def parse_text_data(text: str) -> np.ndarray:
    """
    Parse numbers from a free-form text area.
    Accepts commas, spaces, and newlines as separators.
    """
    if not text:
        return np.array([])

    tokens = re.split(r"[,\s]+", text.strip())
    values = []
    for t in tokens:
        if t == "":
            continue
        try:
            values.append(float(t))
        except ValueError:
            # Ignore non-numeric tokens
            continue
    return np.array(values)


def load_csv_data(file) -> pd.DataFrame:
    """Read uploaded CSV into a DataFrame."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception:
        return pd.DataFrame()


def build_distribution(dist_obj, params):
    """
    Given a scipy.stats distribution object and a fitted parameter tuple,
    return (shape_names, shapes, loc, scale, frozen_dist).
    """
    shape_names = []
    if dist_obj.shapes:
        # e.g. "a, b" -> ["a", "b"]
        shape_names = [s.strip() for s in dist_obj.shapes.split(",")]

    n_shapes = len(shape_names)
    shapes = params[:n_shapes]
    loc = params[n_shapes]
    scale = params[n_shapes + 1]

    frozen = dist_obj(*shapes, loc=loc, scale=scale)
    return shape_names, shapes, loc, scale, frozen


def compute_fit_metrics(frozen_dist, data: np.ndarray):
    """
    Compute a couple of simple fit quality metrics:
    - Kolmogorov-Smirnov statistic and p-value
    - RMSE between histogram and PDF evaluated at bin centers
    """
    if len(data) < 5:
        return {"ks_stat": np.nan, "ks_pvalue": np.nan, "rmse_hist_pdf": np.nan}

    # KS test
    ks_stat, ks_p = stats.kstest(data, frozen_dist.cdf)

    # Histogram vs PDF
    hist_vals, bin_edges = np.histogram(data, bins="auto", density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = frozen_dist.pdf(bin_centers)
    rmse = np.sqrt(np.mean((hist_vals - pdf_vals) ** 2))

    return {"ks_stat": ks_stat, "ks_pvalue": ks_p, "rmse_hist_pdf": rmse}


def create_param_slider(name, default, min_val=None, max_val=None):
    """
    Helper to create a float slider with a sensible range around a default value.
    """
    if np.isnan(default):
        default = 1.0

    # Generic positive-range heuristic
    if min_val is None or max_val is None:
        if default > 0:
            min_val = default * 0.2
            max_val = default * 5.0
        else:
            min_val = default - 5.0
            max_val = default + 5.0

    # Avoid zero-width sliders
    if min_val == max_val:
        max_val = min_val + 1.0

    # Ensure ordering
    if min_val > max_val:
        min_val, max_val = max_val, min_val

    return st.slider(
        name,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default),
        step=float((max_val - min_val) / 200.0),
    )


# -----------------------------
# Sidebar: data input controls
# -----------------------------
st.sidebar.title("Data Input")

data_source = st.sidebar.radio(
    "How would you like to provide data?",
    ["Type / paste data", "Upload CSV", "Use example gamma data"],
)

data = np.array([])

if data_source == "Type / paste data":
    example_text = "1.2\n2.3\n2.4\n3.1\n3.3\n4.0\n5.2"
    text = st.sidebar.text_area(
        "Enter numeric data (one value per line, or separated by commas/spaces):",
        value=example_text,
        height=200,
    )
    data = parse_text_data(text)

elif data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is not None:
        df = load_csv_data(uploaded)
        if df.empty:
            st.sidebar.error("Could not read CSV or file is empty.")
        else:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.sidebar.error("No numeric columns found in CSV.")
            else:
                col_choice = st.sidebar.selectbox(
                    "Select numeric column to analyze", numeric_cols
                )
                data = df[col_choice].dropna().values

                st.sidebar.caption("Preview of uploaded data:")
                st.sidebar.dataframe(df.head())

elif data_source == "Use example gamma data":
    st.sidebar.write("Using synthetic data from Gamma(5, loc=1, scale=1)")
    rng = np.random.default_rng(12345)
    data = stats.gamma.rvs(5, loc=1, scale=1, size=1000, random_state=rng)

# Display basic info about data
st.sidebar.markdown("---")
st.sidebar.subheader("Data summary")
if data.size > 0:
    st.sidebar.write(f"Number of points: {len(data)}")
    st.sidebar.write(f"Min: {np.min(data):.3f}")
    st.sidebar.write(f"Max: {np.max(data):.3f}")
    st.sidebar.write(f"Mean: {np.mean(data):.3f}")
    st.sidebar.write(f"Std dev: {np.std(data, ddof=1):.3f}")
else:
    st.sidebar.info("No data loaded yet.")

# -----------------------------
# Main layout
# -----------------------------
st.title("Histogram Fitter with scipy.stats")

st.markdown(
    """
This app lets you:

- Load or type numerical data  
- Fit several probability distributions from `scipy.stats`  
- Visualize the histogram and fitted curve  
- Compare **automatic** vs **manual** parameter choices  
"""
)

if data.size == 0:
    st.warning("Please provide some data in the sidebar to begin.")
    st.stop()

# Distribution selection
st.markdown("### 1. Choose a distribution to fit")
dist_name = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
dist_obj = DISTRIBUTIONS[dist_name]

# Histogram settings
col_hist1, col_hist2 = st.columns(2)
with col_hist1:
    bins = st.slider("Number of histogram bins", min_value=5, max_value=100, value=25)
with col_hist2:
    density = st.checkbox("Normalize histogram (density=True)", value=True)

# -----------------------------
# Fit distribution automatically
# -----------------------------
st.markdown("### 2. Automatic Fit")

try:
    fitted_params = dist_obj.fit(data)
    shape_names, shapes, loc, scale, frozen_auto = build_distribution(
        dist_obj, fitted_params
    )
except Exception as e:
    st.error(f"Error fitting distribution: {e}")
    st.stop()

metrics_auto = compute_fit_metrics(frozen_auto, data)

# Show fitted parameters and metrics
col_params, col_metrics = st.columns(2)

with col_params:
    st.subheader("Fitted parameters")
    if shape_names:
        for name, val in zip(shape_names, shapes):
            st.write(f"{name} = {val:.5f}")
    st.write(f"loc = {loc:.5f}")
    st.write(f"scale = {scale:.5f}")

with col_metrics:
    st.subheader("Fit quality (automatic)")
    st.write(f"KS statistic = {metrics_auto['ks_stat']:.5f}")
    st.write(f"KS p-value   = {metrics_auto['ks_pvalue']:.5f}")
    st.write(f"RMSE (hist vs PDF) = {metrics_auto['rmse_hist_pdf']:.5e}")

# -----------------------------
# Manual fitting (sliders)
# -----------------------------
st.markdown("### 3. Manual Fitting")

manual_tab, auto_tab = st.tabs(["Manual parameters", "Plot / comparison"])

# Prepare basic stats for slider ranges
data_min, data_max = float(np.min(data)), float(np.max(data))
data_std = float(np.std(data, ddof=1)) if len(data) > 1 else 1.0

with manual_tab:
    st.write(
        "Use these sliders to manually adjust distribution parameters. "
        "Defaults come from the automatic fit."
    )

    manual_shapes = []
    if shape_names:
        st.markdown("**Shape parameters**")
        for name, val in zip(shape_names, shapes):
            slider = create_param_slider(
                f"{name}",
                default=val,
                min_val=max(val * 0.2, 1e-6) if val > 0 else val - 5,
                max_val=val * 5 if val > 0 else val + 5,
            )
            manual_shapes.append(slider)
    else:
        manual_shapes = []

    st.markdown("**Location and scale**")
    loc_slider = create_param_slider(
        "loc",
        default=loc,
        min_val=data_min - 2 * data_std,
        max_val=data_max + 2 * data_std,
    )

    # scale must be positive
    default_scale = scale if scale > 0 else max(data_std, 1e-3)
    scale_slider = create_param_slider(
        "scale", default=default_scale, min_val=max(default_scale * 0.2, 1e-6)
    )

    # Build the manually parameterized frozen distribution
    try:
        frozen_manual = dist_obj(*manual_shapes, loc=loc_slider, scale=scale_slider)
        metrics_manual = compute_fit_metrics(frozen_manual, data)
    except Exception as e:
        frozen_manual = None
        metrics_manual = {
            "ks_stat": np.nan,
            "ks_pvalue": np.nan,
            "rmse_hist_pdf": np.nan,
        }
        st.error(f"Error building manual distribution: {e}")

    st.subheader("Fit quality (manual)")
    st.write(f"KS statistic = {metrics_manual['ks_stat']:.5f}")
    st.write(f"KS p-value   = {metrics_manual['ks_pvalue']:.5f}")
    st.write(f"RMSE (hist vs PDF) = {metrics_manual['rmse_hist_pdf']:.5e}")

with auto_tab:
    st.write("Histogram with automatic and manual fits overlaid.")

    x_min = data_min - 0.1 * abs(data_min)
    x_max = data_max + 0.1 * abs(data_max)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    x = np.linspace(x_min, x_max, 400)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=bins, density=density, alpha=0.4, label="Data histogram")

    # Automatic fit curve
    y_auto = frozen_auto.pdf(x)
    ax.plot(x, y_auto, label="Automatic fit", linewidth=2)

    # Manual fit curve (if available)
    if 'frozen_manual' in locals() and frozen_manual is not None:
        y_manual = frozen_manual.pdf(x)
        ax.plot(x, y_manual, linestyle="--", linewidth=2, label="Manual fit")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(f"Histogram and fitted {dist_name} distribution")
    ax.legend()

    st.pyplot(fig)

# -----------------------------
# Footer / instructions
# -----------------------------
st.markdown("---")
st.caption(
    "Tip: run this locally with `streamlit run histogram_fitter.py`. "
    "You can add more distributions by editing the `DISTRIBUTIONS` dictionary."
)




