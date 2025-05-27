#ORIGINAL CODE:




import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Animated Visualization: Compare 5X vs ΣXᵢ or X vs mean(Xᵢ)")

# --- Controls ---
dist = st.selectbox("Distribution", ["Uniform(0,1)", "Normal(0,1)", "Exponential(λ=1)"])
mode = st.radio("Plot mode", ["Sum", "Mean"], index=0)
n_samples = st.slider("Number of samples", 10, 500, 100, step=10)
frame_delay = st.slider("Animation speed (ms per frame)", 10, 2000, 1500, step=100)
confidence_slider = st.slider("Confidence level for range (e.g. 0.9999)", min_value=0.95, max_value=0.9999, value=0.99, step=1e-5, format="%.5f")
trail_opacity = st.slider(
    "Memory trail opacity for previous X and mean(Xᵢ)",
    min_value=0.0, max_value=1.0,
    value=0.1, step=0.05
)

# --- Distribution Setup ---
rng = np.random.default_rng(seed=42)
max_draws = 500  # max 500 samples, each with X and 5 Xᵢ
confidence = confidence_slider

if dist == "Uniform(0,1)":
    sample = lambda size: rng.uniform(0, 1, size=size)
    theoretical_mean = 0.5
    lower, upper = 0, 1
elif dist == "Normal(0,1)":
    sample = lambda size: rng.normal(0, 1, size=size)
    theoretical_mean = 0.0
    p = confidence ** (1 / max_draws)
    z = norm.ppf(p)
    lower, upper = -z, z
else:
    sample = lambda size: rng.exponential(scale=1.0, size=size)
    theoretical_mean = 1.0
    p = confidence ** (1 / max_draws)
    upper = -np.log(1 - p)
    lower = 0

# Rescale for Sum or Mean
if mode == "Sum":
    x_range = [lower * 5, upper * 5]
else:
    x_range = [lower, upper]

# --- Data ---
n = 5
X_vals = sample(n_samples)
Xi_vals = sample((n_samples, n))
fiveX_vals = 5 * X_vals
sumXi_vals = Xi_vals.sum(axis=1)
meanXi_vals = sumXi_vals / n

if mode == "Sum":
    left_vals = fiveX_vals
    right_vals = sumXi_vals
    left_label = "5X"
    right_label = "ΣXᵢ"
else:
    left_vals = X_vals
    right_vals = meanXi_vals
    left_label = "X"
    right_label = "mean(Xᵢ)"

# --- Histogram bins ---
bin_edges = np.linspace(x_range[0], x_range[1], 51)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# --- Build animation frames ---
frames = []
for i in range(n_samples):
    xi_vals = Xi_vals[i]
    x_val = X_vals[i]
    mean_xi = np.mean(xi_vals)

    left_hist = left_vals[:i+1]
    right_hist = right_vals[:i+1]
    counts_left, _ = np.histogram(left_hist, bins=bin_edges)
    counts_right, _ = np.histogram(right_hist, bins=bin_edges)

    mean_left = np.mean(left_hist)
    var_left = np.var(left_hist)
    mean_right = np.mean(right_hist)
    var_right = np.var(right_hist)


        # Add memory trail lines
    trail_lines = []

    # Blue lines for previous X values
    for prev_x in X_vals[:i]:
        trail_lines.append(go.Scatter(
            x=[-0.5, 5.5], y=[prev_x]*2,
            mode="lines",
            line=dict(color="blue", width=1),
            opacity=trail_opacity,
            showlegend=False,
            xaxis="x1", yaxis="y1"
        ))

    # Orange lines for previous mean(Xᵢ) values
    for prev_mean_xi in Xi_vals[:i].mean(axis=1):
        trail_lines.append(go.Scatter(
            x=[-0.5, 5.5], y=[prev_mean_xi]*2,
            mode="lines",
            line=dict(color="orange", width=1),
            opacity=trail_opacity,
            showlegend=False,
            xaxis="x1", yaxis="y1"
        ))


    frames.append(go.Frame(
        name=str(i + 1),
        data=[
            go.Scatter(x=[0], y=[x_val], mode="markers", name="X",
                       marker=dict(color="blue", size=10), xaxis="x1", yaxis="y1"),
            go.Scatter(x=list(range(1, n + 1)), y=xi_vals, mode="markers", name="X₁...X₅",
                       marker=dict(color="orange", size=10), xaxis="x1", yaxis="y1"),
            go.Scatter(x=[-0.5, 5.5], y=[x_val]*2, mode="lines", name="mean(X)",
                       line=dict(color="blue", dash="dash"), xaxis="x1", yaxis="y1"),
            go.Scatter(x=[-0.5, 5.5], y=[mean_xi]*2, mode="lines", name="mean(Xᵢ)",
                       line=dict(color="orange", dash="dash"), xaxis="x1", yaxis="y1"),
            go.Scatter(x=[-0.5, 5.5], y=[theoretical_mean]*2, mode="lines", name="E[X]",
                       line=dict(color="red", width=2), xaxis="x1", yaxis="y1"),
            go.Bar(x=bin_centers, y=counts_left, name=left_label,
                   marker_color="blue", opacity=0.6, xaxis="x2", yaxis="y2"),
            go.Bar(x=bin_centers, y=counts_right, name=right_label,
                   marker_color="green", opacity=0.6, xaxis="x2", yaxis="y2"),
        ] + trail_lines,
        layout=go.Layout(
            annotations=[
                dict(
                    text=f"Sample #{i + 1} — {left_label}: Mean={mean_left:.3f}, Var={var_left:.3f} | "
                         f"{right_label}: Mean={mean_right:.3f}, Var={var_right:.3f}",
                    x=0.5, y=-0.115, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13)
                )
            ]
        )
    ))

# --- Build figure ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Current Sample: X and Xᵢ", f"Distributions of {left_label} and {right_label}"),
    column_widths=[0.4, 0.6]
)

# Empty base traces
fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="X", marker=dict(color="blue", size=10)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="X₁...X₅", marker=dict(color="orange", size=10)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="mean(X)", line=dict(color="blue", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="mean(Xᵢ)", line=dict(color="orange", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="E[X]", line=dict(color="red", width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=[], y=[], name=left_label, marker_color="blue", opacity=0.6), row=1, col=2)
fig.add_trace(go.Bar(x=[], y=[], name=right_label, marker_color="green", opacity=0.6), row=1, col=2)
fig.add_trace(go.Scatter(
    x=[], y=[], mode="lines", line=dict(color="blue", width=1),
    opacity=0.1, showlegend=False, xaxis="x1", yaxis="y1"
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[], y=[], mode="lines", line=dict(color="orange", width=1),
    opacity=0.1, showlegend=False, xaxis="x1", yaxis="y1"
), row=1, col=1)


# Precompute histogram to get max bin count
counts_right, _ = np.histogram(right_vals, bins=bin_edges)
max_y = int(np.max(counts_right) * 1.1)  # 10% headroom

fig.update_layout(
    height=650,
    margin=dict(b=150),
    title=f"{n}-sample decomposition of {dist}",
    barmode="overlay",
    xaxis1=dict(title="Index (0=X, 1–5=Xᵢ)", range=[-0.5, 5.5]),
    yaxis1=dict(title="Value", range=[lower, upper]),
    xaxis2=dict(title="Value", range=x_range),
    yaxis2=dict(title="Count", range=[0, max_y]),
    legend=dict(orientation="h", x=0.2, y=-0.35),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(
            label="▶ Play",
            method="animate",
            args=[None, {
                "frame": {"duration": frame_delay, "redraw": True},
                "fromcurrent": True
            }]
        )]
    )],
    sliders=[dict(
        steps=[dict(
            method="animate",
            args=[[str(i + 1)], {"frame": {"duration": frame_delay, "redraw": True}, "mode": "immediate"}],
            label=str(i + 1)
        ) for i in range(n_samples)],
        currentvalue=dict(prefix="Sample #: ")
    )]
)

fig.frames = frames

# --- Display plot ---
st.plotly_chart(fig, use_container_width=True)
