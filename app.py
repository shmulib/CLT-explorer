#ORIGINAL CODE:




import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Animated Visualization: Compare 5X vs ΣXᵢ or X vs mean(Xᵢ)")

# --- Controls ---
dist = st.sidebar.selectbox("Distribution", ["Uniform(0,1)", "Normal(0,1)", "Exponential(λ=1)"])
mode = st.sidebar.radio("Plot mode", ["Sum", "Mean"], index=0)
n_samples = st.sidebar.slider("Number of samples", 10, 500, 100, step=10)
fps = st.sidebar.slider("Animation speed", 1, 7, 2, step=1)
frame_delay = int(1000 / (0.5*2**(fps-1)))  # convert to ms/frame for Plotly

confidence_slider = st.sidebar.slider("Probability that all samples are visible in plot", min_value=0.95, max_value=0.99, value=0.99, step=0.001, format="%.3f")
trail_opacity = 0.65 #= st.slider(
#     "Memory trail opacity for previous X and mean(Xᵢ)",
#     min_value=0.0, max_value=1.0,
#     value=0.1, step=0.05
# )




# st.sidebar.markdown("### Subplot Title Positioning")

# # Left title sliders
# x_left_title = st.sidebar.slider("Left Title X Position", 0.0, 1.0, 0.15, 0.01)
# y_left_title = st.sidebar.slider("Left Title Y Position", 0.9, 1.1, 1.045, 0.005)

# # Right title sliders
# x_right_title = st.sidebar.slider("Right Title X Position", 0.0, 1.0, 0.8, 0.01)
# y_right_title = st.sidebar.slider("Right Title Y Position", 0.9, 1.1, 1.02, 0.005)


# --- Distribution Setup ---
rng = np.random.default_rng(seed=42)
max_draws = n_samples  # max 500 samples, each with X and 5 Xᵢ
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




from scipy.stats import norm, expon

# Based on max sample size (e.g. 500)
p_95 = 0.95
p_9999 = 0.9999
sample_count = n_samples  # or n_samples if you want to dynamically adjust

# Convert confidence levels for max of n samples
p95_single = p_95 ** (1 / sample_count)
p9999_single = p_9999 ** (1 / sample_count)

if dist == "Uniform(0,1)":
    allow_manual = False
    default_ymax = 1.0
    max_ymax = 1.0
elif dist == "Normal(0,1)":
    allow_manual = True
    default_ymax = norm.ppf(p95_single)
    max_ymax = norm.ppf(p9999_single)
elif dist == "Exponential(λ=1)":
    allow_manual = True
    default_ymax = expon.ppf(p95_single)
    max_ymax = expon.ppf(p9999_single)

if allow_manual:
    st.sidebar.markdown("### Left Plot Y-Axis Max Control")
    use_manual_ymax = st.sidebar.checkbox("Manually set left y-axis max")
    
    if use_manual_ymax:
        y_max_left = st.sidebar.slider(
            "Left y-axis max",
            min_value=0.0,
            max_value=float(max_ymax),
            value=float(default_ymax),
            step=0.1
        )
    else:
        y_max_left = upper*1.2
        if dist == "Normal(0,1)":
            lower = -upper*1.2  # this comes from confidence_slider (existing logic)
else:
    y_max_left = upper  # always auto for Uniform(0,1)



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





x_memory_x = [-0.6] * n_samples
x_memory_y = X_vals



mean_memory_x = [5.6] * n_samples
mean_Xi_vals = Xi_vals.mean(axis=1)
mean_memory_y = mean_Xi_vals


persistent_titles = [
    dict(
        text="Current Sample: X and Xᵢ",
        x=0.18, xref="paper", y=1.02, yref="paper",
        showarrow=False, font=dict(size=14)
    ),
    dict(
        text=f"Distributions of {left_label} and {right_label}",
        x=0.80, xref="paper", y=1.10, yref="paper",
        showarrow=False, font=dict(size=14)
    )
]


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

    trail_markers = [
    go.Scatter(
        x=x_memory_x[:i], y=x_memory_y[:i],
        mode="markers",
        marker=dict(color="blue", opacity=trail_opacity, size=4),
        showlegend=False,
        xaxis="x1", yaxis="y1"
    ),
    go.Scatter(
        x=mean_memory_x[:i], y=mean_memory_y[:i],
        mode="markers",
        marker=dict(color="orange", opacity=trail_opacity, size=4),
        showlegend=False,
        xaxis="x1", yaxis="y1"
    )
]


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
        ] + trail_markers,
        layout=go.Layout(
            annotations=[
                dict(
                    text=f"Sample #{i + 1} | {left_label}: Mean={mean_left:.3f}, Var={var_left:.3f} | "
                         f"{right_label}: Mean={mean_right:.3f}, Var={var_right:.3f}",
                    x=0.5, y=-0.25, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=13)
                )
            ] + persistent_titles
        )
    ))

# --- Build figure ---
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.4, 0.6],
    column_titles=["Current Sample: X and Xᵢ", f"Distributions of {left_label} and {right_label}"])



# Empty base traces
fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="X", marker=dict(color="blue", size=10)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="markers", name="X₁...X₅", marker=dict(color="orange", size=10)), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="mean(X)", line=dict(color="blue", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="mean(Xᵢ)", line=dict(color="orange", dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="E[X]", line=dict(color="red", width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=[], y=[], name=left_label, marker_color="blue", opacity=0.6), row=1, col=2)
fig.add_trace(go.Bar(x=[], y=[], name=right_label, marker_color="green", opacity=0.6), row=1, col=2)

fig.add_trace(go.Scatter(
    x=[], y=[], mode="markers",
    marker=dict(color="blue", size=4),
    name="Memory X",
    showlegend=False
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=[], y=[], mode="markers",
    marker=dict(color="orange", size=4),
    name="Memory mean(Xᵢ)",
    showlegend=False
), row=1, col=1)


# Precompute histogram to get max bin count
counts_right, _ = np.histogram(right_vals, bins=bin_edges)
max_y = int(np.max(counts_right) * 1.2)  # 10% headroom

fig.update_layout(
    height=650,
    margin=dict(t=200, b=150),
    title=dict(
    text=f"{n}-sample decomposition of {dist}",
    x=0.5,
    xanchor="center",
    xref="container" ), # ensures it's centered across full layout)
    barmode="overlay",
    xaxis1=dict(title="Index (0=X, 1–5=Xᵢ)", range=[-1, 6]),
    yaxis1=dict(title="Value", range=[lower, y_max_left]),
    xaxis2=dict(title="Value", range=x_range),
    yaxis2=dict(title="Count", range=[0, max_y]),
    legend=dict(orientation="h", x=0.2, y=-0.35),
updatemenus=[dict(
    type="buttons",
    direction="down",
    showactive=False,
    x=-0.15,  # left of left plot
    y=1.0,
    xanchor="left",
    yanchor="top",
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    font=dict(color="black", size=13),
    buttons=[
        dict(
            label="▶ Play",
            method="animate",
            args=[None, {
                "frame": {"duration": frame_delay, "redraw": True},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": 0}
            }]
        ),
        dict(
            label="⏸ Pause",
            method="animate",
            args=[[None], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}
            }]
        ),
        dict(
            label="⏮ Reset",
            method="animate",
            args=[["1"], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": True},
                "transition": {"duration": 0},
                "slider": {"active": 0}
            }]
        )
    ]
)],
sliders=[dict(
    steps=[
        dict(
            method="animate",
            args=[[str(i + 1)], {
                "mode": "immediate",
                "frame": {"duration": frame_delay, "redraw": True},
                "transition": {"duration": 0}
            }],
            label=str(i + 1)
        )
        for i in range(n_samples)
    ],
    currentvalue=dict(
        prefix="Sample #: ",
        font=dict(size=14, color="white")
    ),
    transition=dict(duration=0),
    x=0.5,
    y=-0.35,
    xanchor="center"
)]


    # sliders=[dict(
    #     steps=[dict(
    #         method="animate",
    #         args=[[str(i + 1)], {"frame": {"duration": frame_delay, "redraw": True}, "mode": "immediate"}],
    #         label=str(i + 1)
    #     ) for i in range(n_samples)],
    #     currentvalue=dict(prefix="Sample #: ")
    # )]
)

fig.frames = frames

# --- Display plot ---
st.plotly_chart(fig, use_container_width=True)
