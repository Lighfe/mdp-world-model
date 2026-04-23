import marimo

__generated_with = "0.23.1"
app = marimo.App(width="full")

with app.setup:
    import subprocess
    import sys
    import os

    # Molab / cloud: clone the repo and add it to the path.
    # Locally (run from project root) neural_networks/ and data_generation/ already
    # exist in the cwd, so nothing happens.
    if not (os.path.exists("neural_networks") and os.path.exists("data_generation")):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "anywidget>=0.9.0", "sqlalchemy", "--quiet"],
            check=True,
        )
        if not os.path.exists("mdp-world-model"):
            subprocess.run(
                ["git", "clone", "https://github.com/Lighfe/mdp-world-model.git"],
                check=True,
            )
        _root = os.path.abspath("mdp-world-model")
        if _root not in sys.path:
            sys.path.insert(0, _root)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import threading
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    except NameError:
        pass
    import base64
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    import io
    import tempfile
    import yaml
    import pandas as pd

    return (
        base64,
        io,
        matplotlib,
        np,
        pd,
        plt,
        tempfile,
        threading,
        to_rgba,
        yaml,
    )


@app.cell
def _():
    from data_generation.models.saddle_system import MultiSaddleSystem
    from data_generation.models.general_ode_solver import GeneralODENumericalSolver
    from data_generation.simulations.grid import Grid, logistic_transformation
    from data_generation.simulations.simulator import Simulator
    from neural_networks.train_drm_simple import train_drm_simple
    from neural_networks.drm import DiscreteRepresentationsModel
    from neural_networks.drm_viz import visualize_final_state_assignments
    from neural_networks.system_registry import SystemType, get_visualization_bounds

    return (
        DiscreteRepresentationsModel,
        GeneralODENumericalSolver,
        Grid,
        MultiSaddleSystem,
        Simulator,
        SystemType,
        get_visualization_bounds,
        logistic_transformation,
        train_drm_simple,
        visualize_final_state_assignments,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # DRM Playground

    Many real-world systems — climate tipping points, economic transitions,
    ecological regime shifts — are described by continuous differential equations
    but reasoned about as discrete state machines. The **Discrete Representation
    Model (DRM)** is a proof-of-concept that this gap can be closed via
    self-supervised learning.

    This notebook interactively walks through the full pipeline on a 2D toy system:

    - Configure a saddle system (two saddle points, two control actions)
    - Sample transition data from the system and store it in a local SQLite database
    - Train the DRM to discover discrete states and transition dynamics
    - Visualise the resulting state assignments

    ## Architecture

    The DRM learns a finite discrete MDP from continuous dynamical system transition data
    $(x, c, y)$ — current state, control, next state; self-supervised with **no state labels required**.
    """)
    return


@app.cell(hide_code=True)
def _(base64, mo):
    _candidates = [
        "docs/figures/DRM_architecture.jpeg",
        "mdp-world-model/docs/figures/DRM_architecture.jpeg",
    ]
    _arch_bytes = None
    for _p in _candidates:
        if os.path.exists(_p):
            with open(_p, "rb") as _f:
                _arch_bytes = _f.read()
            break
    if _arch_bytes is None:
        import urllib.request
        _url = "https://raw.githubusercontent.com/Lighfe/mdp-world-model/main/docs/figures/DRM_architecture.jpeg"
        with urllib.request.urlopen(_url) as _r:
            _arch_bytes = _r.read()
    _arch_img = mo.Html(
        f'<img src="data:image/jpeg;base64,{base64.b64encode(_arch_bytes).decode()}"'
        ' style="width:700px;max-width:100%">'
    )
    _table = mo.md(r"""
    | Component | Input → Output | Role |
    |---|---|---|
    | **Encoder** | $x \to s_x$ | Encodes continuous observation into discrete state |
    | **Target Encoder** | $y \to s_y$ | Stable training target; updated via EMA |
    | **Predictor** | $(s_x, c) \to \hat{P}(s_y)$ | Learned MDP transition function |
    | **Value Network** | $s_i \to v_i$ | Prevents state collapse during training |
    """)
    _formula = mo.md(r"""$\displaystyle \mathcal{L} = \mathcal{L}_\text{state}(s_y,\,\hat{P}(s_y)) + w_v\,\mathcal{L}_\text{value}(v_\text{true}, v_\text{pred}) + w_e\,\mathcal{L}_\text{entropy}(s_x)$""")
    mo.vstack(
        [
            mo.hstack(
                [_arch_img, mo.vstack([_table, _formula], gap="5rem")],
                gap="2rem",
                align="start",
            )
        ],
        gap="1rem",
    )
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(
        "---\n"
        "**Workflow:** Configure saddle system → generate dataset → train DRM → view state assignments  \n"
        "*Developed as a master's thesis at "
        "[TU Berlin](https://www.tu.berlin/) in cooperation with the "
        "[Potsdam Institute for Climate Impact Research (PIK)](https://www.pik-potsdam.de/). "
        "Source code: [github.com/Lighfe/mdp-world-model](https://github.com/Lighfe/mdp-world-model)*   \n"
        "The Dynamical Systems and Data Sampling approach was developed in coordination with Karolin Stiller."
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Saddle System

    The DRM should be able to learn **any** dynamical system.
    To test this we designed the Saddle System as a simple way to create customizable 2D dynamics with interesting structure (saddle points, stable/unstable manifolds) that are still easy to visualize and (humanly) understand.
    The system contains two saddle points, each with a stable manifold that divides space into two halfspaces. Where the two stable manifolds intersect, they
    partition the observation space into four regions. As this partition appears to be the most straight-
    forward classification of observations into discrete states, we define these four regions as the ground
    truth states for this system.

    Each control action is defined by:
    - A **saddle point** $x^* \in \mathbb{R}^2$ (drag the circles below)
    - A **stable manifold angle** — direction along which trajectories converge
    - Shared **Lyapunov exponents** $\lambda_1 > 0$ (unstable) and $\lambda_2 < 0$ (stable)

    The dynamics are $\dot{x} = A\,(x - x^*)$ where $A = V\,\text{diag}(\lambda_1,\lambda_2)\,V^{-1}$.

    ---
    """)
    return


@app.cell
def _():
    import anywidget
    from traitlets.traitlets import List as TList

    class SaddleDragWidget(anywidget.AnyWidget):
        __version__ = "0.1.0"
        _esm = """
    function render({ model, el }) {
      const W = 520, H = 520, M = 56;
      const PW = W - 2*M, PH = H - 2*M;
      const COLORS = ['#44AA99', '#882255'];
      const R = 9;

      el.style.cssText = 'font-family:system-ui,sans-serif;display:inline-block;user-select:none;';
      const canvas = document.createElement('canvas');
      canvas.width = W; canvas.height = H;
      canvas.style.cssText = 'display:block;cursor:default;border:1px solid #ddd;border-radius:8px;background:#fff;';
      el.append(canvas);
      const ctx = canvas.getContext('2d');

      function toCanvas(px, py) {
    const [x0, x1] = model.get('x_range');
    const [y0, y1] = model.get('y_range');
    return [M + (px - x0) / (x1 - x0) * PW, H - M - (py - y0) / (y1 - y0) * PH];
      }
      function toPlot(cx, cy) {
    const [x0, x1] = model.get('x_range');
    const [y0, y1] = model.get('y_range');
    return [x0 + (cx - M) / PW * (x1 - x0), y0 + (H - M - cy) / PH * (y1 - y0)];
      }

      function draw() {
    ctx.clearRect(0, 0, W, H);
    const xr = model.get('x_range');
    const yr = model.get('y_range');
    const pts = model.get('saddle_points');
    const angles = model.get('angles_deg');

    ctx.fillStyle = '#fafafa'; ctx.fillRect(M, M, PW, PH);

    const rawStep = (xr[1] - xr[0]) / 5;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const niceStep = [1, 2, 5].map(n => n * mag).find(s => s >= rawStep) || mag;

    ctx.strokeStyle = '#eee'; ctx.lineWidth = 1;
    for (let v = Math.ceil(xr[0] / niceStep) * niceStep; v <= xr[1] + 1e-9; v += niceStep) {
      const [cx] = toCanvas(v, 0);
      ctx.beginPath(); ctx.moveTo(cx, M); ctx.lineTo(cx, M + PH); ctx.stroke();
    }
    for (let v = Math.ceil(yr[0] / niceStep) * niceStep; v <= yr[1] + 1e-9; v += niceStep) {
      const [, cy] = toCanvas(0, v);
      ctx.beginPath(); ctx.moveTo(M, cy); ctx.lineTo(M + PW, cy); ctx.stroke();
    }

    const [ox, oy] = toCanvas(0, 0);
    ctx.strokeStyle = '#bbb'; ctx.lineWidth = 1.5;
    if (oy >= M && oy <= M + PH) { ctx.beginPath(); ctx.moveTo(M, oy); ctx.lineTo(M + PW, oy); ctx.stroke(); }
    if (ox >= M && ox <= M + PW) { ctx.beginPath(); ctx.moveTo(ox, M); ctx.lineTo(ox, M + PH); ctx.stroke(); }

    ctx.fillStyle = '#666'; ctx.strokeStyle = '#aaa'; ctx.lineWidth = 1;
    ctx.font = '11px system-ui';
    ctx.textAlign = 'center';
    for (let v = Math.ceil(xr[0] / niceStep) * niceStep; v <= xr[1] + 1e-9; v += niceStep) {
      if (Math.abs(v) < 1e-9) continue;
      const [cx] = toCanvas(v, 0);
      const yTick = Math.min(oy + 4, M + PH - 1);
      ctx.beginPath(); ctx.moveTo(cx, yTick - 4); ctx.lineTo(cx, yTick + 4); ctx.stroke();
      ctx.fillText(Number.isInteger(v) ? v : v.toFixed(1), cx, Math.min(oy + 16, H - 6));
    }
    ctx.textAlign = 'right';
    for (let v = Math.ceil(yr[0] / niceStep) * niceStep; v <= yr[1] + 1e-9; v += niceStep) {
      if (Math.abs(v) < 1e-9) continue;
      const [, cy] = toCanvas(0, v);
      const xTick = Math.max(ox, M + 1);
      ctx.beginPath(); ctx.moveTo(xTick - 4, cy); ctx.lineTo(xTick + 4, cy); ctx.stroke();
      ctx.fillText(Number.isInteger(v) ? v : v.toFixed(1), Math.max(ox - 6, M + 24), cy + 4);
    }

    ctx.font = 'bold 12px system-ui'; ctx.fillStyle = '#444'; ctx.textAlign = 'center';
    ctx.fillText('x\u2081', M + PW / 2, H - 4);
    ctx.save(); ctx.translate(13, M + PH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('x\u2082', 0, 0); ctx.restore();

    // Stable manifold lines (dashed), read angles from model
    pts.forEach((pt, i) => {
      const aRad = angles[i] * Math.PI / 180;
      const ca = Math.cos(aRad), sa = Math.sin(aRad);
      const tv = [];
      if (Math.abs(ca) > 1e-10) { tv.push((xr[0] - pt[0]) / ca); tv.push((xr[1] - pt[0]) / ca); }
      if (Math.abs(sa) > 1e-10) { tv.push((yr[0] - pt[1]) / sa); tv.push((yr[1] - pt[1]) / sa); }
      const inter = tv.map(t => [pt[0] + t * ca, pt[1] + t * sa])
        .filter(([x, y]) => x >= xr[0] - 1e-9 && x <= xr[1] + 1e-9 && y >= yr[0] - 1e-9 && y <= yr[1] + 1e-9);
      if (inter.length >= 2) {
        const [ax0, ay0] = toCanvas(inter[0][0], inter[0][1]);
        const [ax1, ay1] = toCanvas(inter[1][0], inter[1][1]);
        ctx.beginPath(); ctx.setLineDash([7, 4]);
        ctx.strokeStyle = COLORS[i % COLORS.length] + 'bb'; ctx.lineWidth = 2;
        ctx.moveTo(ax0, ay0); ctx.lineTo(ax1, ay1); ctx.stroke(); ctx.setLineDash([]);
      }
    });

    ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1; ctx.strokeRect(M, M, PW, PH);

    pts.forEach((pt, i) => {
      const [cx, cy] = toCanvas(pt[0], pt[1]);
      ctx.beginPath(); ctx.arc(cx, cy, R + 4, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[i % COLORS.length] + '28'; ctx.fill();
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 2 * Math.PI);
      ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fill();
      ctx.strokeStyle = 'white'; ctx.lineWidth = 2.5; ctx.stroke();
      ctx.font = '10px system-ui'; ctx.fillStyle = '#333'; ctx.textAlign = 'left';
      ctx.fillText(`S${i}  (${pt[0].toFixed(1)}, ${pt[1].toFixed(1)})`, cx + R + 5, cy - 1);
    });
      }

      let dragging = null;
      function getPos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height];
      }
      canvas.addEventListener('mousedown', e => {
    const [mx, my] = getPos(e);
    model.get('saddle_points').forEach((pt, i) => {
      const [cx, cy] = toCanvas(pt[0], pt[1]);
      if (Math.hypot(mx - cx, my - cy) <= R + 5) { dragging = i; canvas.style.cursor = 'grabbing'; }
    });
      });
      canvas.addEventListener('mousemove', e => {
    const [mx, my] = getPos(e);
    if (dragging !== null) {
      const cx = Math.max(M, Math.min(M + PW, mx));
      const cy = Math.max(M, Math.min(M + PH, my));
      const [px, py] = toPlot(cx, cy);
      const pts = model.get('saddle_points').map(p => [...p]);
      pts[dragging] = [+px.toFixed(2), +py.toFixed(2)];
      model.set('saddle_points', pts);
      draw();
    } else {
      const onPt = model.get('saddle_points').some(pt => {
        const [cx, cy] = toCanvas(pt[0], pt[1]);
        return Math.hypot(mx - cx, my - cy) <= R + 5;
      });
      canvas.style.cursor = onPt ? 'grab' : 'default';
    }
      });
      canvas.addEventListener('mouseup', () => {
    if (dragging !== null) model.save_changes();
    dragging = null; canvas.style.cursor = 'default';
      });
      canvas.addEventListener('mouseleave', () => {
    if (dragging !== null) model.save_changes();
    dragging = null;
      });
      model.on('change:saddle_points', draw);
      model.on('change:angles_deg', draw);
      model.on('change:x_range', draw);
      model.on('change:y_range', draw);
      draw();
    }
    export default { render };
    """
        _css = ""
        saddle_points = TList([[-1.0, 2.0], [1.5, 0.0]]).tag(sync=True)
        angles_deg = TList([120.0, 0.0]).tag(sync=True)
        x_range = TList([-4.0, 4.0]).tag(sync=True)
        y_range = TList([-4.0, 4.0]).tag(sync=True)

    return (SaddleDragWidget,)


@app.cell
def _(SaddleDragWidget, mo):
    drag_raw = SaddleDragWidget()
    drag_widget = mo.ui.anywidget(drag_raw)
    return drag_raw, drag_widget


@app.cell
def _(mo):
    angle0_s = mo.ui.slider(
        start=0, stop=360, step=5, value=135,
        label="S0 stable manifold angle",
        full_width=True,
    )
    angle1_s = mo.ui.slider(
        start=0, stop=360, step=5, value=0,
        label="S1 stable manifold angle",
        full_width=True,
    )
    lambda1_s = mo.ui.slider(
        start=0.5, stop=2.0, step=0.1, value=1.0,
        label="λ₁  unstable exponent (> 0)",
    )
    lambda2_s = mo.ui.slider(
        start=-2.0, stop=-0.5, step=0.1, value=-1.0,
        label="λ₂  stable exponent (< 0)",
    )
    return angle0_s, angle1_s, lambda1_s, lambda2_s


@app.cell
def _(angle0_s, angle1_s, drag_raw):
    drag_raw.angles_deg = [float(angle0_s.value), float(angle1_s.value)]
    return


@app.cell
def _(
    GeneralODENumericalSolver,
    MultiSaddleSystem,
    angle0_s,
    angle1_s,
    drag_widget,
    io,
    lambda1_s,
    lambda2_s,
    matplotlib,
    np,
    plt,
    to_rgba,
):
    saddle_pts = [np.array(p) for p in drag_widget.value["saddle_points"]]
    angles = [float(angle0_s.value), float(angle1_s.value)]
    lam1 = lambda1_s.value
    lam2 = lambda2_s.value

    x_range = (-5.0, 5.0)
    y_range = (-5.0, 5.0)
    res = 51
    COLORS = ["#44AA99", "#882255"]

    model_sys = MultiSaddleSystem(
        k=2, saddle_points=saddle_pts, angles=angles, lambda1=lam1, lambda2=lam2
    )
    solver = GeneralODENumericalSolver(model_sys)

    X1lin = np.linspace(x_range[0], x_range[1], res)
    X2lin = np.linspace(y_range[0], y_range[1], res)
    X1, X2 = np.meshgrid(X1lin, X2lin)
    X_flat = np.column_stack([X1.ravel(), X2.ravel()])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    for i in range(2):
        derivs = solver.get_derivative(X_flat, i)
        U = derivs[:, 0].reshape(res, res)
        V = derivs[:, 1].reshape(res, res)
        mag = np.sqrt(U**2 + V**2)
        max_mag = np.percentile(mag, 95) if mag.max() > 0 else 1.0
        intensity = 0.5 + 0.5 * np.sqrt(np.clip(mag / max_mag, 0, 1))
        base = to_rgba(COLORS[i])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", [(1, 1, 1, 0.3), (*base[:3], 1.0 if i == 0 else 0.9)]
        )
        strm = ax.streamplot(
            X1lin, X2lin, U, V,
            color=intensity, cmap=cmap,
            linewidth=1.2, density=0.8,
            arrowsize=1.5, arrowstyle="fancy",
        )
        if i == 1:
            strm.lines.remove()
            for seg in strm.lines.get_segments():
                ax.plot(seg[:, 0], seg[:, 1], ":", color=COLORS[i], lw=1.6, alpha=0.8)

    for i, (pt, adeg) in enumerate(zip(saddle_pts, angles)):
        ax.plot(
            pt[0], pt[1], "o",
            markerfacecolor=COLORS[i], markeredgecolor="white",
            markeredgewidth=2, markersize=10, zorder=10,
        )
        arad = np.radians(adeg)
        px, py = float(pt[0]), float(pt[1])
        ca, sa = np.cos(arad), np.sin(arad)
        tv = []
        if abs(ca) > 1e-10:
            tv += [(x_range[0] - px) / ca, (x_range[1] - px) / ca]
        if abs(sa) > 1e-10:
            tv += [(y_range[0] - py) / sa, (y_range[1] - py) / sa]
        isecs = [
            (px + t * ca, py + t * sa) for t in tv
            if x_range[0] <= px + t * ca <= x_range[1]
            and y_range[0] <= py + t * sa <= y_range[1]
        ]
        if len(isecs) >= 2:
            ax.plot(
                [isecs[0][0], isecs[1][0]], [isecs[0][1], isecs[1][1]],
                color=COLORS[i], linestyle="--", linewidth=1.8,
                alpha=0.9 if i == 0 else 0.7, zorder=1,
            )

    ax.set_xlabel(r"$\mathbf{x_1}$", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"$\mathbf{x_2}$", fontsize=13, fontweight="bold")
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    streamplot_img = buf.read()
    plt.close(fig)
    return (streamplot_img,)


@app.cell
def _(
    angle0_s,
    angle1_s,
    base64,
    drag_widget,
    lambda1_s,
    lambda2_s,
    mo,
    streamplot_img,
):
    pts = drag_widget.value["saddle_points"]
    angs = [float(angle0_s.value), float(angle1_s.value)]

    # ── D: Current Configuration table ─────────────────────────
    rows = [
        ["", "<span style='color:#44AA99;font-size:1.1em'>●</span> **S0**", "<span style='color:#882255;font-size:1.1em'>●</span> **S1**"],
        ["Position",        f"({pts[0][0]:.2f}, {pts[0][1]:.2f})", f"({pts[1][0]:.2f}, {pts[1][1]:.2f})"],
        ["Manifold angle",  f"{angs[0]:.0f}°",  f"{angs[1]:.0f}°"],
        ["λ₁ (unstable)",   f"{lambda1_s.value:.1f}", f"{lambda1_s.value:.1f}"],
        ["λ₂ (stable)",     f"{lambda2_s.value:.1f}", f"{lambda2_s.value:.1f}"],
    ]
    _hdr  = "| " + " | ".join(rows[0]) + " |"
    _sep  = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    _body = "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])
    config_md = mo.md(f"### Current Configuration\n\n{_hdr}\n{_sep}\n{_body}")

    # ── A: drag widget panel ────────────────────────────────────
    panel_A = mo.vstack(
        [
            mo.md("## Dynamics & Controls"),
            mo.md(
                "Design your custom dynamics.  \n"
                "**Drag** the circles to reposition saddle points.  \n"
                "Change the **angles** of the stable manifolds (dashed-lines) and **Lyapunov exponents** with the sliders below."
            ),
            drag_widget,
        ],
        gap="0.5rem",
    )

    # ── B: streamplot panel ─────────────────────────────────────
    panel_B = mo.vstack(
        [
            mo.md("## Streamplot Visualization"),
            mo.md(
                "The Streamplots show the dynamics of the system under each control action.  \n"
                "**Teal (solid)** — action 0 &nbsp;·&nbsp; **Purple (dotted)** — action 1  \n"
                "Dashed lines show stable manifolds. Opacity indicates speed of dynamics."
            ),
            mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(streamplot_img).decode()}" style="width:600px;max-width:100%">'),
        ],
        gap="0.5rem",
    )

    # ── C: slider panel ─────────────────────────────────────────
    _css = mo.Html("""<style>
      [data-slider-group="teal"] input[type=range] { accent-color: #44AA99; }
      [data-slider-group="purple"] input[type=range] { accent-color: #882255; }
    </style>""")
    _teal = mo.Html(
        f"<div data-slider-group='teal'>{mo.vstack([angle0_s, angle1_s], gap='0.4rem')._repr_html_()}</div>"
    )
    _purple = mo.Html(
        f"<div data-slider-group='purple'>{mo.vstack([lambda1_s, lambda2_s], gap='0.4rem')._repr_html_()}</div>"
    )
    panel_C = mo.vstack(
        [
            _css,
            mo.md("**Stable Manifold Angles**"),
            _teal,
            mo.md("**Lyapunov Exponents** — shared across both saddles"),
            _purple,
        ],
        gap="0.4rem",
    )

    # ── D: config table wrapped in vstack for top-alignment ─────
    panel_D = mo.vstack([config_md], gap="0")

    mo.vstack(
        [
            mo.hstack([panel_A, panel_B], gap="2rem", align="start"),
            mo.hstack(
                [panel_C, panel_D],
                gap="4rem",
                align="start",
                justify="start",
            ),
        ],
        gap="2rem",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    db_path_input = mo.ui.text(
        value="saddle_notebook.db", label="Output database"
    )
    generate_btn = mo.ui.run_button(label="Generate Dataset")
    return db_path_input, generate_btn


@app.cell(hide_code=True)
def _(
    GeneralODENumericalSolver,
    Grid,
    MultiSaddleSystem,
    Simulator,
    angle0_s,
    angle1_s,
    db_path_input,
    drag_widget,
    generate_btn,
    lambda1_s,
    lambda2_s,
    logistic_transformation,
    np,
    pd,
):
    _RESOLUTION = 20   # 20×20 grid
    _SAMPLES = 25      # avg samples per cell → 10 000 transitions per control
    _DELTA_T = 1.0
    sim_df = None
    sim_db_path = None
    if generate_btn.value:
        _saddle_pts = [np.array(p) for p in drag_widget.value["saddle_points"]]
        _angles = [float(angle0_s.value), float(angle1_s.value)]
        _model_sys = MultiSaddleSystem(
            k=2,
            saddle_points=_saddle_pts,
            angles=_angles,
            lambda1=lambda1_s.value,
            lambda2=lambda2_s.value,
        )
        _solver = GeneralODENumericalSolver(_model_sys)
        _bounds = [(float("-inf"), float("inf")), (float("-inf"), float("inf"))]
        _tf = [
            logistic_transformation({"k": 0.5, "x_0": 0.0}),
            logistic_transformation({"k": 0.5, "x_0": 0.0}),
        ]
        _grid = Grid(_bounds, [_RESOLUTION, _RESOLUTION], _tf)
        _sim = Simulator(_grid, _model_sys, _solver)
        _dfs = []
        for _ctrl in range(2):
            _dfs.append(
                _sim.simulate(
                    control=_ctrl,
                    delta_t=_DELTA_T,
                    avg_samples_per_cell=_SAMPLES,
                    num_steps=1,
                    save_result=True,
                )
            )
        # Both simulate() calls complete within the same second → identical
        # timestamp-based run_ids → UNIQUE constraint failure in configs table.
        # Suffix each config row with the control index to guarantee uniqueness.
        for _i, _idx in enumerate(_sim.configs.index):
            _sim.configs.at[_idx, "run_id"] = _sim.configs.at[_idx, "run_id"] + f"_c{_i}"
        sim_db_path = db_path_input.value
        os.makedirs(os.path.dirname(os.path.abspath(sim_db_path)), exist_ok=True)
        if os.path.exists(sim_db_path):
            os.remove(sim_db_path)
        _sim.store_results_to_sqlite(filename=sim_db_path)
        sim_df = pd.concat(_dfs, ignore_index=True)
    return sim_db_path, sim_df


@app.cell(hide_code=True)
def _(base64, db_path_input, generate_btn, io, mo, plt, sim_db_path, sim_df):
    # ── E: generation controls ──────────────────────────────────
    panel_E = mo.vstack(
        [
            mo.md("---\n## Data Generation"),
            mo.md(
                "Uses a numerical ODE solver to simulate the configured saddle system under "
                "both controls, sampling initial conditions from a uniform grid. Results are "
                "stored in a local SQLite database for use during training.  \n"
                "**Grid:** 20×20 &nbsp;·&nbsp; **Samples/cell:** 25 &nbsp;·&nbsp; "
                "**δt:** 1.0 &nbsp;·&nbsp; **~10 000 transitions per control**  \n"
                "Press **Generate Dataset** to simulate and preview the data on the right."
            ),
            db_path_input,
            generate_btn,
        ],
        gap="0.5rem",
    )

    # ── F: sample summary + dataset preview ─────────────────────
    if sim_df is not None:
        _n_traj = sim_df["trajectory_id"].nunique()
        _n0 = int((sim_df["c0"] == 0).sum())
        _n1 = int((sim_df["c0"] == 1).sum())
        _summary = mo.md(
            f"**Stored** → `{sim_db_path}`  \n"
            f"{len(sim_df):,} transitions &nbsp;·&nbsp; "
            f"Control 0: {_n0:,} &nbsp;·&nbsp; Control 1: {_n1:,}  \n\n"
            "Each row is a single $(x, c, y)$ tuple — current state, control, next state. "
            "**No saddle positions, manifold angles, or Lyapunov exponents are stored**: "
            "the DRM must learn meaningful dynamics from raw transitions alone.  \n"
            "*Table and plot show one random batch (128 transitions).*"
        )
        _cols = ["x0", "x1", "c0", "y0", "y1"]
        _preview = sim_df[_cols].sample(min(128, len(sim_df))).reset_index(drop=True)
        for _col in ["x0", "x1", "y0", "y1"]:
            _preview[_col] = _preview[_col].round(4)
        _COLORS = ["#44AA99", "#882255"]
        _samp = sim_df.sample(min(200, len(sim_df)))
        _fig, _ax = plt.subplots(figsize=(5.0, 5.0))
        for _c in [0, 1]:
            _mask = _samp["c0"] == _c
            _ax.scatter(_samp.loc[_mask, "x0"], _samp.loc[_mask, "x1"],
                        c=_COLORS[_c], s=15, alpha=0.5, label=f"Control {_c}")
        _ax.set_xlabel(r"$x_1$", fontsize=12)
        _ax.set_ylabel(r"$x_2$", fontsize=12)
        _ax.set_title("Initial conditions — denser near (0, 0)", fontsize=12)
        _ax.legend(markerscale=2)
        _ax.set_aspect("equal", adjustable="box")
        plt.tight_layout()
        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
        _buf.seek(0)
        plt.close(_fig)
        _scatter = mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(_buf.read()).decode()}" style="width:420px;max-width:100%">')
        panel_F = mo.vstack(
            [_summary, mo.ui.tabs({"Table": mo.ui.table(_preview), "Plot": _scatter})],
            gap="0.8rem",
        )
    else:
        panel_F = mo.vstack(
            [mo.md("_No data yet. Configure parameters and click **Generate Dataset**._")],
            gap="0",
        )

    mo.hstack([panel_E, panel_F], gap="3rem", align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    num_states_s = mo.ui.slider(
        3, 8, step=1, value=4, label="Number of discrete states",
    )
    epochs_s = mo.ui.slider(
        25, 100, step=25, value=50, label="Epochs",
    )
    train_btn = mo.ui.run_button(label="Train DRM")
    return epochs_s, num_states_s, train_btn


@app.cell(hide_code=True)
def _(
    DiscreteRepresentationsModel,
    db_path_input,
    epochs_s,
    mo,
    num_states_s,
    sim_df,
    tempfile,
    threading,
    train_btn,
    train_drm_simple,
    yaml,
):
    # num_states_s and epochs_s are intentionally NOT declared as dependencies.
    # They live in the module's global namespace (exported by the slider cell) and are
    # read at the moment train_btn fires. Declaring them as dependencies would cause this
    # cell to re-run — and reset training_state — every time a slider moves, which creates
    # a network-ordering bug on hosted marimo where the button-click message can arrive
    # before the slider-change message.
    training_state = {"history": {}, "model": None, "done": False, "error": None, "started": False}
    mo.stop(
        train_btn.value and sim_df is None,
        mo.md("**Generate a dataset first** — click Generate Dataset in the section above."),
    )
    if train_btn.value:
        _config = {
            "meta": {
                "db_path": os.path.abspath(db_path_input.value),
                "seed": 42,
                "output_dir": os.path.abspath("datasets/results/training"),
                "run_id": "notebook_run",
            },
            "data": {
                "system_type": "saddle_system",
                "val_size": 1000,
                "test_size": 1000,
            },
            "model": {
                "num_states": num_states_s.value,  # type: ignore[name-defined]
                "hidden_dim": 32,
                "predictor_type": "standard",
                "value_method": "angle",
                "use_target_encoder": True,
                "ema_decay": 0.66,
                "use_gumbel": True,
                "initial_temp": 1.5,
                "min_temp": 0.25,
                "encoder_init_method": "he",
            },
            "training": {
                "epochs": epochs_s.value,  # type: ignore[name-defined]
                "batch_size": 128,
                "checkpoint_every": epochs_s.value,  # type: ignore[name-defined]
                "optimizer_type": "adamw",
                "lr": 3.3e-4,
                "weight_decay": 0.017,
                "scheduler_type": "cosine",
                "min_lr": 3.3e-5,
                "use_warmup": True,
                "warmup_epochs": 10,
                "restart_period": 20,
                "restart_mult": 1,
            },
            "loss": {
                "state_loss_weight": 1.0,
                "value_loss_weight": 0.55,
                "state_loss_type": "kl_div",
                "value_loss_type": None,
                "use_entropy_reg": True,
                "entropy_weight": 0.6,
                "use_entropy_decay": True,
                "entropy_decay_proportion": 0.5,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as _f:
            yaml.dump(_config, _f)
            _tmp_path = _f.name

        def _run():
            try:
                _model, _ = train_drm_simple(
                    _tmp_path, shared_history=training_state["history"]
                )
                training_state["model"] = _model
            except Exception as _e:
                training_state["error"] = str(_e)
            finally:
                training_state["done"] = True

        training_state["started"] = True
        training_state["num_states"] = num_states_s.value  # type: ignore[name-defined]
        training_state["epochs"] = epochs_s.value  # type: ignore[name-defined]
        training_state["architecture_model"] = DiscreteRepresentationsModel(
            obs_dim=2, control_dim=2, value_dim=2,
            num_states=num_states_s.value,  # type: ignore[name-defined]
            hidden_dim=32, predictor_type="standard",
            use_gumbel=True, initial_temp=1.5, min_temp=0.25,
            use_target_encoder=True, ema_decay=0.66, value_activation="tanh",
        )
        threading.Thread(target=_run, daemon=True).start()
    return (training_state,)


@app.cell(hide_code=True)
def _(mo):
    drm_refresh = mo.ui.refresh(default_interval="30s")
    return (drm_refresh,)


@app.cell(hide_code=True)
def _(
    base64,
    drm_refresh,
    epochs_s,
    io,
    mo,
    num_states_s,
    plt,
    train_btn,
    training_state,
):
    # ── G: training controls ────────────────────────────────────
    _active = training_state.get("started")
    _cfg_line = (
        mo.md(
            f"**Running:** {training_state['num_states']} states · {training_state['epochs']} epochs"
        )
        if _active and "num_states" in training_state
        else mo.md("Set parameters, then click **Train DRM**.")
    )
    _model_items = [training_state["architecture_model"]] if training_state.get("architecture_model") is not None else []
    panel_G = mo.vstack(
        [
            mo.md("---\n## DRM Training"),
            mo.md(
                "The model trains self-supervised on the transitions you just generated — "
                "no state labels, no ground truth MDP. "
                "The encoder learns to assign observations to discrete states while the predictor "
                "learns how those states evolve under each control action."
            ),
            _cfg_line,
            num_states_s,
            epochs_s,
            train_btn,
            drm_refresh,
            *_model_items,
        ],
        gap="0.5rem",
    )

    # ── H: progressive loss curves ───────────────────────────────
    _hist = training_state["history"]
    _train_loss = _hist.get("train_loss", [])
    _is_running = training_state.get("started", False) and not training_state["done"] and not training_state["error"]

    _SPINNER_HTML = (
        '<div style="display:flex;align-items:center;gap:8px;margin-top:2px">'
        '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="10" cy="10" r="8" stroke="#ccc" stroke-width="2"/>'
        '<path d="M10 2a8 8 0 0 1 8 8" stroke="#332288" stroke-width="2" stroke-linecap="round">'
        '<animateTransform attributeName="transform" type="rotate" from="0 10 10" to="360 10 10" dur="1s" repeatCount="indefinite"/>'
        "</path></svg>"
        '<span style="color:#888;font-size:0.9em;">Training in progress…</span>'
        "</div>"
    )

    if _train_loss:
        _ep = list(range(1, len(_train_loss) + 1))
        _total_ep = epochs_s.value
        _val_loss_data = _hist.get("val_loss", [])
        _state_loss_raw = _hist.get("train_state_loss", [])
        _value_loss_raw = _hist.get("train_value_loss", [])
        _entropy_loss_raw = _hist.get("train_entropy_loss", [])
        _entropy_weights = _hist.get("train_entropy_weight", [])
        _probing = _hist.get("intermediate_probing", [])

        # Weighted components (weights match config: state=1.0, value=0.55, entropy varies)
        _w_state = _state_loss_raw  # weight 1.0, no change
        _w_value = [v * 0.55 for v in _value_loss_raw]
        _w_entropy = [
            e * w for e, w in zip(_entropy_loss_raw, _entropy_weights)
        ] if _entropy_weights else _entropy_loss_raw

        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4))

        # ── Left: total loss + weighted components ────────────────
        def _clip(lst, n):
            return lst[:n]

        _n = len(_ep)
        _ax1.plot(_ep, _train_loss, color="#332288", linewidth=2, label="Train total")

        if _w_state:
            _ax1.plot(_ep[:len(_w_state)], _w_state, color="#DDCC77",
                      linewidth=1, alpha=0.8, label="State (×1.0)")
        if _w_value:
            _ax1.plot(_ep[:len(_w_value)], _w_value, color="#117733",
                      linewidth=1, alpha=0.8, label="Value (×0.55)")
        if _w_entropy:
            _ax1.plot(_ep[:len(_w_entropy)], _w_entropy, color="#882255",
                      linewidth=1, alpha=0.8, label="Entropy reg")
        if _n < _total_ep:
            _ax1.set_xlim(1, _total_ep)
            _ax1.axvline(_n, color="#bbb", linestyle=":", linewidth=1)
        _ax1.set_xlabel("Epoch")
        _ax1.set_ylabel("Loss")
        _ax1.set_title(f"Loss  ({_n}/{_total_ep} epochs)")
        _ax1.legend(fontsize=8)
        _ax1.grid(True, alpha=0.3)

        # ── Right: state accuracy from intermediate probing ───────
        if _probing:
            _pe = [p["epoch"] for p in _probing]
            _pa = [p["discrete_accuracy"] for p in _probing]
            _ax2.plot(_pe, _pa, color="#332288", marker="o", markersize=4, label="State Accuracy")
            _ax2.set_ylim(0, 1)
            _ax2.legend(fontsize=8)
        else:
            _ax2.text(0.5, 0.5, "Probing runs every 5 epochs",
                      ha="center", va="center", transform=_ax2.transAxes, color="#aaa")
        if _n < _total_ep:
            _ax2.set_xlim(1, _total_ep)
        _ax2.set_xlabel("Epoch")
        _ax2.set_ylabel("State Accuracy")
        _ax2.set_title("State Accuracy")
        _ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        _buf = io.BytesIO()
        _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight")
        _buf.seek(0)
        _loss_bytes = _buf.read()
        plt.close(_fig)

        _parts = []
        if _val_loss_data:
            _parts.append(f"Val loss: **{_val_loss_data[-1]:.4f}**")
        if _probing:
            _parts.append(f"Accuracy: **{_probing[-1]['discrete_accuracy']:.1%}**")
        if training_state["done"]:
            _parts.append("✓ done")
        elif training_state["error"]:
            _parts.append(f"⚠ {training_state['error']}")
        _status_md = mo.md("  ·  ".join(_parts)) if _parts else mo.md("")
        _spinner = mo.Html(_SPINNER_HTML) if _is_running else mo.md("")

        panel_H = mo.vstack(
            [_spinner, _status_md, mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(_loss_bytes).decode()}" style="width:700px;max-width:100%">')],
            gap="0.4rem",
        )
    elif _is_running:
        panel_H = mo.vstack([mo.Html(_SPINNER_HTML.replace(
            "Training in progress…",
            "Training in progress — chart appears after first refresh."
        ))], gap="0")
    else:
        _err = training_state.get("error")
        panel_H = mo.vstack(
            [mo.callout(mo.md(f"**Training error:** `{_err}`"), kind="danger")]
            if _err
            else [mo.md("_No results yet. Generate a dataset above, then click **Train DRM**._")],
            gap="0",
        )

    mo.hstack([panel_G, panel_H], gap="3rem", align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell(hide_code=True)
def _(
    SystemType,
    angle0_s,
    angle1_s,
    base64,
    drag_widget,
    drm_refresh,
    get_visualization_bounds,
    mo,
    plt,
    tempfile,
    training_state,
    visualize_final_state_assignments,
):
    _ = drm_refresh  # re-run on each refresh tick so model becomes visible automatically
    mo.stop(
        training_state["model"] is None,
        mo.md(
            "_State assignments appear here after training completes._"
            if training_state["history"]
            else "_Train a model above to see state assignments._"
        ),
    )
    _model = training_state["model"]
    _points = drag_widget.value["saddle_points"]
    _angles = [float(angle0_s.value), float(angle1_s.value)]
    _bounds = get_visualization_bounds(SystemType.SADDLE_SYSTEM)

    _tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    _tmp_png.close()
    visualize_final_state_assignments(
        model=_model,
        output_path=_tmp_png.name,
        system_type="saddle_system",
        device="cpu",
        num_points=100,
        visualization_style="scatter",
        jitter_scale=0.3,
        points=_points,
        angles_degrees=_angles,
        bounds=_bounds,
    )
    plt.close("all")
    with open(_tmp_png.name, "rb") as _f:
        _viz_bytes = _f.read()
    os.unlink(_tmp_png.name)
    mo.vstack(
        [mo.md("### State Assignments"), mo.Html(f'<img src="data:image/png;base64,{base64.b64encode(_viz_bytes).decode()}" style="width:450px;max-width:100%">')],
        gap="0.5rem",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Example Results
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    show_examples_btn = mo.ui.button(
        label="Show example results",
        value=0,
        on_click=lambda v: v + 1,
    )
    mo.vstack(
        [
            mo.md(
                "In a hurry or want to see exemplary MDPs?   \n"
                "Press the button below to reveal example results from two pre-trained runs.   \n"
            ),
            show_examples_btn,
        ],
        gap="0.5rem",
    )
    return (show_examples_btn,)


@app.cell(hide_code=True)
def _(base64, mo, show_examples_btn):
    mo.stop(show_examples_btn.value == 0)

    def _load(rel_path):
        for prefix in ["", "mdp-world-model/"]:
            p = prefix + rel_path
            if os.path.exists(p):
                with open(p, "rb") as _f:
                    return _f.read()
        import urllib.request
        url = (
            "https://raw.githubusercontent.com/Lighfe/mdp-world-model/main/"
            + rel_path
        )
        with urllib.request.urlopen(url) as _r:
            return _r.read()

    def _img(rel_path, width="450px"):
        data = _load(rel_path)
        ext = rel_path.rsplit(".", 1)[-1].lower()
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        return mo.Html(
            f'<img src="data:{mime};base64,{base64.b64encode(data).decode()}"'
            f' style="width:{width};max-width:100%">'
        )

    _row1 = mo.vstack([
        mo.md("### Saddle System Dynamics"),
        mo.hstack([
            mo.vstack([
                        _img("docs/figures/multi_saddle_9_streamplot.png"),
                        mo.md("Centered Saddles, Imbalanced Lyapunov")
                    ], align="center").style({"width": "450px"}),
            mo.vstack([
                _img("docs/figures/multi_saddle_8_streamplot.png"),
                mo.md("Shifted Saddles, Rotated Manifold")
            ], align="center").style({"width": "450px"})
        ], gap="6rem", justify="start"),
    ], gap="0.5rem")

    _row2 = mo.vstack([
        mo.md("### Learned State Assignments"),
        mo.hstack([
            _img("docs/figures/final_state_scatter_ablation_baseline_ds9_seed_713.png"),
            _img("docs/figures/final_state_scatter_ablation_baseline_ds8_seed_713.png"),
        ], gap="6rem", justify="start"),
    ], gap="0.5rem")

    _row3 = mo.vstack([
        mo.md("### Learned Markov Decision Processes"),
        mo.hstack([
            mo.vstack([
                _img("docs/figures/saddle_mdp_9.jpeg"),
                mo.md("All States are Self-Loops")
            ], align="center").style({"width": "450px"}),
            mo.vstack([
                _img("docs/figures/saddle_mdp_8.jpeg"),
                mo.md("States 1 and 4 as Attractor States")
            ], align="center").style({"width": "450px"}),
        ], gap="6rem", justify="start"),
    ], gap="0.5rem")

    mo.vstack([_row1, _row2, _row3], gap="4rem")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
