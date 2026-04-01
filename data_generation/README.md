# data_generation

Generates trajectory datasets from 2D dynamical systems for training the DRM.
The pipeline has three layers: **Grid** → **Model + Solver** → **Simulator**.

---

## Coordinate spaces

Two spaces are used throughout:

| Space | Symbol | Description |
|-------|--------|-------------|
| Original space | x-space | Physical state space, possibly unbounded (e.g. `[0, ∞)`) |
| Transformed space | z-space | Bounded grid space used for cell construction and sampling |

The grid is always built in z-space. Initial conditions are sampled uniformly within z-space cells and then inverse-transformed back to x-space before simulation.

---

## Grid (`simulations/grid.py`)

```python
Grid(bounds, resolution, grid_transformations=None)
```

`bounds` defines the x-space domain per dimension (use `np.inf` for unbounded axes).
`resolution` is the number of grid cells per dimension.
`grid_transformations` is an optional list of transformation 3-tuples — one per dimension.

Each transformation 3-tuple has the form `(f, f_inv, f')`:
- `f(x)` — forward transform x → z
- `f_inv(z)` — inverse transform z → x
- `f'(x)` — derivative dz/dx (needed for vector field visualization)

**Available transformations** (all in `simulations/grid.py`):

| Function | Domain | Range | Notes |
|----------|--------|-------|-------|
| `tangent_transformation(x0, alpha)` | `[0, ∞)` | `[0, 1)` | Center at x0; default for tech substitution |
| `fractional_transformation(x0)` | `[0, ∞)` | `[0, 1)` | Scale odds; simpler but less control over spread |
| `logistic_transformation({'k', 'x_0'})` | `(-∞, ∞)` | `(0, 1)` | Used for FitzHugh-Nagumo |
| `negative_log_transformation(x0, alpha)` | `[0, ∞)` | `[0, 1)` | Alternative to tangent |
| `identity_transformation()` | any | same | No transformation |

**Example:**
```python
from data_generation.simulations.grid import Grid, tangent_transformation

grid = Grid(
    bounds=[(0, np.inf), (0, np.inf)],
    resolution=[30, 30],
    grid_transformations=[
        tangent_transformation(x0=3, alpha=0.5),
        tangent_transformation(x0=3, alpha=0.5),
    ]
)
```

Key methods:
- `grid.get_initial_conditions(n)` — returns `(X, ids)`, n uniform samples per cell in x-space
- `grid.transform(coords)` — x → z
- `grid.inverse_transform(coords)` — z → x
- `grid.get_cell_centers()` — returns cell centers in x-space (or z-space with `transformed_space=True`)

---

## Models and Solvers (`models/`)

Each dynamical system is split into a **model** (the ODE definition) and a **solver** (the numerical integrator).

### Model interface

Every model exposes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `x_dim` | int | State space dimension |
| `control_dim` | int | Number of control parameters |
| `control_params` | list[str] | Names of parameters treated as control inputs |
| `params` | dict | Default values for all parameters (`None` for control params) |
| `odes_vectorized(t, Z, control_params)` | method | Batched ODE RHS; `Z` is flattened `[x1_1,...,x1_n, x2_1,...,x2_n]` |
| `get_config()` | method | Returns serializable config dict |

### Available systems

| File | Model | Solver | Notes |
|------|-------|--------|-------|
| `saddle_system.py` | `MultiSaddleSystem` | `GeneralODENumericalSolver` | k saddle nodes; control selects active saddle |
| `tech_substitution.py` | `TechnologySubstitution` | `TechSubNumericalSolver` | Market equilibrium solved per step via `least_squares` |
| `social_tipping.py` | `SocialTipping` | `GeneralODENumericalSolver` | Opinion/behaviour tipping dynamics |
| `fitz_hugh_nagumo_system.py` | `FitzHughNagumoModel` | `GeneralODENumericalSolver` | Neuronal excitability; supports limit cycle and fixed-point regimes |
| `simple_test_models.py` | various | `GeneralODENumericalSolver` | Simple 2D models for debugging |
| `general_ode_solver.py` | — | `GeneralODENumericalSolver` | Shared solver used by all models except `TechnologySubstitution` |

### Solver interface

Both solvers expose:

```python
solver.step(X, control, delta_t, num_steps, steady_control) -> trajectory
# trajectory: shape (num_steps+1, n_samples, x_dim)

solver.get_derivative(X, control) -> dX
# dX: shape (n_samples, x_dim)
```

**Control array format** passed to `step()`: shape `(num_steps, n_samples, control_dim)`.
The `Simulator` builds this automatically from any scalar, 1D array, or full array input.

**`steady_control`** flag: when `True` (control constant across all timesteps), the solver uses
`scipy.solve_ivp` with adaptive RK45 for the full trajectory in one call — more accurate.
When `False` (time-varying control), it uses fixed-step RK4 step-by-step.

`TechSubNumericalSolver` differs from `GeneralODENumericalSolver`: instead of integrating a
closed-form ODE, it calls `scipy.least_squares` per sample at each step to solve a market
equilibrium, then uses the result as the system derivative.

---

## Simulator (`simulations/simulator.py`)

```python
Simulator(grid, model, solver)
results = simulator.simulate(control, delta_t, avg_samples_per_cell, num_steps, save_result)
```

The simulator orchestrates sampling, integration, and output formatting.

### Simulate call flow

1. Sample initial conditions `X` from the grid (uniform or importance-based)
2. Broadcast the control input to shape `(num_steps, n_samples, control_dim)`
3. Call `solver.step()` to get the full trajectory
4. Flatten into a DataFrame of `(x, c, y)` transition tuples

### Output format

Each row in the results DataFrame is one transition:

| Column | Description |
|--------|-------------|
| `run_id` | Timestamp string identifying the simulation run |
| `trajectory_id` | Cell index + sample number, e.g. `"3-7_2"` |
| `t0`, `t1` | Start and end time of the transition |
| `x0`, `x1` | State at time t0 |
| `c0` (+ `c1`, ...) | Control parameter(s) |
| `y0`, `y1` | State at time t1 |

### Importance-based sampling

By default, samples are drawn uniformly across grid cells. Optionally, cells can be
weighted by an importance measure — the mean angular difference between derivative
vectors under different control values. Cells where the control has a large effect on
the dynamics receive more samples.

```python
results = simulator.simulate(
    control=0.5,
    delta_t=2.65,
    use_importance=True,
    possible_controls=[0.5, 1.0],   # controls used to compute importance
    importance_method='angular',     # 'angular' | 'component_wise'
    importance_alpha=1.0,            # compresses distribution toward uniform (0 < alpha <= 1)
    min_samples_per_cell=1,
    total_samples=10000,
)
```

### Storing results

```python
simulator.store_results_to_sqlite(filename='datasets/results/my_run.db')
```

Creates a SQLite database with two tables:
- `{ModelName}_{control_params}` — the transition data
- `configs` — run metadata (grid config, solver config, timing, system info)

---

## Running simulations

All scripts are run from the **project root**:

```bash
# Saddle system
python datasets/run_saddle_simulation.py

# Technology substitution
python datasets/run_simulation.py \
    --resolution 30 30 \
    --control 1.0 \
    --delta-t 2.65 \
    --num-steps 1 \
    --db-name my_run.db

# With importance sampling
python datasets/run_simulation.py \
    --use-importance \
    --importance-method angular \
    --possible-controls 0.5 1.0 \
    --total-samples 10000 \
    --db-name my_run_importance.db
```

SLURM scripts for HPC are in `datasets/`:
- `run_simulation.slurm` — technology substitution
- `run_saddle_simulation.slurm` — saddle system
- `run_social_tipping_simulation.slurm` — social tipping
