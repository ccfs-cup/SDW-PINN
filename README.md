# SDW-PINN: Groundwater Flow in Heterogeneous Media

This repository contains the research code, conductivity fields, reference simulation data, and saved neural network parameters associated with the SDW-PINN study. It supports physics-informed prediction of transient two-dimensional groundwater hydraulic head under rainfall recharge and pumping, together with comparisons of conductivity-gradient treatments, loss weighting, and sensitivity to heterogeneous-field parameters. Generated figures, logs, and local environments are excluded from the source distribution; plotting scripts and the demo regenerate their outputs.

The implementation consists of standalone Python experiment scripts, plus a portable `run_demo.py` inference tutorial. Experiment settings and file paths for the original training scripts are defined inside each script. The manuscript should be consulted for the full physical assumptions, units, and interpretation of the reported experiments.

## 1. Method and implementation

The network maps spatial coordinates and time `(x, y, t)` to hydraulic head `h`. The principal case scripts use six fully connected hidden layers, Swish activation, and a linear scalar output. Spatial and temporal derivatives of the predicted head are computed with PyTorch automatic differentiation.

For isotropic conductivity, the recharge-case residual implemented in the code is

```text
r = 0.1 * h_t - K * (h_xx + h_yy) - K_x * h_x - K_y * h_y - R(x)
```

Uniform recharge uses `R = 0.025`; linear recharge uses `R(x) = 0.0005*x`. Pumping cases use directional conductivity terms and positive Gaussian sink terms in the residual. The single-well case centers its sink at `(50, 50)`; the multiple-well case includes three sinks.

Training combines mean squared errors for the PDE residual, prescribed-head boundaries, initial head, and no-flow boundaries:

```text
L = lambda_f * L_PDE + lambda_bc * L_BC
  + lambda_ic * L_IC + lambda_nf * L_NF
```

The main cases use a spatial domain `[0, 100] x [0, 100]` and time interval `[0, 20]`, with initial head 5, head 5 on the left/right boundaries, and zero normal head derivative on the upper/lower boundaries.

### Which script implements which variant?

| Script | Role |
| --- | --- |
| `PINN_applications/Case1_uniform_rainfall_recharge/Case1_main.py` | Uniform recharge; finite-difference conductivity gradients with added Gaussian gradient noise; fixed loss weights. |
| `PINN_applications/Case1_uniform_rainfall_recharge/Case1_main2.py` | Uniform recharge variant using `K1_noisy.txt`; fixed loss weights. |
| `PINN_applications/Case1_uniform_rainfall_recharge/Case1_main3.py` | SDW-PINN implementation combining Scharr conductivity gradients, Xavier initialization, and dynamic loss weighting; uses `K1_noisy.txt`. |
| `PINN_applications/Case2_linear_rainfall_recharge/Case2_main.py` | Linear rainfall recharge; fixed loss weights. |
| `PINN_applications/Case3_single_pumping_well/Case3_main.py` | Single pumping well; fixed loss weights. |
| `PINN_applications/Case4_multiple_pumping_wells/Case4_main.py` | Three pumping wells; fixed loss weights. |

In `Case1_main3.py`, conductivity gradients are calculated with replicated boundary padding and 3-by-3 Scharr kernels, scaled by `1/(32*2*dx)` or `1/(32*2*dy)` as written in the implementation. Network weights receive Xavier uniform initialization and biases start at zero. After each optimizer step, the four loss weights are updated separately:

```text
lambda_i <- clip(lambda_i + lr_lambda * L_i, lambda_min, lambda_max)
```

Their initial values are `(100, 1, 1, 8)` for `(PDE, BC, IC, NF)`, with `lr_lambda = 1e-4` and bounds `[0.1, 1000]`. The `Dynamic_weighting` directory primarily contains scripts for plotting existing loss comparisons; its filenames should not be interpreted as training entry points.

## 2. Repository layout

```text
Code/
|-- README.md
|-- requirements.txt               # Pinned direct scientific dependencies
|-- requirements-lock.txt          # Pinned dependencies including transitive packages
|-- requirements-legacy.txt        # Optional TensorFlow/pyDOE dependencies
|-- environment.yml                # Conda environment using the same pip lock
|-- run_demo.py                    # Portable CPU inference tutorial
|-- PINN_applications/
|   |-- Case1_uniform_rainfall_recharge/
|   |-- Case2_linear_rainfall_recharge/
|   |-- Case3_single_pumping_well/
|   |-- Case4_multiple_pumping_wells/
|   |-- Dynamic_weighting/          # Loss comparison plots
|   |-- Correlation_length/         # Conductivity correlation-length experiments
|   |-- Logarithm_variance/         # Logarithmic-variance experiments
|   `-- Parameterization/           # Q1-Q4 pumping-parameter experiments
`-- random_field_generation/        # Random fields, interpolation, and conversion
```

Case directories contain combinations of `K*.txt`/Excel conductivity data, `Comsol-*.xlsx` reference data, `pth/` checkpoints, and analysis scripts. The `loss/` and `result/` directories are output locations for regenerated figures; create them when adapting the original script paths. Some directories also contain `sanwei/` visualization scripts. Existing files document individual experiment configurations and may use different checkpoint selections.

## 3. Environment setup

Use Python **3.9.25** in an isolated environment. `requirements.txt` pins all direct dependencies for the main PyTorch experiments, analysis, and GSTools random-field tools; `requirements-lock.txt` additionally pins their transitive dependencies. These files define a reviewer environment, rather than a recovered record of the original authors' training environment. The verification scope is stated below.

From the `Code` directory:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the dependencies used by the main PyTorch cases and analysis scripts:

```bash
python -m pip install pip==23.0.1
python -m pip install -r requirements-lock.txt
```

Alternatively, create the same Python/pip package environment with Conda, running from the `Code` directory:

```bash
conda env create -f environment.yml
conda activate sdw-pinn
```

`environment.yml` uses the same pip lock to avoid maintaining conflicting pip and Conda package versions. GSTools is already included. To install only the pinned direct dependencies while allowing pip to resolve transitive versions, use `python -m pip install -r requirements.txt`; use the lock for the documented reviewer environment.

The auxiliary TensorFlow script `random_field_generation/tgnn1_succ.py` additionally imports `tensorflow` and `pyDOE`. Their versions are listed in `requirements-legacy.txt`, which can be installed in a separate environment with `python -m pip install -r requirements-legacy.txt`. It is outside the verified PyTorch workflow and also requires the absent local `makedata.make_1_data` module and external data. Dependency installation alone cannot make that legacy script reproducible.

A CUDA-capable GPU is useful for full training. Main case scripts automatically select `cuda:0` when CUDA is available and otherwise use the CPU; some auxiliary/parameterization scripts explicitly select `cuda:1`, which must be adjusted on a single-GPU machine. Several training scripts enable `DataParallel` when multiple GPUs are visible. A GPU-enabled PyTorch installation must match the local system.

Check the selected environment:

```bash
python -c "import torch, numpy, scipy, pandas, matplotlib, seaborn, openpyxl; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

### One-command example (no path editing or retraining)

After installing the environment, run:

```bash
python run_demo.py
```

The tutorial uses CPU inference, the supplied Case 1 checkpoint `pth/qianxiang_pth/final.pth`, and the supplied `comsol-label1.txt` reference. It reuses the `Swish` and `Net` class definitions from `Case1_Error.py` without executing that script's file-loading or plotting code. It evaluates all 10,201 reference coordinates at `t=10` and automatically creates:

| File | Contents |
| --- | --- |
| `demo_output/head_comparison.png` | Reference head, predicted head, and absolute error side by side. |
| `demo_output/predictions.csv` | x, y, reference head, predicted head, and absolute error. |
| `demo_output/metrics.json` | MAE, RMSE, relative L2 error, checkpoint/reference paths, and PyTorch version. |

For another reference time or output directory:

```bash
python run_demo.py --time 20 --output-dir demo_output_t20
```

Use `python run_demo.py --help` for checkpoint and output options. Paths to the default assets are resolved relative to the script, so it can also be launched from another working directory. A custom checkpoint must match the Case 1 network with six hidden layers of width 60.

This is an archived-checkpoint inference example. The selected checkpoint's provenance does not establish correspondence to a manuscript table. Its comparison with the bundled reference is illustrative, and the script reports the measured errors without assuming a publication-level accuracy threshold. Original training scripts still require the path configuration described next.

### Environment verification

On 2026-10-16, the direct requirements were installed into a fresh Windows x86_64 virtual environment with Python 3.9.25 and pip 23.0.1. All nine direct dependency imports succeeded, `python -m pip check` reported no broken requirements, and `python run_demo.py` completed using PyTorch 2.5.1 CPU. The lock file was exported from that environment with `pip freeze --all`.

The default demo produced 10,201 finite predictions at `t=10`: MAE `0.964826`, RMSE `1.074426`, and relative L2 error `0.166441`. These values document successful execution and the actual archived-checkpoint comparison, rather than a claim of manuscript accuracy. The three output files were generated and the comparison figure was visually inspected.

The Conda YAML syntax and its shared-lock reference were checked; a separate Conda environment creation was not executed. Linux/macOS installation, GPU training, full training, and the optional legacy TensorFlow workflow have not been validated. Platform-specific wheels, CUDA runtimes, and driver versions can differ even when Python package versions match. For a matching tested setup, use Windows x86_64, Python 3.9.25, the supplied lock, and CPU inference. Keep `requirements-lock.txt` beside `environment.yml` and run environment creation from this directory.

## 4. Configure paths before running

Scripts retain absolute paths from the original Linux servers, including `/home/users/liyifan/LYF/`, `/home/cc/CCFs/LYF/`, and other experiment locations. These paths must be changed for the local checkout. Changing the working directory alone is insufficient.

For each script you intend to run:

1. Update all active input paths, including conductivity files, reference workbooks, and checkpoints.
2. Update all output paths passed to `torch.save`, `to_excel`, and `savefig`.
3. Create the chosen output directories before execution. Training scripts do not consistently create them.
4. Use a new output directory for a new run to preserve the supplied checkpoints and figures.

For example, in the Case 1 training script, a portable path setup can be introduced as follows, then used to replace its original read/save paths:

```python
from pathlib import Path

case_dir = Path(__file__).resolve().parent
run_dir = case_dir / "reviewer_run"
(run_dir / "pth").mkdir(parents=True, exist_ok=True)
(run_dir / "loss").mkdir(parents=True, exist_ok=True)

K_data = np.loadtxt(case_dir / "K1_noisy.txt")
# Inside the checkpoint-saving block:
# torch.save(net.state_dict(), run_dir / "pth" / f"final_epoch_{epoch}.pth")
# After training:
# torch.save(net.state_dict(), run_dir / "pth" / "final.pth")
# plt.savefig(run_dir / "loss" / "loss_curve.png")
```

This is an adaptation example; the supplied experiment scripts have not been changed by this README.

## 5. Run training

After configuring paths and output directories, run the SDW-PINN uniform-recharge experiment from the repository root:

```bash
python PINN_applications/Case1_uniform_rainfall_recharge/Case1_main3.py
```

The principal defaults in this script are:

| Setting | Value |
| --- | --- |
| Hidden layers / width | 6 / 60 |
| Optimizer | Adam |
| Initial learning rate | `1e-3` |
| Learning-rate schedule | StepLR, factor `0.5` every `4000` iterations |
| Training iterations | `30000` |
| Initial-condition samples | `80000` interior + `5000` at each left/right boundary |
| Prescribed-head samples | `5000` per left/right boundary |
| No-flow samples | `10000` per upper/lower boundary |
| PDE grid | `99 x 99 x 20 = 196020` space-time points |

The variable `n_c = 100` is retained in the source but does not control the actual PDE grid size. To reduce the PDE workload, change the spatial/time `linspace` definitions and their corresponding reshape sizes and indexing consistently.

For a preliminary execution check, use a separate copy/configuration with fewer iterations and fewer condition/grid samples. This checks execution only and does not reproduce the manuscript's results. Full training evaluates large sets of points with second derivatives and can require substantial memory; no validated runtime or minimum GPU-memory requirement is supplied.

Other main experiments can be launched after the same path preparation:

```bash
python PINN_applications/Case1_uniform_rainfall_recharge/Case1_main.py
python PINN_applications/Case2_linear_rainfall_recharge/Case2_main.py
python PINN_applications/Case3_single_pumping_well/Case3_main.py
python PINN_applications/Case4_multiple_pumping_wells/Case4_main.py
```

The fixed-weight recharge scripts use `(100, 1, 1, 8)`; the principal pumping scripts use `(150, 3, 1, 8)` and hidden width 120. Inspect each script before comparing runs: inputs, gradient treatment, architecture, and loss weights can differ.

Training prints loss components and learning rates periodically, saves `final_epoch_<epoch>.pth` every 1000 iterations (including iteration 0), writes `final.pth` at completion, and saves a loss curve. Checkpoints contain the network `state_dict`; they do not include optimizer state, scheduler state, random-number state, or the dynamic loss weights.

## 6. Evaluate saved checkpoints

Reviewers can inspect saved-model predictions without retraining. For example:

1. Open `PINN_applications/Case1_uniform_rainfall_recharge/Case1_Error.py`.
2. Set `pretrained_model_path` to the checkpoint to evaluate. The supplied active selection is an intermediate checkpoint, not automatically `final.pth`.
3. Set `file_path` to the local `Comsol-Label1.xlsx` and update all output paths.
4. Run:

```bash
python PINN_applications/Case1_uniform_rainfall_recharge/Case1_Error.py
```

This script evaluates head at reference coordinates and times and exports a workbook with `Prediction_<time column>` and absolute `Error_<time column>` columns. Other case-specific `Case*_Error*.py`, `Line_drawing.py`, contour, correlation, and well-head scripts generate additional metrics and figures. Inspect their input workbook paths because some consume predictions exported by an earlier evaluation script.

For CPU evaluation of GPU-saved weights, replace the checkpoint-loading call with:

```python
state_dict = torch.load(pretrained_model_path, map_location=device)
```

The evaluation architecture must match the training architecture, including hidden width. Multi-GPU checkpoints may have a `module.` prefix; the Case 1 evaluation script already includes `remove_module_prefix` to handle it. Avoid importing a training script solely to obtain `Net`, because training executes at module level.

## 7. Data conventions and additional experiments

- **Conductivity text files:** the main cases read whitespace-separated rows with columns `(x, y, K)` and reshape the third column to `(101, 101)`. The ordering is significant: the code interprets dimension 0 as x and dimension 1 as y. Preserve the supplied row order.
- **Reference workbooks:** Case 1 evaluation reads x and y from the first two columns and head values from subsequent time columns. It parses time labels using `float(time_col.split('=')[1])`; custom data must follow this convention or the parser must be adjusted.
- **COMSOL data:** exported reference files are included for comparison. The provided PyTorch training loops generate their initial/boundary samples and do not require a running COMSOL installation. Commented observation-data sections are not active supervision.
- **Correlation length:** `Correlation_length/` includes separate K10, K30, and K50 configurations and associated analysis scripts.
- **Logarithmic variance:** `Logarithm_variance/` includes M0.1, M0.3, and M0.5 configurations.
- **Pumping parameterization:** `Parameterization/Q1` through `Q4` contain separate experiments. Some load fields from another case directory; inspect the actual file paths rather than assuming inputs are local.
- **Random fields:** `random_field_generation/` contains GSTools examples, interpolation scripts, and text/Excel conversion tools. For example, `demoGaussian.py` generates a Gaussian field on a 100-by-100 grid. Its output is not directly interchangeable with the 101-by-101 input expected by the main cases; interpolation/conversion and a check of the data ordering are required.

## 8. Reproducibility notes

The scripts do not consistently set random seeds. Randomly sampled conditions, added noise, network initialization, and GPU execution can affect results. For a controlled new experiment, set Python, NumPy, and PyTorch seeds before sampling or model construction, and record dependency versions, hardware, input field, script, hyperparameters, and checkpoint selection. A seed alone does not guarantee identical results across different systems.

The supplied checkpoints and scientific data can help inspect the archived experiments, but filenames alone do not establish which checkpoint produced a particular manuscript figure. Select the matching experiment inputs and architecture explicitly. Full training and numerical reproduction of the manuscript have not been validated as part of this documentation update.
