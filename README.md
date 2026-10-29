# Zero-Cost NAS Competition (AutoML-Conf 22) - NASLib fork

<div align="center">
  <img src="images/zcp_competition.png" width="1000" height="200">
</div>

This repository is a working copy of [NASLib](https://github.com/automl/NASLib) (branch `automl-conf-competition`) adapted for the [Zero-Cost NAS Competition](https://sites.google.com/view/zero-cost-nas-competition/home) organized at [AutoML-Conf 22](https://automl.cc/). Participants implement zero-cost proxies on top of a lightweight version of NASLib and are scored on how well the proxy ranks architectures from tabular/surrogate NAS benchmarks.

The challenge: given N models from a search space (e.g. NASBench301), the proxy scores and ranks them for a given task (e.g. CIFAR10 classification). The metric is the Kendall-Tau rank correlation between predicted and true ranks, averaged across benchmarks (search space and dataset combinations). Scoring must use negligible compute (CPU only, hard runtime limit).

## What was added in this fork

Work by [casarinsof](https://github.com/casarinsof) on top of upstream NASLib (git history, commit `5d9eaf93`):

- **Zero-cost proxies** in `naslib/predictors/pruners/measures/`: `swap`, `zen`, `zico` (plus variants `my_zico`, `zico_lga`, `zico_nolog`, `zico_nomu`, `zico_nomunolog`), `nwot`, `lga`, `epe_nas`; modified `jacov` and `l2_norm`. Registered through `naslib/predictors/zerocost.py` and `naslib/predictors/pruners/`.
- **NASBench101 search space** (`naslib/search_spaces/nasbench101/`) with API loader `naslib/utils/nb101_api.py`; extended **TransNAS-Bench-101** support (macro/micro models in `naslib/search_spaces/transbench101/`).
- Updates to `naslib/runners/runner.py`, `naslib/evaluators/zc_evaluator.py`, `naslib/utils/` (dataset API, ops loading).
- **Experiment configs** (`configs/`) and **launch scripts** (`scripts/`) for batch-size ablations (batch sizes 2 to 128), proxy comparisons and benchmark runs across NB101/NB201/NB301/TNB101 micro and macro.

## Repository structure

```
naslib/                  NASLib library (search spaces, predictors/zero-cost measures, runners, utils)
configs/
  ablation/ablation_b*/  generated configs per batch size (2..128): <predictor>/<search_space>-<seed>/<dataset>/config_<seed>.yaml
  all/                   configs for the full runs (swap, zico, my_zico)
  benchmarks/            configs for benchmark runs
  legacy_runs/           older generated config sets (p201_c10_0, tnb101_macro_0)
scripts/
  create_configs.py      generates the YAML configs
  test_benchmark_apis.py checks that benchmark APIs load
  bash_scripts/          main run scripts (run_nb101/nb201/nb301/tnb101/tnb101_macro, download_data.sh)
  experiments/           per-experiment run scripts (abl_batch*, all, my_zico, nolog, nomunolog)
competition/             sample evaluation code and sample submission
tutorial/                tutorial notebook and sample submission
dataset_preparation/     manual dataset preparation instructions
tests/                   unit tests
images/                  figures
```

## Installation

Python 3.7-3.9 is required (see `setup.py`). Install PyTorch for your platform first, then:

```bash
git clone -b automl-conf-competition https://github.com/automl/NASLib/
cd NASLib
conda create -n automl-competition python=3.9
conda activate automl-competition
pip install --upgrade pip setuptools wheel
pip install -e .
pip install tornado archai transforms3d
```

Without conda: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
Note: the pinned `torch` version may not match your CUDA version.

## Data

Download all benchmarks (on macOS make sure `wget` is installed):

```bash
source scripts/bash_scripts/download_data.sh all
```

or a single search space/dataset:

```bash
source scripts/bash_scripts/download_data.sh <search_space> <dataset>
source scripts/bash_scripts/download_data.sh nb201 cifar10
source scripts/bash_scripts/download_data.sh nb201 all
```

- NASBench101: download from https://drive.google.com/drive/folders/1rwmkqyij3I24zn5GSO6fGv2mzdEfPIEa
- TransNAS-Bench-101: download from https://drive.google.com/drive/folders/1HlLr2ihZX_ZuV3lJX_4i7q4w-ZBdhJ6o (the original link is no longer working), and place `transnas-bench_v10141024.pth` in `naslib/data/`. The `segment` folder of Taskonomy must be renamed `segmentsemantic`.

If downloads fail, follow [dataset_preparation/](dataset_preparation/). Data is not tracked in git.

## Usage

Run from the repository root.

```bash
bash scripts/bash_scripts/run_nb201.sh
bash scripts/bash_scripts/run_nb301.sh
bash scripts/bash_scripts/run_tnb101.sh
```

These scripts generate configs with `scripts/create_configs.py` and run `naslib/runners/runner.py --config-file <config>`. A single run:

```bash
python naslib/runners/runner.py --config-file configs/ablation/ablation_b2/swap/nasbench201-0/cifar10/config_0.yaml
```

Batch-size ablations and variant experiments live in `scripts/experiments/`, for example `bash scripts/experiments/abl_batch2/run_nb201.sh`.

For the competition submission format and a local evaluation example see [competition/](competition/) and the [tutorial](tutorial/).

## Credits

NASLib is developed by the AutoML groups at Freiburg/Tubingen (original authors include Colin White, Arber Zela, Robin Ru, Yang Liu, Julien Siems, Yash Mehta, Rhea Sukthanker and others; see the git history and the [upstream repository](https://github.com/automl/NASLib)). The zero-cost proxies are re-implementations/adaptations of published methods; see the module files for details.

## License

Apache License 2.0, inherited from NASLib. See [LICENSE](LICENSE).
