# Annual Plant Model

The goal of this framework is to extend and re-examine the coexistence metrics utilized by Yenni et al. (2012), incorporating broader parameter ranges and corrected equilibrium calculations.

Try out running this repository in your browser: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Tungdil01/Annual_Plant_Model/HEAD)

## Core Analysis Scripts

### Deterministic simulations

[main_simulation.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/main_simulation.py) reproduces the core numerical experiments and generates the main results (Figures 1 and S2). It also produces Figure S1 (phase plane diagrams). The script performs Latin hypercube sampling of model parameters for both Beverton-Holt and Ricker competition models, computes analytical equilibria and coexistence metrics (e.g., strength of self-limitation &#957; and competition effect &#957;<sub>C</sub>), and evaluates how well these metrics predict coexistence versus competitive exclusion under rare-species scenarios.

### Stochastic simulations

[main_stochasticity.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/main_stochasticity.py) extends the analysis to demographic stochasticity, producing Figures 2 and S3. It simulates population dynamics with Poisson noise, compares mean coexistence times under strong vs. weak self-limitation, and illustrates stochastic time series for selected parameter sets.

### Appendix analyses

[main_appendix.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/main_appendix.py) generates supplementary tables (Tables S1 and S2) that detail classification contingency tables and sources of misclassification between the original Yenni et al. method and the correct mathematical conditions. It also reproduces the original Yenni et al. parameter grids and compares theoretical coexistence classes against outcomes from equilibrium formulas and the S >= 1 filter.

### Reproducibility of Yenni et al. (2012)

The file [code_for_reproducibility.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/code_for_reproducibility.py) intends to reproduce the results of Yenni et al. (2012). The files [yenni_fig1.py](github.com/Tungdil01/Annual_Plant_Model/blob/main/yenni_fig1.py) and [yenni_fig2_3.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/yenni_fig2_3.py) reproduce Yenni et al. (2012) Figures 1 to 3.

Yenni, Glenda Marie, Peter Adler, and S. K. Morgan Ernest. 2012. "Strong self-limitation promotes the persistence of rare species." Ecology. 93 (3) pp. 456 - 461. [DOI](http://doi.org/10.1890/11-1087.1)

The original code by Yenni can be found at: [RareStabilizationSimulation](https://github.com/gmyenni/RareStabilizationSimulation/blob/master/Annual%20Plant/).

### Auxiliary files

[test_pars.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/test_pars.py) and [timeseries.py](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/timeseries.py) are auxiliary files that do not generate any core analysis, but help provide insights into the system.

## Development Environment

We use jupyter notebooks to document results and for the sake of reproducibility. All codes have their jupyter extensions (.ipynb) as well as their raw Python ones (.py) which you can choose from to run. We use Python version >= 3.11.

For creating a virtual environment, you can use [venv](https://docs.python.org/3/tutorial/venv.html).

## Required Packages

For the utilized packages, see the [requirements](https://github.com/Tungdil01/Annual_Plant_Model/blob/main/requirements.txt).
