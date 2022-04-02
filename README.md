# SW-inference
Code repository for reproducing the numerical studies in the paper 

Manole, T., Balakrishnan, S., Wasserman, L. (2022). [Minimax Confidence Intervals for the Sliced Wasserstein Distance](https://arxiv.org/abs/1909.07862). _Electronic Journal of Statistics 16(1), 2252-2345._


## Dependencies 
This code has only been tested using Python 3.5 on a standard Linux machine. Beyond the Python Standard Library, the only required packages for the main code are `NumPy` and `sklearn`. The plotting code also requires the packages `alphashape` and `descartes`.

## Usage  
### Reproducing the Simulation Results
To reproduce the simulation results for Models 1-5 in Section 6, navigate to the `simulations` directory, and run the following command
```{python}
python experiment.py -mod <Model Number> -meth <Method Name> 
```
for `<Model Number>` ranging from `1` to `5`, and for `<Method Name>` equal to
`exact`, `pretest` or `boot`. 
Further options can be found in `experiment.py`. To reproduce the simulations results
for Model 6, run the commands
```{python}
python asymp_experiment.py -asymp 1 -eps <epsilon Value> -r 2 -J True
python asymp_experiment.py -asymp 1 -eps <epsilon Value> -r 2
```
for `<epsilon Value>` equal to `0, 0.1, 0.2, 0.3, 0.4`, and
```{python}
python -u asymp_experiment.py -asymp 2 -r <r Value> -J True 
python -u asymp_experiment.py -asymp 2 -r <r Value> 
```
for `<r Value>` equal to `1, 2, 4, 8, 16`.
To plot results, run `plotting.py`, `plotting_asymp1.py` and `plotting_asymp2.py`.

### Reproducing the Application
To reproduce the application to the toggle switch model in Section 7, navigate to the `applications`
directory. For the well-specified model, run
```{python}
python analysis.py -ns <m Value> 
```
for `<m Value>` ranging in `5000, 10000, 20000`. Furthermore, for the misspecified model, run
```{python}
python analysis.py -ns 10000 -ms True
```
To estimate the projection parameter in the misspecified case, run  
```{python}
python projection.py
```
To plot the results, run
```{python}
python plotting_well_specified.py
python plotting_misspec.py
``` 
