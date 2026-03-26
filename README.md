# dvfs_thesis

## results tuesday 24

### Energy comparisons

![alt text](result-processing/energy_comparison_plots/cholesky_mustard_f795_t4_energy_plot.png) 
![alt text](result-processing/energy_comparison_plots/cholesky_mustard_f795_t16_energy_plot.png) 
![alt text](result-processing/energy_comparison_plots/cholesky_mustard_f2040_t4_energy_plot.png) 
![alt text](result-processing/energy_comparison_plots/cholesky_mustard_f2040_t16_energy_plot.png) 
![alt text](result-processing/energy_comparison_plots/lu_mustard_f795_t4_energy_plot.png) 
![alt text](result-processing/energy_comparison_plots/lu_mustard_f795_t16_energy_plot.png) ![alt text](result-processing/energy_comparison_plots/lu_mustard_f2040_t4_energy_plot.png) ![alt text](result-processing/energy_comparison_plots/lu_mustard_f2040_t16_energy_plot.png)

Which observations do we make?

GPU0 have higher power usage than the other GPUS

The other GPUS have almost identical power draw

Which conclusions can be drawn?

GPU0 is doing the most work

Questions:

The time calculation is most likely inaccurate, however, how should I measure the time it takes to run the benchmark?

### Dags

#### cholesky t4
![alt text](dags/cholesky_t4.svg) 
#### cholesky t16
![alt text](dags/cholesky_t16.svg) 
#### lu t4
![alt text](dags/lu_t4.svg) 
#### lu t16
![alt text](dags/lu_t16.svg)

In the initial version of baseline experiments, gpu usage looked perfectly distributed. The graphs do show a relationship between tasks which makes that improbable, this is further reinforced from looking at the current baseline graphs.

The plots for 3/4 gpus do still look suspicously similar, potential cause is lag between starting profiler and program. Also the timer in the program shows times that are 1-2 orders of magnitude of the amount of time profiled, which needs looking in to.

