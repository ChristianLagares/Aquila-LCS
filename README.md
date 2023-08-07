# Aquila-LCS: Particle Simulation and FTLE Calculation Script

This script (`ftle_cuda.py`) is designed to run particle simulations and calculate the Finite-Time Lyapunov Exponent (FTLE) on a given dataset using Nvidia GPUs (a CPU version of the code is available upon reasonable request).

## Overview

1. The script initializes by selecting the appropriate GPU via the `select_gpu()` function.
2. Various configuration parameters are set, which influence how the particle simulations and FTLE calculations are performed.
3. Particle simulations are run with different configurations, directions (backward or forward), and centered around various flow fields.
4. After particle simulations, the FTLE calculations are performed.
5. All necessary resources are released at the end.

## Configuration Parameters

Users need to modify the following parameters to suit their specific datasets and requirements:

- `NFIELDS`: Number of Flow Fields. This determines how many flow fields will be processed in the simulation.
- `UPSCALE`: Determines the upscaling factor for additional flow fields between every real field. Setting this to `1` avoids upscaling.
- `FLOWFIELDS`: If a dynamic FTLE is required, set this to the desired number of FTLEs.
- `SKIP`: If skipping underlying flow fields is needed, adjust this number.
- `CONFIGS`: This allows for a list of refinement values along x, y, z directions. For example, `(8, 8, 3)` indicates a refinement of `8x` along the streamwise and wall-normal directions and `3x` along the spanwise direction.
- `BASE_DIR`: The base directory where the data is located.
- `CASE_NAME`: Name of the case being processed.
- `MACH_MOD`: Machine model or identifier.
- `WALL_CONDITION`: Condition of the wall (e.g., "Adiabatic").
- `coord_path`: Path to the coordinate file.
- `dt`: Time difference used in simulations. There are two instances of `dt` in the code that need to be modified according to the requirements.

## Running the Script

After modifying the necessary configuration parameters, run the script. Ensure all dependencies (such as necessary libraries and datasets) are available and accessible.

`run_aquila_lcs.pbs` offers an example of how to run the script using the PBS batch scheduler.

## Sample Data

For the purposes of validation, we provide a dataset corresponding to a Low Reynolds number (Low Re) condition at Mach 0.8. This dataset can be instrumental for those seeking to benchmark or validate their methodologies against well-established data. The dataset contains 4001 flow fields of a turbulent boundary layer over an adiabatic flat plate. The flow fields are stored in compressed (lossless) HDF5 files which occupy about 70-75% of the total file size.


**Access the Dataset:**  
[Low Re, Mach 0.8 Dataset on Google Drive](https://drive.google.com/drive/folders/1vBzDYxTwJjJ-0f6A6mpb3K35MAqRkhOu?usp=share_link)


## Notes

- The script expects data to be in 4 dimensional HDF5 files with shape (`Nx,Ny,Nv,Nz`) where
  * `Nx` is the number of nodes along the streamwise direction.
  * `Ny` is the number of nodes along the wall-normal direction.
  * `Nz` is the number of nodes along the spanwise direction.
  * `Nv` is the number of variables and is assumed to be 5 internally (Pressure, 3 Components of Velocity & Temperature).
- The script provides extensive print outputs to monitor the progress, including the particle count, time elapsed during various stages, and the current configuration being processed.
- The script uses both backward (`-1`) and forward (`1`) directions for the particle simulations.
- If `RUNSIM` is set to `True`, particle simulations will be executed. If `CALCULATE_FTLE` is set to `True`, the FTLE and FSLE calculations will be performed.
- The script is designed to handle parallel processing and uses MPI for communication. Ensure that an MPI environment is set up correctly before running the script.
- For efficient memory management, the garbage collector is called regularly throughout the script.

Please make sure you understand each parameter's purpose before modifying and running the script. If unsure, consult the accompanying documentation or the person responsible for the initial setup.

## Citation Requirement
If you use this software in your research, academic, or professional work, please cite our related paper to acknowledge the software's origins and offer credit to its developers:

Lagares, Christian, and Guillermo Araya. 2023. "A GPU-Accelerated Particle Advection Methodology for 3D Lagrangian Coherent Structures in High-Speed Turbulent Boundary Layers." Energies 16, no. 12: 4800. [https://doi.org/10.3390/en16124800](https://doi.org/10.3390/en16124800)

For those using BibTeX (for LaTeX), here's the corresponding entry:

```
@article{lagares2023gpu,
  title={A GPU-Accelerated Particle Advection Methodology for 3D Lagrangian Coherent Structures in High-Speed Turbulent Boundary Layers},
  author={Lagares, Christian and Araya, Guillermo},
  journal={Energies},
  volume={16},
  number={12},
  pages={4800},
  year={2023},
  publisher={Multidisciplinary Digital Publishing Institute},
  doi={https://doi.org/10.3390/en16124800}
}
```

Your citation helps support and recognize our research and development efforts. Thank you!


