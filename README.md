# Entropy Production Rate in the Refractory Density Neural Model

Non-markovian simulations (spin, discrete and continuous time mean-field models) of the time-discrete Spike-Response Hopfield model, a generalization of the classical Hopfield model that incorporates spike-response dynamics. The goal of this code is to model the system under various conditions and calculate information-theoretic measures of the system's dynamics such asthe steady-state entropy production. 

## Publications

[Insert publication details here]

## Results Replication 

Clone the repository and follow these instructions to set up the environment and run the simulations to replicate the results from the publications. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

To install the code in editable mode, run:

```bash
pip install -e .
```

## Project Structure
- `src/rdme/`: Reusable source code for the project.
    - `src/rdme/spin_model.py`: Shared implementation of spin-based simulations.
    - `src/rdme/mean_field.py`: Shared implementation of mean-field calculations.
- `scripts/`: Development scripts for running and validating simulations.
- `paper/`: Scripts that generate the figures in the publication.
- `tests/`: Unit tests for the project.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: This file, providing an overview of the project.
- `LICENSE`: License information for the project.