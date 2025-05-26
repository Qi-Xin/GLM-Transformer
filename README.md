# GLM-Transformer

GLM-Transformer is a hybrid neural spike train modeling framework that combines interpretable point process Generalized Linear Models (GLMs) with deep latent variable models, including a flexible Transformer-based variational autoencoder (VAE). This approach enables interpretable modeling of cross-population neural interactions while leveraging the power of deep learning to capture individual-neuron dynamics (also known as background dynamics, e.g., neural activity caused by locomotion or arousal).

## Features

- **Point Process GLM**: Interpretable modeling of cross-population interactions, quantifying how a neuron's spike is associated with changes in the firing rate of neurons in other populations, including time delays.
- **Deep Latent Variable Model**: Transformer-based VAE for capturing unobserved background or individual-neuron dynamics.
- **Flexible Data Handling**: Supports multi-session Allen Brain Observatory data, with efficient memory management for large-scale datasets.

## Installation

1. **Download the repository:**
   Clone or download this repository to your local machine.

2. **Set up the environment:**
   It is recommended to use the provided conda environment for reproducibility:
   ```bash
   conda env create -f allen_env_full.yml
   conda activate allen
   ```
   Note: CUDA 12.1 or higher is required due to known bugs with `torch.fft.fft` in CUDA 11.8 or earlier (see https://github.com/pytorch/pytorch/issues/111884).

## Usage

Please see `tutorial.ipynb` to get started.

### Key Components

- **GLMTransformer.py**: Deep latent variable model for capturing complex neural dynamics.
- **model_trainer.py**: Handles batching, training, validation, and model saving/loading.
- **GLM.py**: Provide B-spline and Raised Cosine basis and other supports for GLM-Transformer

### Data

- For simulated data using GLM or EIF neurons, data generation can be found in the corresponding Jupyter Notebooks.
- For the Allen Institute dataset, you need to download the data using the AllenSDK and set the appropriate manifest path in your environment.
- Due to limited memory when loading multiple Allen Institute sessions, we provide `Allen_dataloader_multi_session`, a multi-session dataloader that handles this efficiently. The loader converts spike times (a pandas DataFrame where each row is a spike with features including time and neuron_id) to spike trains (a 3D matrix: time × neuron × trial) on the fly while loading a batch (typically 64 trials from the same session). For a quick and lightweight introduction, see `Allen_dataloader_multi_session_tutorial.ipynb`.

### Example Notebooks

- `Fig2.ipynb`, `Fig3.ipynb`, `Fig4.ipynb`, `Fig5.ipynb`, `SupFig1.ipynb`, `SupTable_Ablation_experiments.ipynb`: Example analyses and figures from the manuscript.

### Model Training

A typical workflow involves:

1. **Prepare a dataloader:**
   ```python
   from DataLoader import Allen_dataloader_multi_session
   dataloader = Allen_dataloader_multi_session(
       session_ids=[...],  # List of Allen session IDs
       train_ratio=0.7,
       val_ratio=0.15,
       batch_size=32,
       shuffle=True
   )
   ```

2. **Set hyperparameters and initialize the trainer:**
   ```python
   from model_trainer import Trainer
   params = {...}  # See model_trainer.py for all options
   trainer = Trainer(dataloader, path='./results', params=params)
   ```

3. **Train the model:**
   ```python
   trainer.train()
   ```

4. **Save or load models:**
   ```python
   trainer.save_model_and_hp()
   # or
   trainer.load_model_and_hp('path_to_saved_model.pth')
   ```

## Repository Structure

- `GLMTransformer.py`: Implementation of the GLM-Transformer model.
- `GLM.py`: Core implementation of the point process GLM and related utilities.
- `model_trainer.py`: Training pipeline, including model initialization, optimizer setup, and training loop.
- `utility_functions.py`: Helper functions for data processing, plotting, and neuroscientific analysis.
- `EIF_params.pickle`: Example parameter file for EIF neuron simulations.
- `group_id_all_a_c/`: Example data files for group membership and condition IDs.
- `allen_env_full.yml`: Conda environment file specifying all dependencies.
- `LICENSE`: License information.
- `README.md`: This file.

<!-- ## Citing

If you use this codebase in your research, please cite the associated publication (add citation here if available). -->

<!-- ## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details. -->

## Acknowledgments

- Allen Institute for Brain Science for the Allen Brain Observatory dataset and AllenSDK.
- PyTorch, NumPy, SciPy, and other open-source libraries.

