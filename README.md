# GLM-Transformer

GLM-Transformer is a hybrid neural spike train modeling framework that combines interpretable point process Generalized Linear Models (GLMs) with deep latent variable models, including a flexible Transformer-based variational autoencoder (VAE). This approach enables interpretable modeling of cross-population neural interactions while leveraging the representational power of deep learning.

## Features

- **Point Process GLM**: Classical interpretable modeling of neural spike trains, including baseline, coupling, and history effects.
- **Deep Latent Variable Model**: Transformer-based VAE for capturing complex, high-dimensional latent structure in neural data.
- **Flexible Data Handling**: Supports multi-session Allen Brain Observatory data, with efficient memory management for large-scale datasets.

## Repository Structure

- `GLMTransformer.py`: Implementation of the Transformer-based VAE and FCGPFA (Factorized Coupled Gaussian Process Factor Analysis) model.
- `GLM.py`: Core implementation of the point process GLM and related utilities.
- `model_trainer.py`: Training pipeline, including model initialization, optimizer setup, and training loop.
- `utility_functions.py`: Helper functions for data processing, plotting, and neuroscientific analysis.
- `EIF_params.pickle`: Example parameter file for EIF neuron simulations.
- `group_id_all_a_c/`: Example data files for group membership and condition IDs.
- `allen_env_full.yml`: Conda environment file specifying all dependencies.
- `LICENSE`: License information.
- `README.md`: This file.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd GLM-Transformer
   ```

2. **Set up the environment:**
   It is recommended to use the provided conda environment for reproducibility:
   ```bash
   conda env create -f allen_env_full.yml
   conda activate allen
   ```

3. **Install additional dependencies (if needed):**
   All required packages are specified in `allen_env_full.yml`, including:
   - `torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `allensdk`, etc.

## Usage

### Data Preparation

- The code is designed to work with Allen Brain Observatory data. You may need to download the data using the AllenSDK and set the appropriate manifest path in your environment.
- Example group membership and condition ID files are provided in `group_id_all_a_c/`.

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

### Model Components

- **GLM.py**: Add effects (baseline, coupling, history, etc.) and fit interpretable GLMs.
- **GLMTransformer.py**: Deep latent variable model for capturing complex neural dynamics.
- **model_trainer.py**: Handles batching, training, validation, and model saving/loading.

### Example Notebooks

- `Fig2.ipynb`, `Fig3.ipynb`, `Fig4.ipynb`, `Fig5.ipynb`, `SupFig1.ipynb`, `SupTable_Ablation_experiments.ipynb`: Example analyses and figures from the associated publication.

## Citing

If you use this codebase in your research, please cite the associated publication (add citation here if available).

## License

This project is licensed under the terms of the MIT license. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Allen Institute for Brain Science for the Allen Brain Observatory dataset and AllenSDK.
- PyTorch, NumPy, SciPy, and other open-source libraries.

