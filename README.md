# Motor Imagery Decoding Using Ensemble Curriculum Learning and Collaborative Training

## Installation

We recommend installing the required packages using Python's native virtual environment. For Python 3.4+, this can be done as follows:

```bash
$ python -m venv ensemble_venv
$ source ensemble_venv/bin/activate
(ensemble_venv) $ pip install --upgrade pip
(ensemble_venv) $ pip install -r requirements.txt
```

## Dataset preprocessing

Before training a model, you need to preprocess a dataset and then save its preprocessed version. To do this, use `preprocess_dataset.py`, running the following command:

```bash
(ensemble_venv) $ python preprocess_dataset.py
```

This process will create the folder `preprocessed_data` under the root directory.

## Training

To train a model, you need to use `run_experiment.py`, running the following command:

```bash
(ensemble_venv) $ python run_experiment.py
```

The training process will create the following directories:

```
json        (note: contains the arguments of each experiment, stored in JSON format)
checkpoints (note: contains the saved model checkpoints, stored as .pth files)
results     (note: contains the experimental results as an .xlsx file and the train/val/test splits as a .pkl file)
plots_topo  (note: contains the visualizations of the model's spatial filters, saved as .eps and .svg files)
```

## Acknowledgement

The research of Georgios Zoumpourlis was supported by QMUL Principal's Studentship.

## Citation

[1] Georgios Zoumpourlis and Ioannis Patras. Motor Imagery Decoding Using Ensemble Curriculum Learning and Collaborative Training. 12th IEEE International Winter Conference on Brain-Computer Interface (BCI), 2024.

## Credits

Special thanks go to the authors of [MOABB](https://github.com/NeuroTechX/moabb), [Braindecode](https://github.com/braindecode/braindecode), [MNE-Python](https://github.com/mne-tools/mne-python) and [pyRiemann](https://github.com/pyRiemann/pyRiemann) libraries, which have been essential for this project.

The current GitHub repo contains code parts from the following repository (heavily chopped, adapting and keeping some logging tools): <br />
Code in `third_party/skorch`: https://github.com/skorch-dev/skorch <br />
Credits go to its owners/developers. Its license is included in the corresponding folder.
