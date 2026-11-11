# AML-KaggleCompetion-Group33

## Preprocessing

preprocess.py contains all pre processing functions. To preprocess data for model training and testing call

```
from preprocess import preprocess_data

X_train, y_train, X_validation, y_validation, X_test = preprocess_data()
```

**featurization** = featurization function for transofrming data into usable data for model training. Default is tf-idf featurization

**n** = sample size of training data, this is calculated after validation split so validation will always be the same 2,000 points for each random state

**random_state** = random state for randomization functions default is 12

The default split uses a stratified 80/20 holdout of `data/train.csv`. Use the same input file and row order, seed, code, and dependency versions for the same split. Share or compare the IDs to verify that everyone is using the same held-out rows. If `n` is less than 8,000, only the training split is sampled; the validation split stays fixed.

To export the raw shared split from the repository root, run:

```
python3 preprocess.py --export-split
```

This writes `data/splits/train_split.csv` and `data/splits/validation_split.csv` (with the original `id`, `text`, and `label` columns) without changing `data/train.csv`. These are raw texts, not the TF-IDF feature matrices. `preprocess_data()` recomputes the split from `data/train.csv` and the chosen seed; it does not read these exported files. Compare the IDs to verify agreement. The `data/` directory is ignored for newly generated files, so the exported CSVs will not automatically be shared through Git. The `--input`, `--output-dir`, and `--random-state` options can be used to select another source, destination, or seed; teammates should agree on the same settings.

## CNN

CNN models, training, experiment code, and tests live in [`cnn/`](cnn/README.md).
Run commands from the repository root:

```sh
venv/bin/python -u -m cnn --help
```

See the [CNN instructions](cnn/README.md) for training, evaluation, pretrained
embeddings, and generating submissions from saved checkpoints. Input datasets
and downloaded embeddings remain in `data/`; CNN-generated results and
checkpoints are stored in `cnn/outputs/` (ignored by Git).

After selecting a configuration on the fixed validation split, refit it on all
10,000 labeled passages and generate test predictions with:

```sh
venv/bin/python -u -m cnn train-full --epochs 12
```
