"""Check split isolation and submission contracts without fitting models."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from argparse import Namespace

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cnn import generalization as experiment
from cnn.neural import CNNConfig, SequenceEncoder


class GeneralizationTests(unittest.TestCase):
    def test_cli_defaults_keep_inputs_in_data_and_outputs_in_cnn(self):
        root = Path(experiment.__file__).resolve().parents[1]
        cases = [
            ("prepare", "prepare", []),
            ("benchmark", "benchmark", []),
            ("train-full", "train_full", ["--epochs", "12"]),
            ("submit", "submit", ["--checkpoint", "model.pt"]),
        ]
        for command, handler, flags in cases:
            with self.subTest(command=command), patch.object(experiment, handler) as run, patch("sys.argv", ["cnn", command, *flags]):
                experiment.main()
                args = run.call_args.args[0]
                self.assertEqual(args.output_dir, root / "cnn/outputs")
                if command != "submit":
                    self.assertEqual(args.train, root / "data/train.csv")
                if command in ("train-full", "submit"):
                    self.assertEqual(args.test, root / "data/test.csv")

    def setUp(self):
        self.frame = pd.DataFrame({
            "id": np.arange(1000, 1200),
            "text": [f"passage {i}" for i in range(200)],
            "label": [0, 1] * 100,
        })
        self.groups = np.repeat(np.arange(10), 20)

    def test_random_split_matches_shared_split_and_topics_never_cross_folds(self):
        splits = experiment.assign_splits(self.frame, self.groups)
        again = experiment.assign_splits(self.frame, self.groups)
        pd.testing.assert_frame_equal(splits, again)
        _, val = train_test_split(self.frame, test_size=0.2, random_state=12, stratify=self.frame.label)
        self.assertEqual(set(val.id), set(splits.loc[splits.random_split == "validation", "id"]))
        self.assertEqual(set(splits.topic_fold), set(range(5)))
        self.assertTrue((splits.groupby("topic_group").topic_fold.nunique() == 1).all())
        for fold in range(5):
            held_out = splits.topic_fold == fold
            self.assertFalse(set(splits.topic_group[held_out]) & set(splits.topic_group[~held_out]))

    def test_changed_dataset_cannot_silently_reuse_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            self.frame.to_csv(source, index=False)
            experiment.assign_splits(self.frame, self.groups).to_csv(root / "splits.csv", index=False)
            metadata = {
                "train_sha256": experiment.file_hash(source), "folds": 5,
                "sklearn_version": experiment.sklearn.__version__,
            }
            (root / "split_metadata.json").write_text(json.dumps(metadata))
            frame, splits, _ = experiment.load_splits(source, root)
            self.assertEqual(frame.id.tolist(), splits.id.tolist())
            self.frame.assign(text="changed text").to_csv(source, index=False)
            with self.assertRaisesRegex(ValueError, "changed since prepare"):
                experiment.load_splits(source, root)

    def test_cleaning_ablation_changes_only_cleaner(self):
        raw = SequenceEncoder("cnn_raw", 100, 32)
        cleaned = SequenceEncoder("cnn_clean", 100, 32)
        self.assertEqual(raw.tokenize("Hello WORLD!"), cleaned.tokenize("Hello WORLD!"))
        self.assertEqual(raw.tokenize("Hello\n\nWORLD"), ["Hello", "\n", "\n", "WORLD"])
        self.assertEqual(cleaned.tokenize("Hello\n\nWORLD"), ["Hello", "WORLD"])
        self.assertIn("123456", raw.tokenize("Number 123456"))
        self.assertNotIn("123456", cleaned.tokenize("Number 123456"))

    def test_vocabulary_does_not_learn_from_validation_texts(self):
        encoder = SequenceEncoder("cnn_raw", 100, 8).fit(["training words"])
        before = dict(encoder.vocabulary)
        encoded = encoder.encode(["validationonly"])
        self.assertEqual(encoder.vocabulary, before)
        self.assertEqual(encoded[0, 0], 1)
        self.assertTrue((encoded[0, 1:] == 0).all())

    def test_inner_topic_holdout_has_no_topic_overlap(self):
        train, stop = experiment.inner_split(self.frame.label, self.groups, "topic", 12)
        self.assertFalse(set(self.groups[train]) & set(self.groups[stop]))
        self.assertEqual(set(train) | set(stop), set(range(len(self.frame))))

    def test_single_class_and_small_topics_excluded_from_summary(self):
        scores = experiment.topic_scores(
            [0, 0, 0, 1, 0, 1], [0.1, 0.2, 0.1, 0.9, 0.2, 0.8],
            [0, 0, 1, 1, 1, 1], minimum_per_class=2,
        ).set_index("topic_group")
        self.assertTrue(np.isnan(scores.loc[0, "auroc"]))
        self.assertFalse(scores.loc[0, "eligible_for_summary"])
        self.assertTrue(scores.loc[1, "eligible_for_summary"])
        self.assertEqual(scores.loc[1, "auroc"], 1.0)

    def test_submission_preserves_test_id_order_and_records_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, test = root / "model.pt", root / "test.csv"
            checkpoint.write_bytes(b"mock checkpoint")
            pd.DataFrame({"id": [9, 3], "text": ["first passage", "second passage"]}).to_csv(test, index=False)
            fake = Mock()
            fake.predict.return_value = np.array([0.2, 0.7])
            fake.config = CNNConfig(device="cpu")
            fake.encoder.mode = "cnn_raw"
            with patch.object(experiment, "load_predictor", return_value=fake) as load, patch.object(experiment, "refit", side_effect=AssertionError("Submission must never train")) as refit:
                experiment.submit(Namespace(checkpoint=checkpoint, test=test, output_dir=root, device="cpu"))
            destination = next(root.glob("cnn_prediction_*"))
            saved = pd.read_csv(destination / "submission_cnn_raw.csv")
            self.assertEqual(list(saved.columns), ["id", "label"])
            self.assertEqual(saved.id.tolist(), [9, 3])
            self.assertEqual(saved.label.tolist(), [0.2, 0.7])
            metadata = json.loads((destination / "submission_cnn_raw.metadata.json").read_text())
            self.assertTrue(metadata["inference_only"])
            self.assertEqual(metadata["checkpoint_sha256"], experiment.file_hash(checkpoint))
            load.assert_called_once_with(checkpoint, device="cpu")
            refit.assert_not_called()

    def test_full_training_uses_every_labeled_row_and_only_predicts_test_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.csv", root / "test.csv"
            self.frame.to_csv(train, index=False)
            pd.DataFrame({"id": [9, 3], "text": ["first passage", "second passage"]}).to_csv(test, index=False)
            predictor = Mock()
            predictor.encoder.mode = "cnn_raw"
            predictor.predict.return_value = np.array([0.2, 0.7])

            def fake_refit(texts, labels, mode, config, epochs, destination):
                self.assertEqual(texts.tolist(), self.frame.text.tolist())
                self.assertEqual(labels.tolist(), self.frame.label.tolist())
                self.assertEqual(epochs, 12)
                predictor.config = config
                (destination / "model.pt").write_bytes(b"mock trained checkpoint")
                return predictor

            with patch.object(experiment, "refit", side_effect=fake_refit) as refit, patch.object(experiment, "choose_epoch", side_effect=AssertionError("Full training needs a fixed epoch count")), patch.object(experiment, "load_splits", side_effect=AssertionError("Full training must not discard held-out rows")):
                path = experiment.train_full(self.options(
                    train=train, test=test, output_dir=root, model="cnn_raw", epochs=12,
                ))
            refit.assert_called_once()
            self.assertEqual(pd.read_csv(path).id.tolist(), [9, 3])
            self.assertEqual(pd.read_csv(path.parent / "training_ids.csv").id.tolist(), self.frame.id.tolist())
            metadata = json.loads(path.with_suffix(".metadata.json").read_text())
            self.assertFalse(metadata["inference_only"])
            self.assertEqual(metadata["training"]["training_rows"], len(self.frame))
            self.assertEqual(metadata["training"]["epochs"], 12)
            self.assertIsNone(metadata["training"]["validation_auroc"])
            self.assertFalse((path.parent / "metrics.csv").exists())
            predictor.predict.assert_called_once()
            self.assertEqual(predictor.predict.call_args.args[0].tolist(), ["first passage", "second passage"])

    @staticmethod
    def options(**kwargs):
        settings = dict(
            seed=12, epochs=30, patience=3, max_vocab=100, max_length=32,
            embedding_dim=128, embeddings=None, lowercase_embedding_fallback=False,
            embedding_modes=["random"], embedding_mode="random",
            batch_size=8, learning_rate=0.001, device="cpu",
            models=["cnn_raw"], fold="0", protocol="topic",
        )
        settings.update(kwargs)
        return Namespace(**settings)

    def test_benchmark_uses_inner_selection_and_outer_refit_without_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            self.frame.to_csv(source, index=False)
            splits = experiment.assign_splits(self.frame, self.groups)
            splits.to_csv(root / "splits.csv", index=False)
            (root / "split_metadata.json").write_text(json.dumps({
                "train_sha256": experiment.file_hash(source), "folds": 5,
                "sklearn_version": experiment.sklearn.__version__,
            }))
            outer_val = set(self.frame.text[splits.topic_fold == 0])
            outer_train = set(self.frame.text) - outer_val

            def fake_select(texts, labels, stopping_text, stopping_labels, mode, config, destination):
                self.assertFalse(set(texts) & set(stopping_text))
                self.assertEqual(set(texts) | set(stopping_text), outer_train)
                return 3

            def fake_refit(texts, labels, mode, config, epochs, destination):
                self.assertEqual(set(texts), outer_train)
                self.assertEqual(epochs, 3)
                predictor = Mock()

                def predict(texts):
                    self.assertEqual(set(texts), outer_val)
                    return np.linspace(0.1, 0.9, len(texts))

                predictor.predict.side_effect = predict
                return predictor

            with patch.object(experiment, "choose_epoch", side_effect=fake_select), patch.object(experiment, "refit", side_effect=fake_refit):
                experiment.benchmark(self.options(train=source, output_dir=root))
            destination = next(root.glob("cnn_benchmark_*"))
            saved = pd.read_csv(destination / "metrics.csv")
            self.assertEqual(saved.selected_epoch.tolist(), [3])
            self.assertEqual(saved.validation_rows.tolist(), [len(outer_val)])


if __name__ == "__main__":
    unittest.main()
