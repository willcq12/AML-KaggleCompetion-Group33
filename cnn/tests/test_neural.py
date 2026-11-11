"""Check checkpoint portability and inference without any optimizer steps."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cnn.neural import (
    CNNConfig, CNNPredictor, initialize, load_predictor, load_word_vectors,
    save_checkpoint, validate_vector_file,
)


class CheckpointTests(unittest.TestCase):
    def test_pretrained_vectors_match_ids_and_freezing_preserves_random_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            vectors = Path(directory) / "vectors.txt"
            vectors.write_text("hello 0.1 0.2\nworld 0.3 0.4\nHello 0.7 0.8\n! 0.9 1.0\n")
            config = CNNConfig(
                max_vocab=20, max_length=8, embedding_dim=2, filters=2,
                embedding_path=str(vectors), embedding_lowercase_fallback=True, device="cpu",
            )
            texts = ["Hello WORLD!\nRareOnly"]
            baseline, _, _ = initialize(texts, "cnn_raw", config)
            frozen, encoder, _ = initialize(texts, "cnn_raw", replace(config, embedding_mode="frozen"))
            finetune, _, _ = initialize(texts, "cnn_raw", replace(config, embedding_mode="finetune"))
            weights = frozen.embedding.weight.detach().numpy()
            np.testing.assert_allclose(weights[encoder.vocabulary["Hello"]], [0.7, 0.8])
            np.testing.assert_allclose(weights[encoder.vocabulary["WORLD"]], [0.3, 0.4])
            np.testing.assert_allclose(weights[0], 0)
            for token in ("\n", "RareOnly"):
                index = encoder.vocabulary[token]
                np.testing.assert_array_equal(weights[index], baseline.embedding.weight.detach().numpy()[index])
            self.assertFalse(frozen.embedding.weight.requires_grad)
            self.assertTrue(finetune.embedding.weight.requires_grad)
            self.assertTrue(frozen.output.weight.requires_grad)
            self.assertEqual(frozen.embedding_report["matched_tokens"], 3)
            self.assertEqual(frozen.embedding_report["lowercase_matches"], 1)
            np.testing.assert_array_equal(frozen.convolutions[0].weight.detach().numpy(), baseline.convolutions[0].weight.detach().numpy())

    def test_pretrained_checkpoint_loads_without_original_vector_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vectors = root / "vectors.txt"
            vectors.write_text("Hello 0.1 0.2\n")
            config = CNNConfig(
                max_vocab=20, max_length=8, embedding_dim=2, filters=2,
                embedding_mode="frozen", embedding_path=str(vectors), device="cpu",
            )
            model, encoder, device = initialize(["Hello!"], "cnn_raw", config)
            expected = CNNPredictor(model, encoder, config, device).predict(["Hello!"])
            checkpoint = root / "model.pt"
            save_checkpoint(checkpoint, model, encoder, config, epoch=1)
            vectors.rename(root / "moved_vectors.txt")
            restored = load_predictor(checkpoint, device="cpu")
            np.testing.assert_allclose(restored.predict(["Hello!"]), expected, atol=1e-7)
            self.assertFalse(restored.model.embedding.weight.requires_grad)

    def test_vector_validation_rejects_wrong_dimensions_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.txt"
            path.write_text("2 2\nhello 0.1 0.2\nworld 0.3 0.4\n")
            validate_vector_file(path, 2)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                validate_vector_file(path, 3)
            found, _ = load_word_vectors(path, {"hello": 2}, 2)
            np.testing.assert_allclose(found[2], [0.1, 0.2])
            path.write_text("hello nan 0.2\n")
            with self.assertRaisesRegex(ValueError, "finite"):
                load_word_vectors(path, {"hello": 2}, 2)

    def test_checkpoint_restores_vocabulary_weights_and_cpu_inference(self):
        config = CNNConfig(max_vocab=20, max_length=8, embedding_dim=4, filters=2, batch_size=2, device="cpu")
        texts = ["Hello\nWORLD!", "Other words"]
        model, encoder, device = initialize(texts, "cnn_raw", config)
        expected = CNNPredictor(model, encoder, config, device).predict(texts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            # A model trained with another device setting must still load on CPU.
            save_checkpoint(path, model, encoder, replace(config, device="cuda"), epoch=1)
            restored = load_predictor(path, device="cpu")
            self.assertEqual(restored.encoder.vocabulary, encoder.vocabulary)
            np.testing.assert_allclose(restored.predict(texts), expected, atol=1e-7)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())


if __name__ == "__main__":
    unittest.main()
