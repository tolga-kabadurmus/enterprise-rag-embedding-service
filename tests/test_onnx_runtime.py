"""Unit tests for E5 prefixing, pooling, and normalization."""

from pathlib import Path
import unittest

import numpy as np

from app.common.settings.model import ModelSettings
from app.repository.embedding.onnx_runtime import OnnxEmbeddingRuntime


class FakeEncoding:
    """Small tokenizers.Encoding substitute."""

    def __init__(self, ids: list[int], mask: list[int], type_ids: list[int]) -> None:
        self.ids = ids
        self.attention_mask = mask
        self.type_ids = type_ids


class FakeTokenizer:
    """Record prepared text and return fixed padded encodings."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_batch(self, texts: list[str]) -> list[FakeEncoding]:
        self.texts = texts
        return [
            FakeEncoding([1, 2], [1, 1], [0, 0])
            for _ in texts
        ]


class FakeInput:
    """ONNX input descriptor."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeOutput:
    """ONNX output descriptor."""

    shape = [None, None, 3]


class FakeSession:
    """Return deterministic token-level embeddings."""

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput("input_ids"), FakeInput("attention_mask")]

    def get_outputs(self) -> list[FakeOutput]:
        return [FakeOutput()]

    def run(self, names, inputs):  # type: ignore[no-untyped-def]
        self.inputs = inputs
        return [
            np.asarray(
                [
                    [[3.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                ]
            )
        ]


class OnnxRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Verify E5 input prefixes and normalized mean pooling."""

    async def test_document_prefix_and_normalized_pooling(self) -> None:
        tokenizer = FakeTokenizer()
        runtime = OnnxEmbeddingRuntime(
            FakeSession(),
            tokenizer,
            ModelSettings(
                local_directory=Path("/models/embedding/intfloat/multilingual-e5-small")
            ),
        )

        vectors = await runtime.embed(["chunk"], "document")

        self.assertEqual(tokenizer.texts, ["passage: chunk"])
        self.assertAlmostEqual(vectors[0][0], 1.0)
        self.assertAlmostEqual(vectors[0][1], 0.0)
        self.assertAlmostEqual(vectors[0][2], 0.0)
        self.assertAlmostEqual(sum(value * value for value in vectors[0]), 1.0)

    async def test_query_prefix(self) -> None:
        tokenizer = FakeTokenizer()
        runtime = OnnxEmbeddingRuntime(
            FakeSession(),
            tokenizer,
            ModelSettings(),
        )

        await runtime.embed(["question"], "query")

        self.assertEqual(tokenizer.texts, ["query: question"])


if __name__ == "__main__":
    unittest.main()
