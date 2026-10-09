import unittest

import torch

from networks.quantizers.auto_grvq import AutoGroupResidualVectorQuantize
from networks.semantic_codec.LongCatAudioCodec_model import LongCatAudioCodecDecoder


class NullAcousticTest(unittest.TestCase):
    def test_training_dropout_can_return_an_all_zero_acoustic_latent(self):
        quantizer = AutoGroupResidualVectorQuantize(
            input_dim=4,
            n_codebooks=3,
            codebook_size=8,
            codebook_dim=2,
            null_acoustic_probability=1.0,
        )
        quantizer.train()

        acoustic, codes, _, _, _, active = quantizer(
            torch.randn(2, 4, 5),
            return_active_codebooks=True,
        )

        self.assertEqual(tuple(codes.shape), (2, 3, 5))
        self.assertTrue(torch.equal(active, torch.zeros(2, dtype=torch.long)))
        self.assertTrue(torch.equal(acoustic, torch.zeros_like(acoustic)))

    def test_decoder_null_acoustic_keeps_native_input_channels(self):
        decoder = LongCatAudioCodecDecoder(
            latent_dim=4,
            decoder_dim=8,
            decoder_rates=[2],
            semantic_dim=3,
            decoder_type="16k",
            n_codebooks=3,
            codebook_size=8,
            codebook_dim=2,
            semantic_token_nums=5,
        )

        merged = decoder._merge_latents(torch.zeros(2, 5, dtype=torch.long), None)

        self.assertEqual(tuple(merged.shape), (2, 7, 5))
        self.assertTrue(torch.equal(merged[:, 3:], torch.zeros(2, 4, 5)))


if __name__ == "__main__":
    unittest.main()
