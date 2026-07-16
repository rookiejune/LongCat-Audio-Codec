import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Union, get_type_hints

import torch
import yaml

from longcat_audio_codec import (
    checkpoint_dir_from_env,
    load_decoder,
    resolve_checkpoint_path,
    resolve_resource_path,
)
from networks.semantic_codec.LongCatAudioCodec_model import LongCatAudioCodecDecoder


class ModelLoadTest(unittest.TestCase):
    def test_public_loader_reads_decoder_checkpoint(self):
        codec_config = {
            "codec_dimension": 4,
            "codec_dec_ratios": [2],
            "decoder_dim": 4,
            "semantic_dim": 4,
            "decoder_type": "16k",
            "codec_codebook_size": 3,
            "codec_codebook_search_dim": 2,
            "codec_codebooks": 1,
            "semantic_token_nums": 5,
        }
        expected = LongCatAudioCodecDecoder(
            latent_dim=codec_config["codec_dimension"],
            decoder_dim=codec_config["decoder_dim"],
            decoder_rates=codec_config["codec_dec_ratios"],
            semantic_dim=codec_config["semantic_dim"],
            decoder_type=codec_config["decoder_type"],
            n_codebooks=codec_config["codec_codebooks"],
            codebook_size=codec_config["codec_codebook_size"],
            codebook_dim=codec_config["codec_codebook_search_dim"],
            semantic_token_nums=codec_config["semantic_token_nums"],
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "decoder.pt"
            config = root / "decoder.yaml"
            torch.save(expected.state_dict(), checkpoint)
            codec_config["ckpt_path"] = str(checkpoint)
            config.write_text(
                yaml.safe_dump({"codec_config": codec_config}),
                encoding="utf-8",
            )

            loaded = load_decoder(str(config), torch.device("cpu"))

        self.assertIsInstance(loaded, LongCatAudioCodecDecoder)
        self.assertFalse(loaded.training)
        self.assertEqual(loaded.output_rate, 16000)
        self.assertTrue(
            torch.equal(
                loaded.semantic_dequantizer.dequant.weight,
                expected.semantic_dequantizer.dequant.weight,
            )
        )

    def test_public_path_type_hints_are_resolvable(self):
        self.assertEqual(
            get_type_hints(checkpoint_dir_from_env),
            {"return": Optional[Path]},
        )
        path_hints = {
            "path": Union[str, os.PathLike[str]],
            "return": str,
        }
        self.assertEqual(get_type_hints(resolve_checkpoint_path), path_hints)
        self.assertEqual(get_type_hints(resolve_resource_path), path_hints)


if __name__ == "__main__":
    unittest.main()
