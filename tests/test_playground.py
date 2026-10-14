import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from export_checkpoint_bundles import ALIGNMENT_SCORERS, MASS_SCALES, export
from download_prepared_data import _safe_extract, _valid_dataset
from models.MSAlign import MSAlign as AlignmentModel
from playground.bundle import load_manifest, sha256_file
from playground.inference import (
    _adduct_offset,
    _binned_spectrum,
    _canonical_smiles,
    _combine_channels,
    _validate_spectrum,
)
from transforms.spectra_transforms import BIN_Transform


class BundleTests(unittest.TestCase):
    def test_raw_manifest_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"weights")
            config = root / "model.yaml"
            config.write_text("model: test\n")
            manifest = {
                "format_version": 1,
                "model_type": "raw",
                "external_models": {
                    "moldeberta": {"revision": "test-revision"}
                },
                "scorers": [{
                    "name": "dreams_moldeberta",
                    "type": "msalign",
                    "checkpoint": "model.ckpt",
                    "sha256": sha256_file(checkpoint),
                    "config": "model.yaml",
                    "config_sha256": sha256_file(config),
                }],
            }
            (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
            loaded_root, loaded = load_manifest(root)
            self.assertEqual(loaded_root, root)
            self.assertEqual(loaded["model_type"], "raw")

            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_manifest(root)

    def test_spectrum_input_and_simple_representations(self):
        peaks, precursor, adduct, energy = _validate_spectrum({
            "peaks": [[10.0, 1.0], [20.0, 0.5]],
            "precursor_mz": 100.0,
        })
        self.assertEqual(precursor, 100.0)
        self.assertIsNone(adduct)
        self.assertTrue(np.isnan(energy))
        bins = _binned_spectrum(peaks, max_mz=100.0, bin_width=1.0)
        self.assertEqual(bins.shape, (100,))
        self.assertAlmostEqual(float(bins.max()), 1.0, places=6)
        self.assertTrue(np.isnan(_adduct_offset(None)))
        self.assertTrue(np.isfinite(_adduct_offset("[M+H]+")))

    def test_nan_collision_energy_is_missing(self):
        spectrum = {
            "peaks": [[10.0, 1.0], [20.0, 0.5]],
            "precursor_mz": 100.0,
        }
        for missing in (None, float("nan"), np.float32("nan"), "nan"):
            spectrum["collision_energy"] = missing
            *_, energy = _validate_spectrum(spectrum)
            self.assertTrue(np.isnan(energy))

        for invalid in (float("inf"), float("-inf")):
            spectrum["collision_energy"] = invalid
            with self.assertRaisesRegex(ValueError, "finite, NaN, or None"):
                _validate_spectrum(spectrum)

    def test_binning_matches_training_transform_and_uses_all_peaks(self):
        peaks = np.column_stack((np.arange(150) + 0.25, np.arange(150) + 1.0))
        expected = BIN_Transform(max_mz=200.0, bin_width=1.0)(peaks)
        actual = _binned_spectrum(peaks, max_mz=200.0, bin_width=1.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertGreater(actual[0], 0.0)

    def test_smiles_canonicalization_preserves_stereochemistry(self):
        canonical, _ = _canonical_smiles(["F[C@H](Cl)Br", "F[C@@H](Cl)Br"])
        self.assertNotEqual(canonical[0], canonical[1])

    def test_raw_and_fused_channel_combination(self):
        channels = {
            "first": torch.tensor([1.0, 2.0]),
            "second": torch.tensor([5.0, 7.0]),
        }
        raw = {"scorers": [{"name": "first"}]}
        fused = {
            "scorers": [{"name": "first"}, {"name": "second"}],
            "weights": {"first": 0.25, "second": 0.75},
        }
        torch.testing.assert_close(_combine_channels(raw, channels), channels["first"])
        torch.testing.assert_close(
            _combine_channels(fused, channels), torch.tensor([4.0, 5.75])
        )


class DownloaderTests(unittest.TestCase):
    def test_dataset_completeness_and_zip_traversal_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "massspecgym"
            for relative in (
                "metadata.csv",
                "unique_smiles.csv",
                "spectra.npy",
                "splits/formula_seed0.csv",
                "splits/mces_1.csv",
                "candidates/official_candidates_by_mass/map.json",
            ):
                path = dataset / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(_valid_dataset(dataset, "massspecgym"))

            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../outside.txt", "unsafe")
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                _safe_extract(archive, root / "extract")


class ExportTests(unittest.TestCase):
    @staticmethod
    def _config(spectrum, molecule):
        return {
            "representations": {"spectrum": spectrum, "molecule": molecule},
            "model": {
                "d_shared": 4,
                "d_hidden": 6,
                "d_metadata": 4,
                "n_hidden_layers_ms_pre": 0,
                "n_hidden_layers_ms_post": 1,
                "n_hidden_layers_mol": 1,
                "dropout": 0.0,
                "temperature_init": 0.07,
                "use_collision_energy": True,
                "use_adduct": True,
            },
            "training": {
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "n_warmup_steps": 1,
                "n_max_steps": 2,
            },
            "spectrum_bins": {"max_mz": 10.0, "bin_width": 1.0},
        }

    def test_export_writes_loadable_raw_and_fused_archives(self):
        pairs = {
            "dreams_moldeberta": ("dreams", "moldeberta_base_123m_mtr"),
            "dreams_fingerprint": ("dreams", "morgan_2_4096"),
            "bins_moldeberta": ("bins", "moldeberta_base_123m_mtr"),
            "bins_fingerprint": ("bins", "morgan_2_4096"),
        }
        dimensions = {
            "d_ms": 5,
            "d_mol": 7,
            "n_adducts": 2,
            "unknown_adduct_index": 1,
            "adduct_signs": [1, 0],
            "adduct_vocabulary": ["[M+H]+"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "checkpoints"
            filename = "massspecgym__official_candidates_by_mass__formula_seed0.ckpt"
            for name, (spectrum, molecule) in pairs.items():
                config = self._config(spectrum, molecule)
                model = AlignmentModel(config, dimensions)
                path = checkpoints / name / filename
                path.parent.mkdir(parents=True)
                torch.save({
                    "state_dict": model.state_dict(),
                    "hyper_parameters": {
                        "config": config,
                        "input_dimensions": dimensions,
                    },
                    "optimizer_states": [{"not": "released"}],
                }, path)

            names = list(ALIGNMENT_SCORERS) + [f"mass_ppm_{scale:g}" for scale in MASS_SCALES]
            weights = {name: 1.0 / len(names) for name in names}
            results = root / "fusion.csv"
            fields = [
                "labelled_dataset_name", "candidate_map_name", "split_method",
                "selected_temperature", "name", "alpha", "R@1", "R@5", "R@20",
            ]
            with results.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "labelled_dataset_name": "massspecgym",
                    "candidate_map_name": "official_candidates_by_mass",
                    "split_method": "formula_seed0",
                    "selected_temperature": 0.1,
                    "name": "MSAlign_fusion",
                    "alpha": json.dumps(weights),
                    "R@1": 0.5, "R@5": 0.7, "R@20": 0.9,
                })
                for name in ALIGNMENT_SCORERS:
                    writer.writerow({
                        "labelled_dataset_name": "massspecgym",
                        "candidate_map_name": "official_candidates_by_mass",
                        "split_method": "formula_seed0",
                        "selected_temperature": 0.1,
                        "name": name,
                        "alpha": weights[name],
                        "R@1": 0.4, "R@5": 0.6, "R@20": 0.8,
                    })

            output = root / "releases"
            entries = export(SimpleNamespace(
                split="formula_seed0",
                fusion_results=results,
                checkpoints=checkpoints,
                output=output,
                overwrite=False,
            ))
            self.assertEqual(len(entries), 2)
            repeated = export(SimpleNamespace(
                split="formula_seed0",
                fusion_results=results,
                checkpoints=checkpoints,
                output=output,
                overwrite=True,
            ))
            self.assertEqual(
                [entry["sha256"] for entry in entries],
                [entry["sha256"] for entry in repeated],
            )
            for entry in entries:
                archive = output / entry["file"]
                self.assertEqual(sha256_file(archive), entry["sha256"])
                with zipfile.ZipFile(archive) as handle:
                    handle.extractall(root / "extracted")
                _, manifest = load_manifest(root / "extracted" / entry["name"])
                self.assertEqual(manifest["name"], entry["name"])
                self.assertIn("configurations", entry)
                self.assertIn("test_metrics", entry)
                for scorer in manifest["scorers"]:
                    if scorer["type"] != "msalign":
                        continue
                    payload = torch.load(
                        root / "extracted" / entry["name"] / scorer["checkpoint"],
                        map_location="cpu",
                        weights_only=False,
                    )
                    self.assertEqual(
                        set(payload),
                        {"format_version", "config", "input_dimensions", "state_dict"},
                    )


if __name__ == "__main__":
    unittest.main()
