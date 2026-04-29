import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "00_download_data.py"


def load_module():
    spec = importlib.util.spec_from_file_location("download_data", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DownloadDataTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_parse_args_uses_hm_competition_defaults(self):
        args = self.module.parse_args([])

        self.assertEqual(
            args.competition,
            "h-and-m-personalized-fashion-recommendations",
        )
        self.assertEqual(args.data_dir, Path("data/raw"))
        self.assertTrue(args.unzip)
        self.assertFalse(args.force)

    def test_competition_target_dir_lives_under_data_dir(self):
        target_dir = self.module.competition_target_dir(
            Path("data/raw"),
            "h-and-m-personalized-fashion-recommendations",
        )

        self.assertEqual(
            target_dir,
            Path("data/raw/h-and-m-personalized-fashion-recommendations"),
        )

    def test_should_skip_download_when_destination_has_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "dataset"
            destination.mkdir()
            (destination / "transactions_train.csv").write_text("data")

            self.assertTrue(self.module.should_skip_download(destination, force=False))
            self.assertFalse(self.module.should_skip_download(destination, force=True))

    def test_download_competition_calls_kagglehub_with_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calls = []

            def downloader(competition, output_dir, force_download):
                calls.append((competition, output_dir, force_download))
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                return str(destination)

            destination = self.module.download_competition(
                competition="h-and-m-personalized-fashion-recommendations",
                data_dir=Path(temp_dir),
                unzip=False,
                force=True,
                downloader=downloader,
            )

        self.assertEqual(
            calls,
            [
                (
                    "h-and-m-personalized-fashion-recommendations",
                    str(Path(temp_dir) / "h-and-m-personalized-fashion-recommendations"),
                    True,
                ),
            ],
        )
        self.assertEqual(
            destination,
            Path(temp_dir) / "h-and-m-personalized-fashion-recommendations",
        )

    def test_download_competition_extracts_zip_returned_by_kagglehub(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            def downloader(competition, output_dir, force_download):
                destination = Path(output_dir)
                destination.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(destination / "dataset.zip", "w") as archive:
                    archive.writestr("sample_submission.csv", "customer_id,prediction\n")
                return str(destination)

            destination = self.module.download_competition(
                competition="h-and-m-personalized-fashion-recommendations",
                data_dir=Path(temp_dir),
                unzip=True,
                force=False,
                downloader=downloader,
            )

            self.assertTrue((destination / "dataset.zip").exists())
            self.assertTrue((destination / "sample_submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
