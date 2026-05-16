import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from twinflow_porting_harness.cli import main


class CliTest(unittest.TestCase):
    def test_missing_inputs_refuse_to_start(self) -> None:
        out = StringIO()
        with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
            code = main(["init", "--cluster", "slurm", "--gpu-count", "8", "--condition-type", "image", "--run-name", "x"])
        self.assertEqual(code, 2)
        text = out.getvalue()
        self.assertIn("Do not start code changes", text)
        self.assertIn("dataset_root", text)
        self.assertIn("slurm_partition or sbatch_template", text)

    def test_init_only_prompts_when_ask_is_explicit(self) -> None:
        out = StringIO()
        with patch("sys.stdin", Mock(isatty=lambda: True)), redirect_stdout(out):
            code = main(["init", "--cluster", "slurm", "--gpu-count", "8", "--condition-type", "image", "--run-name", "x"])
        self.assertEqual(code, 2)
        self.assertIn("Do not start code changes", out.getvalue())

    def test_validate_rejects_whitespace_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            values = argv_to_values(argv)
            values["eval_sample_spec"] = "   "
            contract = root / "INPUT_CONTRACT.json"
            contract.write_text(json.dumps(values), encoding="utf-8")

            out = StringIO()
            with redirect_stdout(out):
                code = main(["validate", str(contract)])

        self.assertEqual(code, 2)
        self.assertIn("eval_sample_spec", out.getvalue())

    def test_complete_dry_run_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            env = root / "env"
            outputs = root / "outputs"
            logs = root / "logs"
            for path in (data, env, outputs, logs):
                path.mkdir()
            train = data / "train.csv"
            eval_file = data / "eval.csv"
            ckpt = root / "base.pt"
            train.write_text("id\n", encoding="utf-8")
            eval_file.write_text("id\n", encoding="utf-8")
            ckpt.write_bytes(b"placeholder")

            out = StringIO()
            with redirect_stdout(out):
                code = main(
                    [
                        "init",
                        "--dry-run",
                        "--project-root",
                        str(root),
                        "--dataset-root",
                        str(data),
                        "--train-metadata",
                        str(train),
                        "--eval-metadata",
                        str(eval_file),
                        "--output-root",
                        str(outputs),
                        "--run-name",
                        "unit",
                        "--base-checkpoint",
                        str(ckpt),
                        "--conda-env",
                        str(env),
                        "--official-twinflow-ref",
                        "https://github.com/inclusionAI/TwinFlow",
                        "--original-inference-entry",
                        "python example.py",
                        "--condition-type",
                        "image",
                        "--cluster",
                        "slurm",
                        "--gpu-count",
                        "8",
                        "--eval-sample-spec",
                        "indices 0 1, seed 42",
                        "--slurm-log-dir",
                        str(logs),
                        "--slurm-partition",
                        "gpu",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())

    def test_dry_run_create_output_root_does_not_create_dirs_or_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            outputs = root / "outputs"
            logs = root / "logs"
            outputs.rmdir()
            logs.rmdir()
            argv.insert(2, "--create-output-root")

            out = StringIO()
            with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                code = main(argv)

            self.assertEqual(code, 2)
            self.assertIn("output_root", out.getvalue())
            self.assertIn("slurm_log_dir", out.getvalue())
            self.assertFalse(outputs.exists())
            self.assertFalse(logs.exists())

    def test_create_output_root_creates_dirs_for_real_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            outputs = root / "outputs"
            logs = root / "logs"
            outputs.rmdir()
            logs.rmdir()
            argv.remove("--dry-run")
            argv.insert(1, "--create-output-root")

            out = StringIO()
            with redirect_stdout(out):
                code = main(argv)

            self.assertEqual(code, 0)
            self.assertTrue(outputs.is_dir())
            self.assertTrue(logs.is_dir())
            self.assertTrue((root / "handoff" / "twinflow_unit" / "INPUT_CONTRACT.json").is_file())

    def test_dataset_root_must_be_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            train_path = argv[argv.index("--train-metadata") + 1]
            argv[argv.index("--dataset-root") + 1] = train_path

            out = StringIO()
            with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 2)
        self.assertIn("dataset_root", out.getvalue())
        self.assertIn("directory does not exist", out.getvalue())

    def test_home_relative_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = complete_args(Path(tmp))
            argv[argv.index("--project-root") + 1] = "~/not-absolute-enough"

            out = StringIO()
            with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 2)
        self.assertIn("project_root", out.getvalue())
        self.assertIn("path must be absolute", out.getvalue())

    def test_malformed_twinflow_urls_are_rejected(self) -> None:
        bad_refs = [
            "https:/github.com/inclusionAI/TwinFlow",
            "https://",
            "https://github.com",
            "git@github.com:",
        ]
        for ref in bad_refs:
            with self.subTest(ref=ref), tempfile.TemporaryDirectory() as tmp:
                argv = complete_args(Path(tmp))
                argv[argv.index("--official-twinflow-ref") + 1] = ref

                out = StringIO()
                with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                    code = main(argv)

                self.assertEqual(code, 2)
                self.assertIn("official_twinflow_ref", out.getvalue())
                self.assertIn("invalid:official_twinflow_ref", out.getvalue())

    def test_local_twinflow_ref_must_be_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            local_file = root / "twinflow_ref.txt"
            local_file.write_text("not a repository\n", encoding="utf-8")
            argv[argv.index("--official-twinflow-ref") + 1] = str(local_file)

            out = StringIO()
            with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 2)
        self.assertIn("official_twinflow_ref", out.getvalue())
        self.assertIn("must be a directory", out.getvalue())

    def test_local_twinflow_ref_must_be_git_clone(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            local_dir = root / "not_twinflow"
            local_dir.mkdir()
            argv[argv.index("--official-twinflow-ref") + 1] = str(local_dir)

            out = StringIO()
            with patch("sys.stdin", Mock(isatty=lambda: False)), redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 2)
        self.assertIn("official_twinflow_ref", out.getvalue())
        self.assertIn("git clone", out.getvalue())

    def test_local_twinflow_ref_accepts_git_worktree_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            local_dir = root / "twinflow"
            local_dir.mkdir()
            (local_dir / ".git").mkdir()
            argv[argv.index("--official-twinflow-ref") + 1] = str(local_dir)

            out = StringIO()
            with redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())

    def test_ask_accepts_partition_without_sbatch_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = complete_args(Path(tmp))
            argv.remove("--dry-run")
            argv.insert(1, "--dry-run")
            argv.insert(1, "--ask")
            idx = argv.index("--slurm-partition")
            del argv[idx : idx + 2]

            out = StringIO()
            with patch("builtins.input", side_effect=["gpu"]), redirect_stdout(out):
                code = main(argv)
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())

    def test_ask_allows_partition_when_sbatch_template_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            idx = argv.index("--slurm-partition")
            del argv[idx : idx + 2]
            argv.extend(["--sbatch-template", str(root / "missing.sh")])
            argv.insert(1, "--ask")

            out = StringIO()
            with patch("builtins.input", side_effect=["gpu"]), redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())

    def test_ask_drops_invalid_sbatch_template_when_partition_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            argv = complete_args(root)
            argv.extend(["--sbatch-template", str(root / "missing.sh")])
            argv.insert(1, "--ask")

            out = StringIO()
            with redirect_stdout(out):
                code = main(argv)

        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())

    def test_ask_repairs_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = complete_args(Path(tmp))
            argv[argv.index("--cluster") + 1] = "Slurm"
            argv[argv.index("--gpu-count") + 1] = "0"
            idx = argv.index("--slurm-partition")
            del argv[idx : idx + 2]
            argv.insert(1, "--ask")

            out = StringIO()
            with patch("builtins.input", side_effect=["slurm", "8", "gpu"]), redirect_stdout(out):
                code = main(argv)
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN OK", out.getvalue())


def complete_args(root: Path) -> list[str]:
    data = root / "data"
    env = root / "env"
    outputs = root / "outputs"
    logs = root / "logs"
    for path in (data, env, outputs, logs):
        path.mkdir()
    train = data / "train.csv"
    eval_file = data / "eval.csv"
    ckpt = root / "base.pt"
    train.write_text("id\n", encoding="utf-8")
    eval_file.write_text("id\n", encoding="utf-8")
    ckpt.write_bytes(b"placeholder")
    return [
        "init",
        "--dry-run",
        "--project-root",
        str(root),
        "--dataset-root",
        str(data),
        "--train-metadata",
        str(train),
        "--eval-metadata",
        str(eval_file),
        "--output-root",
        str(outputs),
        "--run-name",
        "unit",
        "--base-checkpoint",
        str(ckpt),
        "--conda-env",
        str(env),
        "--official-twinflow-ref",
        "https://github.com/inclusionAI/TwinFlow",
        "--original-inference-entry",
        "python example.py",
        "--condition-type",
        "image",
        "--cluster",
        "slurm",
        "--gpu-count",
        "8",
        "--eval-sample-spec",
        "indices 0 1, seed 42",
        "--slurm-log-dir",
        str(logs),
        "--slurm-partition",
        "gpu",
    ]


def argv_to_values(argv: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    it = iter(argv[2:])
    for key, value in zip(it, it):
        values[key.lstrip("-").replace("-", "_")] = value
    return values


if __name__ == "__main__":
    unittest.main()
