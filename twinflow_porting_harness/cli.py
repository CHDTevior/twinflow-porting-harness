from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


CHOICES = {
    "condition_type": ("image", "text", "multiview", "control", "other"),
    "cluster": ("local", "slurm", "ddp", "fsdp", "other"),
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    prompt: str
    help: str
    kind: str = "text"
    required: bool = True
    default: str | None = None


FIELD_SPECS = [
    FieldSpec(
        "project_root",
        "目标项目根目录的绝对路径",
        "Existing absolute path to the project that will receive TwinFlow changes.",
        "existing_dir",
    ),
    FieldSpec(
        "dataset_root",
        "数据集根目录的绝对路径",
        "Existing absolute path to the dataset storage root.",
        "existing_dir",
    ),
    FieldSpec(
        "train_metadata",
        "训练 metadata/CSV/manifest 的绝对路径",
        "Existing absolute path to the train split metadata.",
        "existing_path",
    ),
    FieldSpec(
        "eval_metadata",
        "评估 metadata/CSV/manifest 的绝对路径",
        "Existing absolute path to the eval split metadata.",
        "existing_path",
    ),
    FieldSpec(
        "output_root",
        "实验输出根目录的绝对路径",
        "Absolute output root for checkpoints, samples, eval artifacts, and logs.",
        "output_dir",
    ),
    FieldSpec(
        "run_name",
        "本次迁移 run 名称",
        "Short slug used for output subdirectories and handoff files.",
        "slug",
    ),
    FieldSpec(
        "base_checkpoint",
        "原始预训练 checkpoint 的绝对路径",
        "Existing absolute path to the pretrained model checkpoint.",
        "existing_file",
    ),
    FieldSpec(
        "conda_env",
        "私有 conda 环境目录的绝对路径",
        "Existing absolute path to the copied/private conda environment.",
        "existing_dir",
    ),
    FieldSpec(
        "official_twinflow_ref",
        "官方 TwinFlow repo/commit 或本地 clone 绝对路径",
        "TwinFlow source reference used for call-chain comparison.",
        "url_or_path",
    ),
    FieldSpec(
        "original_inference_entry",
        "原项目原始推理入口或命令",
        "Path or command for the original/pretrained inference path.",
        "text",
    ),
    FieldSpec(
        "condition_type",
        "条件类型 image/text/multiview/control/other",
        "Conditioning modality used by the base project.",
        "choice",
    ),
    FieldSpec(
        "cluster",
        "运行环境 local/slurm/ddp/fsdp/other",
        "Training launch environment.",
        "choice",
    ),
    FieldSpec(
        "gpu_count",
        "计划使用 GPU 数量",
        "Positive integer GPU count for smoke/formal runs.",
        "positive_int",
    ),
    FieldSpec(
        "eval_sample_spec",
        "固定评估样本说明",
        "Sample indices, split path, or deterministic sampling rule for protocol eval.",
        "text",
    ),
    FieldSpec(
        "slurm_log_dir",
        "Slurm 日志目录的绝对路径",
        "Absolute path for sbatch stdout/stderr logs; required when cluster=slurm.",
        "output_dir",
        required=False,
    ),
    FieldSpec(
        "slurm_partition",
        "Slurm partition 名称",
        "Partition to request; required when cluster=slurm unless sbatch_template is set.",
        "text",
        required=False,
    ),
    FieldSpec(
        "sbatch_template",
        "sbatch 模板脚本绝对路径",
        "Existing absolute path to an sbatch template; required when cluster=slurm unless slurm_partition is set.",
        "existing_file",
        required=False,
    ),
    FieldSpec(
        "contract_output_dir",
        "harness 交接文件输出目录",
        "Absolute directory for INPUT_CONTRACT.json and generated prompts. Defaults to <project_root>/handoff/twinflow_<run_name>.",
        "output_dir",
        required=False,
    ),
    FieldSpec(
        "notes",
        "补充说明",
        "Optional extra constraints, known risks, or project-specific notes.",
        "text",
        required=False,
    ),
]


SPEC_BY_NAME = {spec.name: spec for spec in FIELD_SPECS}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "required":
        print_required()
        return 0
    if args.command == "validate":
        return validate_command(args)
    if args.command == "init":
        return init_command(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twinflow_porting_harness",
        description="Collect and validate the required input contract before a TwinFlow porting run starts.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("required", help="Print the required input contract fields.")

    validate = sub.add_parser("validate", help="Validate an existing INPUT_CONTRACT.json.")
    validate.add_argument("contract", type=Path)

    init = sub.add_parser("init", help="Collect inputs, validate them, and write handoff artifacts.")
    init.add_argument("--config", type=Path, help="JSON file with input contract values.")
    init.add_argument("--ask", action="store_true", help="Prompt repeatedly for missing/invalid fields.")
    init.add_argument("--create-output-root", action="store_true", help="Create output_root/slurm_log_dir/contract_output_dir when missing.")
    init.add_argument("--dry-run", action="store_true", help="Validate and print where files would be written without writing.")
    init.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Set an arbitrary contract value.")
    for spec in FIELD_SPECS:
        init.add_argument(f"--{spec.name.replace('_', '-')}", dest=spec.name, help=spec.help)
    return parser


def print_required() -> None:
    print("# Required TwinFlow porting input contract")
    for spec in FIELD_SPECS:
        marker = "required" if spec.required else "conditional/optional"
        choices = f" choices={CHOICES[spec.name]}" if spec.name in CHOICES else ""
        print(f"- {spec.name} ({marker}, {spec.kind}{choices}): {spec.prompt}")
    print()
    print("Slurm rule: when cluster=slurm, slurm_log_dir is required and one of slurm_partition or sbatch_template is required.")


def validate_command(args: argparse.Namespace) -> int:
    values = load_json(args.contract)
    errors = validate_values(values, create_dirs=False)
    if errors:
        print_missing_or_invalid(errors, values)
        return 2
    print(f"OK: {args.contract} is a complete TwinFlow porting input contract.")
    return 0


def init_command(args: argparse.Namespace) -> int:
    values = collect_values(args)
    ask = bool(args.ask)
    create_dirs = bool(args.create_output_root and not args.dry_run)

    while True:
        errors = validate_values(values, create_dirs=create_dirs)
        if not errors:
            break
        if not ask:
            print_missing_or_invalid(errors, values)
            return 2
        prompt_for_errors(values, errors)

    output_dir = contract_output_dir(values)
    if args.dry_run:
        print(f"DRY RUN OK: would write handoff artifacts to {output_dir}")
        return 0
    write_artifacts(values, output_dir)
    print(f"Wrote TwinFlow input contract and handoff artifacts to: {output_dir}")
    print(f"Next: give START_PROMPT.md to the coding agent and do not start porting if validate fails.")
    return 0


def collect_values(args: argparse.Namespace) -> dict[str, str]:
    values: dict[str, str] = {}
    if args.config:
        values.update(load_json(args.config))
    for spec in FIELD_SPECS:
        value = getattr(args, spec.name, None)
        if value not in (None, ""):
            values[spec.name] = value
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    return normalize_values(values)


def load_json(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Contract JSON must be an object: {path}")
    return normalize_values({str(k): str(v) for k, v in data.items() if v is not None})


def normalize_values(values: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in values.items():
        if isinstance(value, str):
            value = value.strip()
        out[key] = value
    return out


def required_names(values: dict[str, str]) -> set[str]:
    names = {spec.name for spec in FIELD_SPECS if spec.required}
    if values.get("cluster") == "slurm":
        names.add("slurm_log_dir")
    return names


def validate_values(values: dict[str, str], create_dirs: bool) -> list[str]:
    errors: list[str] = []
    required = required_names(values)
    for name in sorted(required):
        if not values.get(name):
            errors.append(f"missing:{name}")

    if values.get("cluster") == "slurm" and not (
        values.get("slurm_partition") or has_valid_sbatch_template(values, create_dirs=create_dirs)
    ):
        errors.append("missing:slurm_partition_or_sbatch_template")

    for name, raw_value in values.items():
        spec = SPEC_BY_NAME.get(name)
        if spec is None or raw_value == "":
            continue
        errors.extend(validate_field(spec, raw_value, create_dirs))

    return errors


def validate_field(spec: FieldSpec, value: str, create_dirs: bool) -> list[str]:
    errors: list[str] = []
    try:
        if spec.kind == "choice":
            choices = CHOICES[spec.name]
            if value not in choices:
                errors.append(f"invalid:{spec.name}: expected one of {', '.join(choices)}")
        elif spec.kind == "positive_int":
            try:
                parsed = int(value)
            except ValueError:
                errors.append(f"invalid:{spec.name}: expected positive integer")
                return errors
            if parsed <= 0:
                errors.append(f"invalid:{spec.name}: expected positive integer")
        elif spec.kind == "slug":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", value):
                errors.append(f"invalid:{spec.name}: use letters/numbers/._- and start with a letter or number")
        elif spec.kind == "existing_dir":
            path = checked_abs_path(value, spec.name)
            if not path.is_dir():
                errors.append(f"invalid:{spec.name}: directory does not exist: {path}")
        elif spec.kind == "existing_file":
            path = checked_abs_path(value, spec.name)
            if not path.is_file():
                errors.append(f"invalid:{spec.name}: file does not exist: {path}")
        elif spec.kind == "existing_path":
            path = checked_abs_path(value, spec.name)
            if not path.exists():
                errors.append(f"invalid:{spec.name}: path does not exist: {path}")
        elif spec.kind == "output_dir":
            path = checked_abs_path(value, spec.name)
            if path.exists() and not path.is_dir():
                errors.append(f"invalid:{spec.name}: exists but is not a directory: {path}")
            elif not path.exists():
                if create_dirs:
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    errors.append(f"invalid:{spec.name}: directory does not exist: {path}")
        elif spec.kind == "url_or_path":
            ref_error = validate_twinflow_ref(value, spec.name)
            if ref_error is None:
                return errors
            if ref_error:
                errors.append(ref_error)
                return errors
            path = checked_abs_path(value, spec.name)
            if not path.exists():
                errors.append(f"invalid:{spec.name}: local TwinFlow path does not exist: {path}")
            elif not path.is_dir():
                errors.append(f"invalid:{spec.name}: local TwinFlow path must be a directory: {path}")
            elif not (path / ".git").exists():
                errors.append(f"invalid:{spec.name}: local TwinFlow path must be a git clone/worktree: {path}")
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def has_valid_sbatch_template(values: dict[str, str], create_dirs: bool) -> bool:
    template = values.get("sbatch_template")
    if not template:
        return False
    spec = SPEC_BY_NAME["sbatch_template"]
    return not validate_field(spec, template, create_dirs=create_dirs)


def checked_abs_path(value: str, field_name: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"invalid:{field_name}: path must be absolute: {value}")
    return path


def validate_twinflow_ref(value: str, field_name: str) -> str | None:
    """Return None for a valid remote ref, "" for local-path validation, or an error."""
    git_match = re.fullmatch(r"git@([^:\s]+):(.+)", value)
    if git_match:
        return None if git_match.group(2).strip("/") else f"invalid:{field_name}: expected git@host:owner/repo"
    if value.startswith("git@"):
        return f"invalid:{field_name}: expected git@host:owner/repo"

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "ssh"}:
        if parsed.netloc and parsed.path.strip("/"):
            return None
        return f"invalid:{field_name}: expected full URL with scheme, host, and repo path"
    if parsed.scheme:
        return f"invalid:{field_name}: expected http(s), ssh, git@ URL, or absolute local path"
    return ""


def print_missing_or_invalid(errors: Iterable[str], values: dict[str, str]) -> None:
    missing = []
    invalid = []
    for err in errors:
        if err.startswith("missing:"):
            missing.append(err.split(":", 1)[1])
        else:
            invalid.append(err)
    print("ERROR: TwinFlow porting input contract is incomplete.")
    print("Do not start code changes, training, or evaluation until these are resolved.")
    if missing:
        print()
        print("Ask the user for these required values:")
        for name in missing:
            if name == "slurm_partition_or_sbatch_template":
                print("- slurm_partition or sbatch_template: Slurm queue information or an sbatch template path")
            else:
                spec = SPEC_BY_NAME.get(name)
                prompt = spec.prompt if spec else name
                print(f"- {name}: {prompt}")
    if invalid:
        print()
        print("Invalid values:")
        for err in invalid:
            print(f"- {err}")
    if values:
        print()
        print("Values already supplied:")
        for key in sorted(values):
            if values[key]:
                print(f"- {key}={values[key]}")


def prompt_for_errors(values: dict[str, str], errors: list[str]) -> None:
    names: list[str] = []
    for err in errors:
        if err.startswith("missing:"):
            name = err.split(":", 1)[1]
            names.append(name)
        elif err.startswith("invalid:"):
            parts = err.split(":", 2)
            if len(parts) >= 2:
                names.append(parts[1])
    if not names:
        print_missing_or_invalid(errors, values)
        raise SystemExit(2)
    for name in dict.fromkeys(names):
        if name == "slurm_partition_or_sbatch_template":
            if any(err.startswith("invalid:sbatch_template:") for err in errors):
                values.pop("sbatch_template", None)
            prompt_for_slurm_launch(values, errors)
        elif name == "sbatch_template" and values.get("slurm_partition"):
            values.pop("sbatch_template", None)
            continue
        else:
            if any(err.startswith(f"invalid:{name}:") for err in errors):
                values.pop(name, None)
            prompt_for_field(values, name, errors)


def prompt_for_field(values: dict[str, str], name: str, errors: list[str]) -> None:
    spec = SPEC_BY_NAME.get(name)
    if spec is None:
        return
    while not values.get(name):
        try:
            answer = input(f"{spec.prompt} ({name}): ").strip()
        except EOFError:
            print_missing_or_invalid(errors, values)
            raise SystemExit(2)
        if answer:
            values[name] = answer
            break


def prompt_for_slurm_launch(values: dict[str, str], errors: list[str]) -> None:
    while not (values.get("slurm_partition") or values.get("sbatch_template")):
        try:
            partition = input("Slurm partition 名称，或留空改填 sbatch_template (slurm_partition): ").strip()
            if partition:
                values["slurm_partition"] = partition
                return
            template = input("sbatch 模板脚本绝对路径 (sbatch_template): ").strip()
        except EOFError:
            print_missing_or_invalid(errors, values)
            raise SystemExit(2)
        if template:
            values["sbatch_template"] = template
            return


def contract_output_dir(values: dict[str, str]) -> Path:
    if values.get("contract_output_dir"):
        return checked_abs_path(values["contract_output_dir"], "contract_output_dir")
    project = checked_abs_path(values["project_root"], "project_root")
    return project / "handoff" / f"twinflow_{values['run_name']}"


def write_artifacts(values: dict[str, str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    values = normalize_values(values)
    values["contract_output_dir"] = str(output_dir)
    values["generated_at"] = datetime.now().isoformat(timespec="seconds")
    values["harness_version"] = "0.2.0"

    write_text(output_dir / "INPUT_CONTRACT.json", json.dumps(values, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_text(output_dir / "START_PROMPT.md", render_start_prompt(values))
    write_text(output_dir / "ACCEPTANCE_CHECKLIST.md", render_acceptance_checklist(values))
    write_text(output_dir / "MANIFEST.md", render_manifest(values))


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def render_start_prompt(values: dict[str, str]) -> str:
    rows = "\n".join(f"- `{key}`: `{values[key]}`" for key in sorted(values) if values[key])
    slurm_block = ""
    if values.get("cluster") == "slurm":
        slurm_block = """
## Slurm Rules

- Use `sbatch` for formal runs; local/login runs are only for lightweight smoke or inference when GPU availability is explicitly checked.
- Check `squeue` before and during formal runs.
- Save stdout/stderr under `slurm_log_dir`.
"""
    return f"""# TwinFlow Porting Start Prompt

Use this prompt to start the coding agent. Do not begin code changes until `INPUT_CONTRACT.json` validates with:

```bash
python -m twinflow_porting_harness validate INPUT_CONTRACT.json
```

## Input Contract

{rows}

## Required Workflow

1. Work only under `project_root`, except read-only access to explicitly supplied checkpoints, data, and TwinFlow references.
2. Read the original project call chain bottom-up before editing: model forward, sampler, trainer, dataset/collate, condition encoder, checkpoint/resume, eval/decode/render/export.
3. Compare against `official_twinflow_ref`; document any semantic deviation before implementation.
4. Add `t`/`tt` support, sampler semantics, TwinFlow trainer, checkpoint/resume, and eval glue in the smallest project-native patch.
5. Preserve original pretrained behavior; old checkpoints may only miss the new `tt_embedder.*`, which must initialize from `t_embedder.*`.
6. Run the smoke gates in `ACCEPTANCE_CHECKLIST.md` before any formal training.
7. Distilled inference/eval must use `cfg=0` unless a written review proves otherwise.
8. Every eval contact sheet must include `cond`, denormalized `GT`, old/pretrained baseline, and distilled few/any columns under the same render path.
9. Stop and ask the user if a required path, checkpoint, data split, or output location is missing or inconsistent.
{slurm_block}
## Deliverables

- Updated code and config diffs.
- Handoff with absolute paths.
- Smoke logs and failure fixes.
- Eval artifacts: NPZ, PLY, GLB, front-view PNG, contact sheet, and manifest.
- Clean-context code review before formal run and before claiming results.
"""


def render_acceptance_checklist(values: dict[str, str]) -> str:
    return f"""# TwinFlow Porting Acceptance Checklist

Run: `{values.get("run_name", "")}`

## Contract Gate

- [ ] `INPUT_CONTRACT.json` validates.
- [ ] All data/checkpoint/env paths are absolute.
- [ ] `output_root` and Slurm log paths are explicit.
- [ ] Agent has not started implementation before resolving missing inputs.

## Call-Chain Review Gate

- [ ] Original model forward and timestep embedding reviewed.
- [ ] Official TwinFlow `t/tt` semantics reviewed.
- [ ] Sampler `few/any/mul` semantics mapped to the project.
- [ ] Dataset/collate sorting and branch assignment risk reviewed.
- [ ] Dense vs sparse latent operations reviewed.
- [ ] Condition/uncondition construction matches pretrained behavior.
- [ ] Checkpoint, EMA, resume, and logging paths reviewed.
- [ ] Eval/decode/render/export chain reviewed to the lowest practical layer.

## Implementation Gate

- [ ] Model forward supports both `t` and `tt`.
- [ ] `tt_embedder` warm-starts from `t_embedder`.
- [ ] Old checkpoint loading allows only audited new keys.
- [ ] Branch masks are built on the full local batch before microbatching.
- [ ] Dist-match/e2e losses only affect intended branches.
- [ ] Student, optimizer, EMA/teacher, and TwinFlow state save/resume correctly.
- [ ] Distilled eval rejects nonzero CFG by default.

## Smoke Gate

- [ ] Config parse.
- [ ] Shell `bash -n`.
- [ ] Python `py_compile`.
- [ ] Synthetic 1-step.
- [ ] Real-data fetch preflight on `dataset_root`.
- [ ] Single-GPU real-data 1-step.
- [ ] Multi-GPU/DDP/FSDP 1-step when applicable.
- [ ] Save/resume 1-step.
- [ ] Eval/decode smoke.
- [ ] NPZ/PLY/GLB/front-view PNG/contact/manifest generated.

## Formal Run Gate

- [ ] Slurm queue and GPU availability checked when applicable.
- [ ] GPU utilization target is documented.
- [ ] Checkpoint retention is bounded.
- [ ] Fatal patterns monitored: Traceback, RuntimeError, CUDA OOM, NCCL, NaN/Inf, retry exhaustion.
- [ ] Branch counts match configured probabilities.
- [ ] Loss/mse/grad_norm are finite.

## Protocol Eval Gate

- [ ] Fixed `eval_sample_spec`, seed, and noise.
- [ ] Contact columns include `cond`, denormalized `GT`, old/pretrained baseline, and distilled few/any/mul variants.
- [ ] GT uses the same denormalization and render path as model outputs.
- [ ] Silent data substitution is disabled or sample hashes are recorded.
- [ ] Every reported issue points to a concrete image/crop.
- [ ] Result claims are reviewed by a clean-context reviewer.
"""


def render_manifest(values: dict[str, str]) -> str:
    return f"""# TwinFlow Harness Manifest

- `INPUT_CONTRACT.json`: validated input contract.
- `START_PROMPT.md`: prompt to give the coding agent.
- `ACCEPTANCE_CHECKLIST.md`: gates that must pass before formal training and result claims.

Generated at: `{values.get("generated_at", "")}`
Harness version: `{values.get("harness_version", "")}`
"""


if __name__ == "__main__":
    raise SystemExit(main())
