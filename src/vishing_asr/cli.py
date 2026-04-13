"""CLI для первого этапа `audio -> text`."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from vishing_asr.io_utils import resolve_input_dir, resolve_output_dir
from vishing_asr.transcribe import (
    TranscribeConfig,
    normalize_language,
    resolve_compute_type,
    resolve_device,
    transcribe_dataset,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Точка входа CLI для ASR-этапа."""


@main.command()
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=Path("./vishing"),
    show_default=True,
    help="Dataset root directory containing samples/ and test/.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False, dir_okay=True),
    default=None,
    help="Directory for generated artifacts. Must stay inside <input-dir>/artifacts.",
)
@click.option(
    "--backend",
    type=click.Choice(["faster-whisper"], case_sensitive=False),
    default="faster-whisper",
    show_default=True,
    help="ASR backend.",
)
@click.option(
    "--model-size",
    type=click.Choice(["tiny", "base", "small", "medium"], case_sensitive=False),
    default="base",
    show_default=True,
    help="Whisper model size.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Inference device.",
)
@click.option(
    "--compute-type",
    type=click.Choice(["int8", "float16", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="CTranslate2 compute type.",
)
@click.option(
    "--language",
    default="ru",
    show_default=True,
    help="Language code, for example 'ru'. Use 'auto' for auto-detection.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Recompute transcripts even if .txt/.json artifacts already exist.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Process only the first N discovered files.",
)
def transcribe(
    input_dir: Path,
    output_dir: Path | None,
    backend: str,
    model_size: str,
    device: str,
    compute_type: str,
    language: str,
    overwrite: bool,
    limit: int | None,
) -> None:
    """Транскрибировать датасет vishing в CSV, TXT и JSON-артефакты."""

    console = Console()

    try:
        resolved_input_dir = resolve_input_dir(input_dir)
        resolved_output_dir = resolve_output_dir(resolved_input_dir, output_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    normalized_language = normalize_language(language)
    resolved_device = resolve_device(device.lower())
    resolved_compute_type = resolve_compute_type(compute_type.lower(), resolved_device)

    if resolved_device == "cpu" and resolved_compute_type == "float16":
        raise click.ClickException("compute_type=float16 is not supported for cpu; use int8 or auto")

    config = TranscribeConfig(
        input_dir=resolved_input_dir,
        output_dir=resolved_output_dir,
        backend=backend.lower(),
        model_name=model_size.lower(),
        device=resolved_device,
        compute_type=resolved_compute_type,
        language=normalized_language,
        overwrite=overwrite,
        limit=limit,
    )

    console.print(
        "[bold]Transcription run[/bold]\n"
        f"input_dir={config.input_dir}\n"
        f"output_dir={config.output_dir}\n"
        f"backend={config.backend} model={config.model_name} device={config.device} compute={config.compute_type}"
    )

    try:
        records = transcribe_dataset(config, console=console)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    error_count = sum(1 for record in records if record["error"])
    console.print(
        f"[green]Completed[/green]: {len(records)} files processed, "
        f"{error_count} errors, CSV saved to {config.output_dir / 'transcripts.csv'}"
    )


if __name__ == "__main__":
    main()
