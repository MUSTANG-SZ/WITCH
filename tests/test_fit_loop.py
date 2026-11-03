"""Tests for the fit output bookkeeping in fitter.py: output dirs and archiving old runs."""

from pathlib import Path

import pytest

from witch import fitter


@pytest.fixture(autouse=True)
def _reset_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no outputs recorded as written by this run."""
    monkeypatch.setattr(fitter, "_written", set())


def _touch(outdir: Path, *rels: str) -> None:
    """Create empty files at the given paths relative to `outdir`."""
    for rel in rels:
        path = outdir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def test_ours_records_and_returns_path(tmp_path: Path) -> None:
    """`_ours` returns its argument unchanged and remembers it by absolute path."""
    path = str(tmp_path / "results_final_fit.dill")
    assert fitter._ours(path) == path
    assert str(tmp_path / "results_final_fit.dill") in fitter._written


def test_archive_moves_only_stale_fit_outputs(tmp_path: Path) -> None:
    """Old fit outputs go to `old_runs/<timestamp>`, this run's files and other outputs stay."""
    _touch(
        tmp_path,
        "results_fit0.dill",
        "par_results_fit0.dill",
        "aic.yaml",
        "checkpoints/round_0.pkl",
        "checkpoints/mcmc_step_100.pkl",
        "results_fit1.dill",  # Written by this run
        "residual.fits",  # Not a fit output
    )
    fitter._ours(str(tmp_path / "results_fit1.dill"))

    fitter._archive_old_outputs(str(tmp_path))

    (archive,) = (tmp_path / "old_runs").iterdir()
    moved = sorted(
        p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()
    )
    assert moved == [
        "aic.yaml",
        "checkpoints/mcmc_step_100.pkl",
        "checkpoints/round_0.pkl",
        "par_results_fit0.dill",
        "results_fit0.dill",
    ]
    assert (tmp_path / "results_fit1.dill").exists()
    assert (tmp_path / "residual.fits").exists()
    assert not (tmp_path / "results_fit0.dill").exists()


def test_archive_does_nothing_for_a_clean_dir(tmp_path: Path) -> None:
    """With no leftover outputs no `old_runs` directory is created."""
    _touch(tmp_path, "residual.fits")
    fitter._archive_old_outputs(str(tmp_path))
    assert not (tmp_path / "old_runs").exists()


def test_outdir_separates_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fits to different datasets get different output directories."""
    monkeypatch.setenv("WITCH_OUTROOT", str(tmp_path))
    dirs = []
    for dsets in (["mustang2"], ["mustang2", "act"]):
        cfg = {
            "name": "x",
            "paths": {"outroot": "", "subdir": "s"},
            "datasets": dict.fromkeys(dsets),
            "sub": True,
            "fit": True,
            "sim": False,
        }
        dirs.append(fitter.get_outdir(cfg, None))
    assert dirs[0] != dirs[1]
    assert "mustang2+act" in dirs[1]
