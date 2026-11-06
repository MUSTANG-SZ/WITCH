"""Tests for the fit output bookkeeping in fitter.py: output dirs and archiving old runs."""

from pathlib import Path

import numpy as np
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


def _masks(*blocks: tuple[tuple[bool, ...], int]) -> list[np.ndarray]:
    """Build per-round masks from `(mask, number of rounds)` blocks."""
    return [np.array(mask) for mask, n in blocks for _ in range(n)]


def test_round_masks_restores_the_current_round() -> None:
    """`_round_masks` reads the free parameters of every round and goes back to the round it was on."""
    per_round = [
        np.array([True, False]),
        np.array([True, False]),
        np.array([False, True]),
    ]

    class Stub:
        n_rounds = 3
        cur_round = 1
        to_fit = per_round[1]

        def set_round(self, r: int) -> "Stub":
            self.cur_round = r
            self.to_fit = per_round[r]
            return self

    stub = Stub()
    masks = fitter._round_masks(stub)
    assert all(np.array_equal(a, b) for a, b in zip(masks, per_round, strict=True))
    assert stub.cur_round == 1


def test_next_round_runs_every_round_without_convergence() -> None:
    """Unconverged rounds are run one after the other until the rounds run out."""
    masks = _masks(((True, False), 4))
    assert [fitter._next_round(masks, r, 0, 2) for r in range(4)] == [1, 2, 3, None]


def test_next_round_stops_when_the_only_block_converges() -> None:
    """With one block, converging ends the fit and skips the remaining rounds."""
    masks = _masks(((True, False), 5))
    assert fitter._next_round(masks, 2, 2, 2) is None
    assert fitter._next_round(masks, 2, 1, 2) == 3


def test_next_round_moves_to_the_next_block_on_convergence() -> None:
    """A converged block skips its remaining rounds and goes to the first round of the next block."""
    masks = _masks(((True, False), 3), ((False, True), 2), ((True, True), 2))
    assert fitter._next_round(masks, 1, 2, 2) == 3  # block 1 converged early
    assert fitter._next_round(masks, 3, 2, 2) == 5  # block 2 converged early
    assert fitter._next_round(masks, 5, 2, 2) is None  # last block converged


def test_next_round_goes_on_when_a_block_runs_out_of_rounds() -> None:
    """A block that never converges simply hands over to the next block when its rounds end."""
    masks = _masks(((True, False), 2), ((False, True), 2))
    assert fitter._next_round(masks, 1, 0, 2) == 2
    assert fitter._next_round(masks, 3, 0, 2) is None


def test_next_round_does_not_merge_a_block_that_comes_back() -> None:
    """Only consecutive rounds with the same free parameters form a block."""
    masks = _masks(((True, False), 2), ((False, True), 1), ((True, False), 2))
    assert fitter._next_round(masks, 0, 2, 2) == 2
    assert fitter._next_round(masks, 2, 2, 2) == 3
