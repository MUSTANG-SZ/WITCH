"""
Unit tests for checkpoint functionality in fitter.py
"""

import os

# We need to mock MPI before importing witch
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import dill as pk
import numpy as np
import pytest

sys.modules["mpi4py"] = MagicMock()
sys.modules["mpi4py.MPI"] = MagicMock()

from witch.fitter import _read_checkpoint, _read_model, _save_checkpoint


class TestCheckpointing:
    """Test checkpoint save/load functionality"""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary directory for checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_model(self):
        """Create a mock model object"""
        model = Mock()
        model.pars = np.array([1.0, 2.0, 3.0])
        model.errs = np.array([0.1, 0.2, 0.3])
        model.chisq = 100.0
        return model

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        return {
            "name": "test_cluster",
            "model": {"structures": {}},
            "constants": {"z": 0.5},
        }

    def test_save_checkpoint_creates_file(
        self, temp_checkpoint_dir, mock_model, mock_config
    ):
        """Test that _save_checkpoint creates a file"""
        ckpt_path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pkl")

        with patch("witch.fitter.comm") as mock_comm:
            mock_comm.Get_rank.return_value = 0  # Rank 0 saves

            _save_checkpoint(
                ckpt_path,
                [mock_model],
                None,  # datasets not saved
                mock_config,
                stage="test",
                round_num=0,
            )

        assert os.path.exists(ckpt_path), "Checkpoint file should be created"

    def test_save_checkpoint_non_rank_zero(
        self, temp_checkpoint_dir, mock_model, mock_config
    ):
        """Test that non-rank-0 processes don't save"""
        ckpt_path = os.path.join(temp_checkpoint_dir, "test_checkpoint.pkl")

        with patch("witch.fitter.comm") as mock_comm:
            mock_comm.Get_rank.return_value = 1  # Not rank 0

            _save_checkpoint(
                ckpt_path, [mock_model], None, mock_config, stage="test", round_num=0
            )

        assert not os.path.exists(ckpt_path), "Non-rank-0 should not create checkpoint"

    def test_checkpoint_save_load_roundtrip(
        self, temp_checkpoint_dir, mock_model, mock_config
    ):
        """Test that we can save and load a checkpoint"""
        ckpt_path = os.path.join(temp_checkpoint_dir, "roundtrip.pkl")

        # Save
        with patch("witch.fitter.comm") as mock_comm:
            mock_comm.Get_rank.return_value = 0

            _save_checkpoint(
                ckpt_path,
                [mock_model],
                None,
                mock_config,
                stage="fit_round",
                round_num=2,
            )

        # Load
        loaded_models, start_round, stage, cfg = _read_checkpoint(ckpt_path)

        assert len(loaded_models) == 1, "Should load one model"
        assert start_round == 3, "Should resume from next round (2+1)"
        assert stage == "fit_round", "Stage should match"
        assert cfg["name"] == "test_cluster", "Config should match"

    def test_checkpoint_contains_correct_data(
        self, temp_checkpoint_dir, mock_model, mock_config
    ):
        """Test that checkpoint contains all required data"""
        ckpt_path = os.path.join(temp_checkpoint_dir, "data_test.pkl")

        with patch("witch.fitter.comm") as mock_comm:
            mock_comm.Get_rank.return_value = 0

            _save_checkpoint(
                ckpt_path,
                [mock_model],
                None,
                mock_config,
                stage="mcmc",
                round_num=1,
                step_num=500,
            )

        # Manually load and inspect
        with open(ckpt_path, "rb") as f:
            state = pk.load(f)

        assert "models" in state, "Should contain models"
        assert "cfg" in state, "Should contain config"
        assert "stage" in state, "Should contain stage"
        assert "round" in state, "Should contain round number"
        assert "step" in state, "Should contain step number"
        assert state["stage"] == "mcmc"
        assert state["round"] == 1
        assert state["step"] == 500

    def test_read_checkpoint_missing_file(self):
        """Test that reading non-existent checkpoint raises error"""
        with pytest.raises(FileNotFoundError):
            _read_checkpoint("/nonexistent/path/checkpoint.pkl")

    def test_read_model_function(self, temp_checkpoint_dir, mock_model, mock_config):
        """Test the _read_model function"""
        ckpt_path = os.path.join(temp_checkpoint_dir, "model_test.pkl")

        # Save first
        with patch("witch.fitter.comm") as mock_comm:
            mock_comm.Get_rank.return_value = 0

            _save_checkpoint(
                ckpt_path,
                [mock_model],
                None,
                mock_config,
                stage="fit_round",
                round_num=3,
            )

        # Load with _read_model
        models, datasets, round_num, stage, cfg = _read_model(ckpt_path)

        assert len(models) == 1, "Should load one model"
        assert round_num == 3, "Should return round number (not incremented)"
        assert stage == "fit_round"
        assert datasets is None, "Datasets should be None (not loaded)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
