from pathlib import Path
import zipfile

import pytest

from experiments.em_validation.kaggle import (
    archive_results,
    publication_config_for_kaggle,
    write_session_metadata,
)


def test_publication_config_partitions_k0a_values(tmp_path):
    config, plan = publication_config_for_kaggle(
        output_root=tmp_path / "out",
        mie_k0a_values=(0.5, 1.0, 2.0, 4.0, 6.0),
        mie_batch_axis="k0a",
        mie_batch_index=1,
        mie_batch_count=2,
    )
    assert config.runtime.device == "cuda"
    assert config.mie.k0a_values == (1.0, 4.0)
    assert plan.selected_values == (1.0, 4.0)
    assert plan.label == "k0a-batch-02-of-02"


def test_publication_config_partitions_grid_and_field_reference(tmp_path):
    config, _ = publication_config_for_kaggle(
        output_root=tmp_path / "out",
        mie_grid_sizes=(24, 32, 48),
        mie_batch_axis="grid",
        mie_batch_index=0,
        mie_batch_count=2,
    )
    assert config.mie.grid_sizes == (24, 48)
    assert config.mie.field_reference_grids == (24, 48)


def test_publication_config_rejects_empty_batch(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        publication_config_for_kaggle(
            output_root=tmp_path / "out",
            mie_k0a_values=(0.5,),
            mie_batch_axis="k0a",
            mie_batch_index=1,
            mie_batch_count=2,
        )


def test_archive_results_and_session_metadata(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "result.txt").write_text("ok", encoding="utf-8")
    metadata = write_session_metadata(root, {"commit": "abc"})
    assert metadata.is_file()

    archive, digest = archive_results(root)
    assert archive.is_file()
    assert len(digest) == 64
    with zipfile.ZipFile(archive) as stream:
        names = set(stream.namelist())
    assert "run/result.txt" in names
    assert "run/kaggle_session.json" in names
