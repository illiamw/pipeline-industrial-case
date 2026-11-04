"""Tests for shared data pipeline helpers."""

import pytest

from src.data_pipeline.facilitator import run_pipeline, validate_required_columns


def test_validate_required_columns_accepts_available_columns() -> None:
	validate_required_columns(["id", "created_at"], ["id"])


def test_validate_required_columns_reports_missing_columns() -> None:
	with pytest.raises(ValueError, match="Missing required columns: id, created_at"):
		validate_required_columns(["name"], ["id", "created_at"])


def test_run_pipeline_applies_steps_in_order() -> None:
	steps = [
		("double", lambda value: value * 2),
		("increment", lambda value: value + 1),
	]

	assert run_pipeline(3, steps) == 7


def test_run_pipeline_reraises_step_errors() -> None:
	def fail(value: int) -> int:
		raise RuntimeError(f"transform failed for {value}")

	with pytest.raises(RuntimeError, match="transform failed for 3"):
		run_pipeline(3, [("failing_transform", fail)])
