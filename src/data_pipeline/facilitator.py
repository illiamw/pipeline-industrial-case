import logging
from collections.abc import Callable, Iterable
from logging import Logger
from typing import TypeVar

T = TypeVar("T")


def validate_required_columns(
	available_columns: Iterable[str], required_columns: Iterable[str]
) -> None:
	"""Raise a clear error when required columns are absent from a dataset."""
	available = set(available_columns)
	missing = list(dict.fromkeys(
		column for column in required_columns if column not in available
	))
	if missing:
		raise ValueError(f"Missing required columns: {', '.join(missing)}")


def run_pipeline(
	data: T,
	steps: Iterable[tuple[str, Callable[[T], T]]],
	logger: Logger | None = None,
) -> T:
	"""Run named transformations in order, logging and re-raising failures."""
	pipeline_logger = logger or logging.getLogger(__name__)

	for step_name, transform in steps:
		pipeline_logger.info("Starting pipeline step: %s", step_name)
		try:
			data = transform(data)
		except Exception:
			pipeline_logger.exception("Pipeline step failed: %s", step_name)
			raise
		pipeline_logger.info("Completed pipeline step: %s", step_name)

	return data
