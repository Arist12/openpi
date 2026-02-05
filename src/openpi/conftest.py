import os

import pytest

from openpi.shared import gpu_utils as _gpu


def set_jax_cpu_backend_if_no_gpu() -> None:
    if not _gpu.gpu_available_no_torch():
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()
