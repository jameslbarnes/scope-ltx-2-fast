"""scope-ltx-2-fast: LTX 2.3 with all models GPU-resident (96GB+)."""

import scope.core

from .pipeline import LTX2FastPipeline


@scope.core.hookimpl
def register_pipelines(register):
    register(LTX2FastPipeline)
