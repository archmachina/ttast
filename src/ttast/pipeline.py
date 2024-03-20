
import os
import copy
import inspect

from .util import *
from .exception import *
from . import types
from . import builtin

logger = logging.getLogger(__name__)

def builtin_handlers():

    return {
        "config": builtin.HandlerConfig,
        "import": builtin.HandlerImport,
        "meta": builtin.HandlerMeta,
        "replace": builtin.HandlerReplace,
        "split_yaml": builtin.HandlerSplitYaml,
        "stdin": builtin.HandlerStdin,
        "stdout": builtin.HandlerStdout,
        "template": builtin.HandlerTemplate,
        "sum": builtin.HandlerSum
    }

def builtin_support_handlers():

    return [
        builtin.SupportHandlerMatchTags,
        builtin.SupportHandlerWhen,
        builtin.SupportHandlerTags
    ]

def run_pipeline(pipeline_steps, include_builtin=True, custom_handlers=None, custom_support_handlers=None):
    validate(isinstance(pipeline_steps, list) and all(isinstance(x, dict) for x in pipeline_steps),
        "Pipeline steps passed to run_pipeline must be a list of dictionaries")
    validate(isinstance(include_builtin, bool), "Invalid include_builtin parameter passed to run_pipeline")
    validate(isinstance(custom_handlers, dict) or custom_handlers is None, "Invalid custom_handlers passed to run_pipeline")
    validate(isinstance(custom_support_handlers, list) or custom_support_handlers is None, "Invalid custom_support_handlers passed to run_pipeline")

    pipeline = types.Pipeline()

    if include_builtin:
        pipeline.add_handlers(builtin_handlers())
        pipeline.add_support_handlers(builtin_support_handlers())

    if custom_handlers is not None:
        pipeline.add_handlers(custom_handlers)

    if custom_support_handlers is not None:
        pipeline.add_support_handlers(custom_support_handlers)

    for step in pipeline_steps:
        pipeline.add_step(step)

    # Run the pipeline to completion
    pipeline.run()
