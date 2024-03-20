
import os
import copy
import inspect

from .util import *
from .exception import *
from . import types

logger = logging.getLogger(__name__)

def get_builtin_handlers():

    return {
        "config": types.HandlerConfig,
        "import": types.HandlerImport,
        "meta": types.HandlerMeta,
        "replace": types.HandlerReplace,
        "split_yaml": types.HandlerSplitYaml,
        "stdin": types.HandlerStdin,
        "stdout": types.HandlerStdout,
        "template": types.HandlerTemplate
    }

def process_pipeline(pipeline_steps, builtin_handlers=True, custom_handlers=None):
    validate(isinstance(pipeline_steps, list) and all(isinstance(x, dict) for x in pipeline_steps),
        "Pipeline steps passed to process_pipeline must be a list of dictionaries")
    validate(isinstance(builtin_handlers, bool), "Invalid builtin_handlers passed to process_pipeline. Must be bool")
    validate(isinstance(custom_handlers, dict) or custom_handlers is None, "Invalid custom_handlers passed to process_pipeline. Must be a dict of Handlers")

    # Define the pipeline and add all pipeline steps
    pipeline = types.Pipeline()
    for step in pipeline_steps:
        pipeline.add_step(step)

    # Add built in handlers, if required
    if builtin_handlers:
        pipeline.add_handlers(get_builtin_handlers())

    # Add any defined custom handlers
    if custom_handlers is not None:
        pipeline.add_handlers(custom_handlers)

    # Run the pipeline to completion
    pipeline.run()
