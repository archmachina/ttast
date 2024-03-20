
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
        "template": builtin.HandlerTemplate
    }

def builtin_support_handlers():

    return [
        builtin.SupportHandlerMatchTags,
        builtin.SupportHandlerWhen,
        builtin.SupportHandlerTags
    ]

def build_default_pipeline():
    pipeline = types.Pipeline()

    pipeline.add_handlers(builtin_handlers())
    pipeline.add_support_handlers(builtin_support_handlers())

    return pipeline

def process_pipeline(pipeline_steps):
    validate(isinstance(pipeline_steps, list) and all(isinstance(x, dict) for x in pipeline_steps),
        "Pipeline steps passed to process_pipeline must be a list of dictionaries")

    # Define the pipeline and add all pipeline steps
    pipeline = build_default_pipeline()
    for step in pipeline_steps:
        pipeline.add_step(step)

    # Run the pipeline to completion
    pipeline.run()
