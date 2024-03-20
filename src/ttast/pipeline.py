
import os
import copy
import inspect

from .util import *
from .exception import *
from . import types

logger = logging.getLogger(__name__)

def process_pipeline(pipeline_steps, builtin_handlers=True, custom_handlers=None):
    validate(isinstance(pipeline_steps, list) and all(isinstance(x, dict) for x in pipeline_steps),
        "Pipeline steps passed to process_pipeline must be a list of dictionaries")
    validate(isinstance(builtin_handlers, bool), "Invalid builtin_handlers passed to process_pipeline. Must be bool")
    validate(isinstance(custom_handlers, dict) or custom_handlers is None, "Invalid custom_handlers passed to process_pipeline. Must be a dict of Handlers")
    if custom_handlers is not None:
        # Allow None entry for a handler to effectively disable a specific builtin handler
        validate((all(x is None or (inspect.isclass(x) and issubclass(x, Handler))) for x in custom_handlers.values()), "Invalid custom_handlers passed to process_pipeline. Must be a dict of Handlers")

    pipeline = types.Pipeline()
    pipeline.steps = pipeline_steps

    handler_map = {}

    # Add builtin handlers, if required
    if builtin_handlers:
        handler_map["config"] = types.HandlerConfig
        handler_map["import"] = types.HandlerImport
        handler_map["meta"] = types.HandlerMeta
        handler_map["replace"] = types.HandlerReplace
        handler_map["split_yaml"] = types.HandlerSplitYaml
        handler_map["stdin"] = types.HandlerStdin
        handler_map["stdout"] = types.HandlerStdout
        handler_map["template"] = types.HandlerTemplate

    # Copy custom handlers in to handler list
    if custom_handlers is not None:
        for key in custom_handlers:
            handler_map[key] = custom_handlers[key]

    # This is a while loop with index to allow the pipeline to be appended to during processing
    index = 0
    while index < len(pipeline.steps):

        # Clone current step definition
        step_def = pipeline.steps[index].copy()
        index = index + 1

        # Extract type
        step_type = pop_property(step_def, "type", template_map=None)
        validate(isinstance(step_type, str) and step_type != "", "Step 'type' is required and must be a non empty string")

        # Retrieve the handler for the step type
        handler = handler_map.get(step_type)
        if handler is None:
            raise PipelineRunException(f"Invalid step type in step {step_type}")

        # Create an instance per block for the step type, or a single instance for step types
        # that are not per block.
        if handler.is_per_block():
            # Create a copy of blocks to allow steps to alter the block list while we iterate
            block_list_copy = pipeline.blocks.copy()

            for block in block_list_copy:
                instance = types.PipelineStepInstance(step_def, pipeline=pipeline, handler=handler, block=block)
                instance.process()
        else:
            instance = types.PipelineStepInstance(step_def, pipeline=pipeline, handler=handler)
            instance.process()

