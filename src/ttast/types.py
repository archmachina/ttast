
import yaml
import sys
import re
import glob
import os
import inspect
import copy
import logging

from .util import *

logger = logging.getLogger(__name__)

class TextBlock:
    def __init__(self, text, *, tags=None):
        validate(isinstance(tags, (list, set)) or tags is None, "Tags supplied to TextBlock must be a set, list or absent")

        self.text = text

        self.tags = set()
        if tags is not None:
            for tag in tags:
                self.tags.add(tag)

        self.meta = {}

class Pipeline:
    def __init__(self):
        self._steps = []
        self._handlers = {}
        self._support_handlers = []
        self._vars = {
            "env": os.environ.copy()
        }
        self._blocks = []

        self.step_limit = 100

    def add_block(self, block):
        validate(isinstance(block, TextBlock), "Invalid block passed to pipeline add_block")

        self._blocks.append(block)

    def remove_block(self, block):
        validate(isinstance(block, TextBlock), "Invalid block passed to pipeline remove_block")

        self._blocks.remove(block)

    def copy_vars(self):
        return copy.deepcopy(self._vars)

    def set_var(self, key, value):
        if key in ["env"]:
            raise PipelineRunException(f"Disallowed key in pipeline set_var: {key}")

        self._vars[key] = value

    def add_step(self, step_def):
        validate(isinstance(step_def, dict), "Invalid step definition passed to add_step")

        if self.step_limit > 0 and len(self._steps) > self.step_limit:
            raise PipelineRunException(f"Reached limit of {self.step_limit} steps in pipeline. This is a safe guard to prevent infinite recursion")

        self._steps.append(step_def)

    def add_handlers(self, handlers):
        validate(isinstance(handlers, dict), "Invalid handlers passed to add_handlers")
        validate((all(x is None or (inspect.isclass(x) and issubclass(x, Handler))) for x in handlers.values()), "Invalid handlers passed to add_handlers")

        for key in handlers:
            self._handlers[key] = handlers[key]

    def add_support_handlers(self, handlers):
        validate(isinstance(handlers, list), "Invalid handlers passed to add_support_handlers")
        validate((all(inspect.isclass(x) and issubclass(x, SupportHandler)) for x in handlers), "Invalid handlers passed to add_support_handlers")

        for handler in handlers:
            if handler not in self._support_handlers:
                self._support_handlers.append(handler)

    def run(self):
        # This is a while loop with index to allow the pipeline to be appended to during processing
        index = 0
        while index < len(self._steps):

            # Clone current step definition
            step_def = self._steps[index].copy()
            index = index + 1

            # Extract type
            step_type = pop_property(step_def, "type", template_map=None)
            validate(isinstance(step_type, str) and step_type != "", "Step 'type' is required and must be a non empty string")

            # Retrieve the handler for the step type
            handler = self._handlers.get(step_type)
            if handler is None:
                raise PipelineRunException(f"Invalid step type in step {step_type}")

            # Create an instance per block for the step type, or a single instance for step types
            # that are not per block.
            if handler.is_per_block():
                logger.debug(f"Processing {step_type} - per_block")
                # Create a copy of blocks to allow steps to alter the block list while we iterate
                block_list_copy = self._blocks.copy()

                for block in block_list_copy:
                    self._process_step_instance(step_def, handler, block)
            else:
                logger.debug(f"Processing {step_type} - singular")
                self._process_step_instance(step_def, handler)

    def _process_step_instance(self, step_def, handler, block=None):
        validate(isinstance(step_def, dict), "Invalid step definition passed to _process_step_instance")
        validate(inspect.isclass(handler) and issubclass(handler, Handler), "Invalid handler passed to _process_step_instance")
        validate(block is None or isinstance(block, TextBlock), "Invalid text block passed to _process_step_instance")

        # Create new vars for the instance, based on the pipeline vars, plus including
        # any block vars, if present
        step_vars = self.copy_vars()
        if block is not None:
            step_vars["meta"] = copy.deepcopy(block.meta)
            step_vars["tags"] = copy.deepcopy(block.tags)

        state = PipelineStepState(step_def, self, step_vars)

        #
        # Parsing
        #

        # Initialise and parse support handlers
        support_handlers = [x() for x in self._support_handlers]
        for support in support_handlers:
            support.init(state)
            support.parse()

        # Initialise and parse the main handler
        instance = handler()
        instance.init(state)
        instance.parse()

        # At this point, there should be no properties left in the dictionary as all of the handlers should have
        # extracted their own properties.
        validate(len(state.step_def.keys()) == 0, f"Unknown properties on step definition: {list(state.step_def.keys())}")

        #
        # Execution
        #

        # Processing may start with a single block, which may become more, none or stay at one
        # block.
        # If a block is split, then two blocks replace it, while a block may disappear from the list
        # if the handler filters it out.
        # Any new blocks should also be processed by the following handlers, so the working block
        # list is dynamic.

        # block_list contains a list of the blocks to operate on for the current handler, while working_list
        # is the list being generated for the next handler to operate on.

        block_list = [block]
        working_list = []

        if block is not None:
            logger.debug(f"Operating on block: {hex(id(block))}")
            logger.debug(f" meta: {block.meta}")
            logger.debug(f" tags: {block.tags}")

        # Run any preprocessing handlers
        for support in support_handlers:
            for current_block in block_list:
                logger.debug(f"Calling pre support handler ({support}) for block {hex(id(current_block))}")
                result = support.pre(current_block)
                if result is None:
                    # If the handler didn't return anything, then just add the current block
                    # for the next round of handlers
                    working_list.append(current_block)
                else:
                    # The handler returned a replacement list of blocks, which should replace the current
                    # block in the block list for the next round of handlers.
                    # This list could also be empty, removing the block from further processing
                    for x in result:
                        working_list.append(x)

            # Update block_list with the new list of blocks for the next round of handlers and reset
            # working_list
            block_list = working_list
            working_list = []

        # Perform processing for the main handler
        for current_block in block_list:
            logger.debug(f"Calling handler ({instance}) for block {hex(id(current_block))}")
            result = instance.run(current_block)
            if result is None:
                working_list.append(current_block)
            else:
                for x in result:
                    working_list.append(x)

        block_list = working_list
        working_list = []

        # Run any post processing handlers
        for support in support_handlers:
            for current_block in block_list:
                logger.debug(f"Calling post support handler ({support}) for block {hex(id(current_block))}")
                result = support.post(current_block)
                if result is None:
                    working_list.append(current_block)
                else:
                    for x in result:
                        working_list.append(x)

            block_list = working_list
            working_list = []

class PipelineStepState:
    def __init__(self, step_def, pipeline, step_vars):
        validate(isinstance(step_def, dict), "Invalid step_def passed to PipelineStepState")
        validate(isinstance(pipeline, Pipeline) or pipeline is None, "Invalid pipeline passed to PipelineStepState")
        validate(isinstance(step_vars, dict), "Invalid step vars passed to PipelineStepState")

        self.step_def = step_def.copy()
        self.pipeline = pipeline
        self.vars = step_vars

class SupportHandler:
    def init(self, state):
        validate(isinstance(state, PipelineStepState), "Invalid step state passed to SupportHandler")

        self.state = state

    def parse(self):
        raise PipelineRunException("parse undefined in SupportHandler")

    def pre(self, block):
        raise PipelineRunException("pre undefined in SupportHandler")

    def post(self, block):
        raise PipelineRunException("post undefined in SupportHandler")

class Handler:
    def is_per_block(self):
        raise PipelineRunException("is_per_block undefined in Handler")

    def init(self, state):
        validate(isinstance(state, PipelineStepState), "Invalid step state passed to Handler")

        self.state = state

    def parse(self):
        raise PipelineRunException("parse undefined in Handler")

    def run(self, block):
        raise PipelineRunException("run undefined in Handler")
