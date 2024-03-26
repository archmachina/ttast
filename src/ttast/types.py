
import yaml
import sys
import re
import glob
import os
import inspect
import copy
import logging

from .util import *
from .exception import *

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
        self._filters = {}

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

    def add_filters(self, filters):
        validate(isinstance(filters, dict), "Invalid filters passed to add_filters")
        validate(all((callable(x) or x is None) for x in filters.values()), "Invalid filters passed to add_filters")

        for key in filters:
            self._filters[key] = filters[key]

    def build_template_environment(self):
        environment = Jinja2.Environment()

        for key in self._filters:
            value = self._filters[key]

            if value is not None:
                environment.filters[key] = value

        return environment

    def run(self):
        # This is a while loop with index to allow the pipeline to be appended to during processing
        index = 0
        while index < len(self._steps):

            # Clone current step definition
            step_def = self._steps[index].copy()
            index = index + 1

            # Extract type
            if "type" not in step_def:
                raise PipelineRunException("Missing type property on pipeline step")

            step_type = step_def.pop("type")
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

        # Create a Templater
        templater = Templater(self._filters, step_vars)

        # Create the pipeline step state
        state = PipelineStepState(step_def, self, step_vars, templater)

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
    def __init__(self, step_def, pipeline, step_vars, templater):
        validate(isinstance(step_def, dict), "Invalid step_def passed to PipelineStepState")
        validate(isinstance(pipeline, Pipeline) or pipeline is None, "Invalid pipeline passed to PipelineStepState")
        validate(isinstance(step_vars, dict), "Invalid step vars passed to PipelineStepState")
        validate(isinstance(templater, Templater), "Invalid templater passed to PipelineStepState")

        self.step_def = step_def.copy()
        self.pipeline = pipeline
        self.vars = step_vars
        self.templater = templater

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

class Templater:
    def __init__(self, filters, template_vars=None):
        validate(isinstance(filters, dict), "Invalid filters passed to Templater ctor")
        validate(all((callable(x) or x is None) for x in filters.values()), "Invalid filters passed to Templater ctor")

        # Create a new Jinja2 environment
        self._environment = jinja2.Environment()

        # Update the Jinja2 environment with any custom filters
        for key in filters:
            value = filters[key]

            if value is not None:
                self._environment.filters[key] = value

        # Define the template vars
        if template_vars is None:
            self.vars = {}
        else:
            self.vars = copy.deepcopy(template_vars)

    def template_if_string(self, val, var_override=None):
        if var_override is not None and not isinstance(var_override, dict):
            raise PipelineRunException("Invalid var override passed to template_if_string")

        # Determine which vars will be used for templating
        template_vars = self.vars
        if var_override is not None:
            template_vars = var_override

        if isinstance(val, str):
            template = self._environment.from_string(val)
            return template.render(self.vars)

        return val

    def extract_property(self, spec, key, /, default=None, required=False, var_override=None, recursive_template=True):
        if not isinstance(spec, dict):
            raise PipelineRunException("Invalid spec passed to extract_property. Must be dict")

        if var_override is not None and not isinstance(var_override, dict):
            raise PipelineRunException("Invalid var_override passed to extract_property")

        if key not in spec:
            # Raise exception is the key isn't present, but required
            if required:
                raise KeyError(f'Missing key "{key}" in spec or value is null')

            # If the key is not present, return the default
            return default

        # Retrieve value
        val = spec.pop(key)

        # Recursive template of the object and sub elements, if it is a dict or list
        if recursive_template:
            val = self.recursive_template(val, var_override=var_override)

        return val

    def recursive_template(self, item, var_override=None):

        # Only perform recursive templating if the input object
        # is a dictionary or list
        if not isinstance(item, (dict, list)):
            return self.template_if_string(item, var_override=var_override)

        visited = set()
        item_list = [item]

        while len(item_list) > 0:
            current = item_list.pop()

            # Check if we've seen this object before
            if id(current) in visited:
                continue

            # Save this to the visited list, so we don't revisit again, if there is a loop
            # in the origin object
            visited.add(id(current))

            if isinstance(current, dict):
                for key in current:
                    if isinstance(current[key], (dict, list)):
                        item_list.append(current[key])
                    else:
                        current[key] = self.template_if_string(current[key], var_override=var_override)
            elif isinstance(current, list):
                index = 0
                while index < len(current):
                    if isinstance(current[index], (dict, list)):
                        item_list.append(current[index])
                    else:
                        current[index] = self.template_if_string(current[index], var_override=var_override)

                    index = index + 1
            else:
                # Anything non dictionary or list should never have ended up in this list, so this
                # is really an internal error
                raise PipelineRunException(f"Invalid type for templating in recursive_template: {type(current)}")

        return item
