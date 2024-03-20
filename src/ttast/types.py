
import yaml
import sys
import re
import glob
import os
import inspect
import copy

from .util import *

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
        self._pre_handlers = [
            PreHandlerMatchTags,
            PreHandlerWhen
        ]
        self._post_handlers = [
            PostHandlerTags
        ]
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

        state = PipelineStepState(step_def, self, block)

        #
        # Parsing
        #

        # Initialise and parse pre handlers
        pre_handlers = [x() for x in self._pre_handlers]
        for pre in pre_handlers:
            pre.init(state)
            pre.parse()

        # Initialise and parse post handlers
        # Parsing should happen before the main handler is parsed or executed
        post_handlers = [x() for x in self._post_handlers]
        for post in post_handlers:
            post.init(state)
            post.parse()

        # Initialise and parse the main handler
        instance = handler()
        instance.init(state)
        remainder = instance.parse()

        # At this point, there should be no properties left in the dictionary as all of the handlers should have
        # extracted their own properties.
        validate(len(state.step_def.keys()) == 0, f"Unknown properties on step definition: {list(state.step_def.keys())}")

        #
        # Execution
        #

        # Run any preprocessing handlers
        for pre in pre_handlers:
            pre.run()
            if state.stop_processing:
                return

        # Perform processing for the main handler
        instance.run()
        if state.stop_processing:
            return

        # Run any post processing handlers
        for post in post_handlers:
            post.run()
            if state.stop_processing:
                return

class PipelineStepState:
    def __init__(self, step_def, pipeline, block=None):
        validate(isinstance(step_def, dict), "Invalid step_def passed to PipelineStepState")
        validate(isinstance(pipeline, Pipeline) or pipeline is None, "Invalid pipeline passed to PipelineStepState")
        validate(isinstance(block, TextBlock) or block is None, "Invalid block passed to PipelineStepState")

        self.step_def = step_def.copy()
        self.block = block
        self.pipeline = pipeline
        self.stop_processing = False

        # Shared state that handlers can use to pass information
        self.shared = {}

        # Create new vars for the instance, based on the pipeline vars, plus including
        # any block vars, if present
        self.vars = self.pipeline.copy_vars()
        if block is not None:
            self.vars = copy.deepcopy(self.vars)
            self.vars["meta"] = copy.deepcopy(block.meta)
            self.vars["tags"] = copy.deepcopy(block.tags)

class PrePostHandler:
    def init(self, state):
        validate(isinstance(state, PipelineStepState), "Invalid step state passed to PrePostHandler")

        self.state = state

    def parse(self):
        raise PipelineRunException("parse undefined in PrePostHandler")

    def run(self):
        raise PipelineRunException("run undefined in PrePostHandler")

class PostHandlerTags(PrePostHandler):
    def parse(self):
        # Apply tags
        self.apply_tags = pop_property(self.state.step_def, "apply_tags", template_map=self.state.vars, default=[])
        validate(isinstance(self.apply_tags, list), "Step 'apply_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.apply_tags), "Step 'apply_tags' must be a list of strings")

        # Save apply tags here so that other handlers can access it
        self.state.shared["apply_tags"] = self.apply_tags

    def run(self):
        if self.state.block is not None:
            for tag in self.apply_tags:
                self.state.block.tags.add(tag)

class PreHandlerWhen(PrePostHandler):
    def parse(self):
        # When condition
        self.when = pop_property(self.state.step_def, "when", template_map=self.state.vars, default=[])
        validate(isinstance(self.when, (list, str)), "Step 'when' must be a string or list of strings")
        if isinstance(self.when, str):
            self.when = [self.when]
        validate(all(isinstance(x, str) for x in self.when), "Step 'when' must be a string or list of strings")

    def run(self):
        if len(self.when) > 0:
            environment = jinja2.Environment()
            for condition in self.when:
                template = environment.from_string("{{" + condition + "}}")
                if not parse_bool(template.render(self.state.vars)):
                    self.state.stop_processing = True

class PreHandlerMatchTags(PrePostHandler):
    def parse(self):
        # Extract match any tags
        self.match_any_tags = pop_property(self.state.step_def, "match_any_tags", template_map=self.state.vars, default=[])
        validate(isinstance(self.match_any_tags, list), "Step 'match_any_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.match_any_tags), "Step 'match_any_tags' must be a list of strings")
        self.match_any_tags = set(self.match_any_tags)

        # Extract match all tags
        match_all_tags = pop_property(self.state.step_def, "match_all_tags", template_map=self.state.vars, default=[])
        validate(isinstance(match_all_tags, list), "Step 'match_all_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_all_tags), "Step 'match_all_tags' must be a list of strings")
        self.match_all_tags = set(match_all_tags)

        # Extract exclude tags
        self.exclude_tags = pop_property(self.state.step_def, "exclude_tags", template_map=self.state.vars, default=[])
        validate(isinstance(self.exclude_tags, list), "Step 'exclude_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.exclude_tags), "Step 'exclude_tags' must be a list of strings")
        self.exclude_tags = set(self.exclude_tags)

    def run(self):
        if len(self.match_any_tags) > 0:
            # If there are any 'match_any_tags', then at least one of them has to match with the document
            if len(self.match_any_tags.intersection(self.state.block.tags)) == 0:
                self.state.stop_processing = True

        if len(self.match_all_tags) > 0:
            # If there are any 'match_all_tags', then all of those tags must match the document
            for tag in self.match_all_tags:
                if tag not in self.state.block.tags:
                    self.state.stop_processing = True

        if len(self.exclude_tags) > 0:
            # If there are any exclude tags and any are present in the block, it isn't a match
            for tag in self.exclude_tags:
                if tag in self.state.block.tags:
                    self.state.stop_processing = True

class Handler:
    def is_per_block(self):
        raise PipelineRunException("is_per_block undefined in Handler")

    def init(self, state):
        validate(isinstance(state, PipelineStepState), "Invalid step state passed to Handler")

        self.state = state

    def parse(self):
        raise PipelineRunException("parse undefined in Handler")

    def run(self):
        raise PipelineRunException("run undefined in Handler")
