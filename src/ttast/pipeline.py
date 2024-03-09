
import os

from .util import *
from .exception import *
from . import steps

logger = logging.getLogger(__name__)

class TextBlock:
    def __init__(self, block, *, tags=None):
        validate(isinstance(tags, (list, set)) or tags is None, "Tags supplied to TextBlock must be a set, list or absent")

        self.block = block

        self.tags = set()
        if tags is not None:
            for tag in tags:
                self.tags.add(tag)

        self.meta = {}


class Pipeline:
    def __init__(self):
        self.vars = os.environ.copy()
        self.steps = []
        self.text_blocks = []

    def set_var(self, name, value):
        # No templating here - It's up to the individual type processor to determine when templating is required
        self.vars[name] = value

    def add_step(self, step_def):
        validate(isinstance(step_def, dict), "Invalid step definition passed to add_step")

        if len(self.steps) > 100:
            raise PipelineRunException("Reached limit of 100 steps in pipeline. This is a safe guard to prevent infinite recursion")

        self.steps.append(step_def)

    def process(self):

        # This is a while loop with index to allow the pipeline to be appended to during processing
        index = 0
        while index < len(self.steps):
            # Pipeline Step creation should be deferred until it's actually needed (here)
            # as the ctor may rely on variables updated from prior steps
            step = PipelineStep(self.steps[index], pipeline=self)

            step.process()

            index = index + 1


class PipelineStep:
    """
    Represents the common configuration for a step, independent of a text block and specific
    configuration type (e.g. meta, import)
    """
    def __init__(self, step_def, pipeline):
        validate(isinstance(step_def, dict), "Invalid step_def passed to PipelineStep")
        validate(isinstance(pipeline, Pipeline), "Invalid pipeline passed to PipelineStep")

        step_def = step_def.copy()
        self.step_def = step_def

        self.pipeline = pipeline

        # Extract type
        self.step_type = pop_property(step_def, "type", template_map=self.pipeline.vars)
        validate(isinstance(self.step_type, str) and self.step_type != "", "Step 'type' is required and must be a non empty string")

        # Extract match any tags
        match_any_tags = pop_property(step_def, "match_any_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(match_any_tags, list), "Step 'match_any_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_any_tags), "Step 'match_any_tags' must be a list of strings")
        self.match_any_tags = set(match_any_tags)

        # Extract match all tags
        match_all_tags = pop_property(step_def, "match_all_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(match_all_tags, list), "Step 'match_all_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_all_tags), "Step 'match_all_tags' must be a list of strings")
        self.match_all_tags = set(match_all_tags)

        # Extract exclude tags
        exclude_tags = pop_property(step_def, "exclude_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(exclude_tags, list), "Step 'exclude_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in exclude_tags), "Step 'exclude_tags' must be a list of strings")
        self.exclude_tags = set(exclude_tags)

        # Apply tags
        self.apply_tags = pop_property(step_def, "apply_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(self.apply_tags, list), "Step 'apply_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.apply_tags), "Step 'apply_tags' must be a list of strings")

    def process(self):

        pipeline_steps = list()
        handler = steps.get_handler(self.step_type)

        if not handler.per_block():
            step_instance = PipelineStepInstance(pipeline_step=self)
            pipeline_steps.append(handler(step_instance=step_instance))

        else:
            blocks = self._get_match_blocks()
            for block in blocks:
                step_instance = PipelineStepInstance(pipeline_step=self, block=block)

                pipeline_steps.append(handler(step_instance=step_instance))

            # Apply tags to any relevant blocks
            for block in blocks:
                for tag in self.apply_tags:
                    block.tags.add(tag)

        for instance in pipeline_steps:
            instance.process()

    def _get_match_blocks(self):
        matches = list()

        for block in self.pipeline.text_blocks:
            # Determine if we should be processing this document
            if not self._is_tag_match(block):
                continue

            matches.append(block)

        return matches

    def _is_tag_match(self, text_block):
        validate(isinstance(text_block, TextBlock), "Invalid text_block passed to _is_tag_match")

        if len(self.match_any_tags) > 0:
            # If there are any 'match_any_tags', then at least one of them has to match with the document
            if len(self.match_any_tags.intersection(text_block.tags)) == 0:
                return False

        if len(self.match_all_tags) > 0:
            # If there are any 'match_all_tags', then all of those tags must match the document
            for tag in self.match_all_tags:
                if tag not in text_block.tags:
                    return False

        if len(self.exclude_tags) > 0:
            # If there are any exclude tags and any are present in the block, it isn't a match
            for tag in self.exclude_tags:
                if tag in text_block.tags:
                    return False

        return True

class PipelineStepInstance:
    def __init__(self, pipeline_step, block=None):
        validate(isinstance(pipeline_step, PipelineStep), "Invalid pipeline step passed to PipelineStepInstance")
        validate(isinstance(block, TextBlock) or block is None, "Invalid block passed to PipelineStepInstance")

        self.block = block
        self.pipeline_step = pipeline_step
        self.pipeline = pipeline_step.pipeline

        self.vars = self.pipeline.vars.copy()

        if block is not None:
            self.vars["meta"] = block.meta
            self.vars["tags"] = block.tags

        self.vars["env"] = os.environ.copy()
