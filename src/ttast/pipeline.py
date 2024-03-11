
import os

from .util import *
from .exception import *
from . import steps

logger = logging.getLogger(__name__)


def process_pipeline(pipeline_steps):
    validate(isinstance(pipeline_steps, list) and all(isinstance(x, dict) for x in pipeline_steps),
             "Pipeline steps passed to process_pipeline must be a list of dictionaries")

    pipeline = Pipeline()
    pipeline.steps = pipeline_steps

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
        handler = steps.get_handler_class(step_type)

        # Create an instance per block for the step type, or a single instance for step types
        # that are not per block.
        if handler.per_block():
            # Create a copy of blocks to allow steps to alter the block list while we iterate
            block_list_copy = pipeline.blocks.copy()

            for block in block_list_copy:
                instance = PipelineStepInstance(step_def, pipeline=pipeline, handler=handler, block=block)
                instance.process()
        else:
            instance = PipelineStepInstance(step_def, pipeline=pipeline, handler=handler)
            instance.process()


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
        self.vars = {
            "env": os.environ.copy()
        }
        self.steps = []
        self.blocks = []

    def add_step(self, step_def):
        validate(isinstance(step_def, dict), "Invalid step definition passed to add_step")

        if len(self.steps) > 100:
            raise PipelineRunException("Reached limit of 100 steps in pipeline. This is a safe guard to prevent infinite recursion")

        self.steps.append(step_def)


class PipelineStepInstance:
    def __init__(self, step_def, pipeline, handler, block=None):
        validate(isinstance(step_def, dict), "Invalid step_Def passed to PipelineStepInstance")
        validate(isinstance(pipeline, Pipeline), "Invalid pipeline passed to PipelineStepInstance")
        validate(isinstance(handler, type), "Invalid handler passed to PipelineStepInstance")
        validate(isinstance(block, TextBlock) or block is None, "Invalid block passed to PipelineStepInstance")

        self.step_def = step_def.copy()
        self.pipeline = pipeline
        self.handler = handler
        self.block = block

        # Create new vars for the instance, based on the pipeline vars, plus including
        # any block vars, if present
        self.vars = self.pipeline.vars.copy()
        if block is not None:
            self.vars["meta"] = block.meta
            self.vars["tags"] = block.tags

        # Extract match any tags
        match_any_tags = pop_property(self.step_def, "match_any_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(match_any_tags, list), "Step 'match_any_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_any_tags), "Step 'match_any_tags' must be a list of strings")
        self.match_any_tags = set(match_any_tags)

        # Extract match all tags
        match_all_tags = pop_property(self.step_def, "match_all_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(match_all_tags, list), "Step 'match_all_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_all_tags), "Step 'match_all_tags' must be a list of strings")
        self.match_all_tags = set(match_all_tags)

        # Extract exclude tags
        exclude_tags = pop_property(self.step_def, "exclude_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(exclude_tags, list), "Step 'exclude_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in exclude_tags), "Step 'exclude_tags' must be a list of strings")
        self.exclude_tags = set(exclude_tags)

        # Apply tags
        self.apply_tags = pop_property(self.step_def, "apply_tags", template_map=self.pipeline.vars, default=[])
        validate(isinstance(self.apply_tags, list), "Step 'apply_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.apply_tags), "Step 'apply_tags' must be a list of strings")

        # When condition
        self.when = pop_property(self.step_def, "when", template_map=self.pipeline.vars, default=[])
        validate(isinstance(self.when, (list, str)), "Step 'when' must be a string or list of strings")
        if isinstance(self.when, str):
            self.when = [self.when]
        validate(all(isinstance(x, str) for x in self.when), "Step 'when' must be a string or list of strings")

    def process(self):

        if not self._should_process():
            return

        handler_instance = self.handler(self)

        # The ctor should extract all of the relevant properties from the step_def, leaving any unknown properties.
        # Check that there are no properties left in the step definition
        validate(len(self.step_def.keys()) == 0, f"Unknown properties on step definition: {list(self.step_def.keys())}")

        handler_instance.process()

        if self.block is not None:
            for tag in self.apply_tags:
                self.block.tags.add(tag)

    def _should_process(self):
        if len(self.match_any_tags) > 0:
            # If there are any 'match_any_tags', then at least one of them has to match with the document
            if len(self.match_any_tags.intersection(self.block.tags)) == 0:
                return False

        if len(self.match_all_tags) > 0:
            # If there are any 'match_all_tags', then all of those tags must match the document
            for tag in self.match_all_tags:
                if tag not in self.block.tags:
                    return False

        if len(self.exclude_tags) > 0:
            # If there are any exclude tags and any are present in the block, it isn't a match
            for tag in self.exclude_tags:
                if tag in self.block.tags:
                    return False

        if len(self.when) > 0:
            environment = jinja2.Environment()
            for condition in self.when:
                template = environment.from_string("{{" + condition + "}}")
                if not parse_bool(template.render(self.vars)):
                    return False

        return True
