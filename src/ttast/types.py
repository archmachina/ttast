
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

class HandlerConfig(Handler):
    """
    """
    def parse(self):
        # Read the content from the file and use _process_config_content to do the work
        self.config_file = pop_property(self.state.step_def, "file", template_map=self.state.vars)
        validate(isinstance(self.config_file, str) or self.config_file is None, "Step 'config_file' must be a string or absent")
        validate(not isinstance(self.config_file, str) or self.config_file != "", "Step 'config_file' cannot be empty")

        # Extract the content var, which can be either a dict or yaml string
        self.config_content = pop_property(self.state.step_def, "content", template_map=self.state.vars)
        validate(isinstance(self.config_content, (str, dict)) or self.config_content is None, "Step 'config_content' must be a string, dict or absent")

        # Extract stdin bool, indicating whether to read config from stdin
        self.stdin = pop_property(self.state.step_def, "stdin", template_map=self.state.vars, default=False)
        validate(isinstance(self.stdin, (bool, str)), "Step 'stdin' must be a bool, bool like string or absent")
        self.stdin = parse_bool(self.stdin)

    def is_per_block():
        return False

    def run(self):
        if self.config_file is not None:
            logger.debug(f"config: including config from file {self.config_file}")
            with open(self.config_file, "r", encoding='utf-8') as file:
                content = file.read()

            self._process_config_content(content)

        # Call _process_config_content, which can determine whether to process as string or dict
        if self.config_content is not None:
            logger.debug(f"config: including inline config")
            self._process_config_content(self.config_content)

        if self.stdin:
            # Read configuration from stdin
            logger.debug(f"config: including stdin config")
            stdin_content = sys.stdin.read()
            self._process_config_content(stdin_content)

    def _process_config_content(self, content):
        validate(isinstance(content, (str, dict)), "Included configuration must be a string or dictionary")

        # Don't error on an empty configuration. Just return
        if content == "":
            logger.debug("config: empty configuration. Ignoring.")
            return

        # Parse yaml if it is a string
        if isinstance(content, str):
            content = yaml.safe_load(content)

        validate(isinstance(content, dict), "Parsed configuration is not a dictionary")

        # Extract vars from the config
        config_vars = pop_property(content, "vars", template_map=self.state.vars, default={})
        validate(isinstance(config_vars, dict), "Config 'vars' is not a dictionary")

        for config_var_name in config_vars:
            self.state.pipeline.set_var(config_var_name, config_vars[config_var_name])

        # Extract pipeline steps from the config
        config_pipeline = pop_property(content, "pipeline", template_map=None, default=[])
        validate(isinstance(config_pipeline, list), "Config 'pipeline' is not a list")

        for step in config_pipeline:
            validate(isinstance(step, dict), "Pipeline entry is not a dictionary")

            self.state.pipeline.add_step(step)

        # Validate config has no other properties
        validate(len(content.keys()) == 0, f"Found unknown properties in configuration: {content.keys()}")

class HandlerImport(Handler):
    """
    """
    def parse(self):
        self.import_files = pop_property(self.state.step_def, "files", template_map=self.state.vars)
        validate(isinstance(self.import_files, list), "Step 'files' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.import_files), "Step 'files' must be a list of strings")

        self.recursive = pop_property(self.state.step_def, "recursive", template_map=self.state.vars)
        validate(isinstance(self.recursive, (bool, str)), "Step 'recursive' must be a bool or bool like string")
        self.recursive = parse_bool(self.recursive)

    def is_per_block():
        return False

    def run(self):
        filenames = set()
        for import_file in self.import_files:
            logger.debug(f"import: processing file glob: {import_file}")
            matches = glob.glob(import_file, recursive=self.recursive)
            for match in matches:
                filenames.add(match)

        # Ensure consistency for load order
        filenames = list(filenames)
        filenames.sort()

        # Get apply tags
        apply_tags = self.state.shared.get("apply_tags", [])

        for filename in filenames:
            logger.debug(f"import: reading file {filename}")
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                new_block = TextBlock(content, tags=apply_tags)
                new_block.meta["import_filename"] = filename
                self.state.pipeline.add_block(new_block)

class HandlerMeta(Handler):
    """
    """
    def parse(self):
        self.vars = pop_property(self.state.step_def, "vars", template_map=self.state.vars)
        validate(isinstance(self.vars, dict), "Step 'vars' must be a dictionary of strings")
        validate(all(isinstance(x, str) for x in self.vars), "Step 'vars' must be a dictionary of strings")

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"meta: document tags: {self.state.block.tags}")
        logger.debug(f"meta: document meta: {self.state.block.meta}")

        for key in self.vars:
            self.state.block.meta[key] = self.vars[key]

class HandlerReplace(Handler):
    """
    """
    def parse(self):
        self.replace = pop_property(self.state.step_def, "replace", template_map=self.state.vars, default={})
        validate(isinstance(self.replace, list), "Step 'replace' must be a list")
        validate(all(isinstance(x, dict) for x in self.replace), "Step 'replace' items must be dictionaries")
        for item in self.replace:
            validate('key' in item and isinstance(item['key'], str), "Step 'replace' items must contain a string 'key' property")
            validate('value' in item and isinstance(item['value'], str), "Step 'replace' items must contain a string 'value' property")

        self.regex = pop_property(self.state.step_def, "regex", template_map=self.state.vars, default=False)
        validate(isinstance(self.regex, (bool, str)), "Step 'regex' must be a bool, bool like string or absent")
        self.regex = parse_bool(self.regex)

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"replace: document tags: {self.state.block.tags}")
        logger.debug(f"replace: document meta: {self.state.block.meta}")

        for replace_item in self.replace:
            # Copy the dictionary as we'll change it when removing values
            replace_item = replace_item.copy()

            replace_key = replace_item['key']
            replace_value = replace_item['value']

            replace_regex = pop_property(replace_item, "regex", template_map=self.state.vars, default=False)
            validate(isinstance(replace_regex, (bool, str)), "Replace item 'regex' must be a bool, bool like string or absent")
            replace_regex = parse_bool(replace_regex)

            # replace_value isn't templated by pop_property as it is a list of dictionaries, so it
            # needs to be manually done here
            replace_value = template_if_string(replace_value, self.state.vars)

            logger.debug(f"replace: replacing regex({self.regex or replace_regex}): {replace_key} -> {replace_value}")

            if self.regex or replace_regex:
                self.state.block.text = re.sub(replace_key, replace_value, self.state.block.text)
            else:
                self.state.block.text = self.state.block.text.replace(replace_key, replace_value)

class HandlerSplitYaml(Handler):
    """
    """
    def parse(self):
        self.strip = pop_property(self.state.step_def, "strip", template_map=self.state.vars, default=False)
        validate(isinstance(self.strip, (bool, str)), "Step 'strip' must be a bool or str value")
        self.strip = parse_bool(self.strip)

    def is_per_block():
        return True

    def run(self):
        # Read content from stdin
        logger.debug(f"split_yaml: document tags: {self.state.block.tags}")
        logger.debug(f"split_yaml: document meta: {self.state.block.meta}")

        lines = self.state.block.text.splitlines()
        documents = []
        current = []

        for line in lines:

            # Determine if we have the beginning of a yaml document
            if line == "---" and len(current) > 0:
                documents.append("\n".join(current))
                current = []

            current.append(line)

        documents.append("\n".join(current))

        # Strip each document, if required
        if self.strip:
            documents = [x.strip() for x in documents]

        # Add all documents to the pipeline text block list
        for item in documents:
            new_block = TextBlock(item)
            new_block.meta = self.state.block.meta.copy()
            new_block.tags = self.state.block.tags.copy()

            self.state.pipeline.add_block(new_block)

        # Remove the original source block from the list
        self.state.pipeline.remove_block(self.state.block)

        logger.debug(f"split_yaml: output 1 document -> {len(documents)} documents")

class HandlerStdin(Handler):
    """
    """
    def parse(self):
        self.split = pop_property(self.state.step_def, "split", template_map=self.state.vars)
        validate(isinstance(self.split, str) or self.split is None, "Step 'split' must be a string")

        self.strip = pop_property(self.state.step_def, "strip", template_map=self.state.vars, default=False)
        validate(isinstance(self.strip, (bool, str)), "Step 'strip' must be a bool or str value")
        self.strip = parse_bool(self.strip)

    def is_per_block():
        return False

    def run(self):
        # Read content from stdin
        logger.debug("stdin: reading document from stdin")
        stdin_content = sys.stdin.read()

        # Split if required and convert to a list of documents
        if self.split is not None and self.split != "":
            stdin_items = stdin_content.split(self.split)
        else:
            stdin_items = [stdin_content]

        # strip leading and trailing whitespace, if required
        if self.strip:
            stdin_items = [x.strip() for x in stdin_items]

        # Get apply tags
        apply_tags = self.state.shared.get("apply_tags", [])

        # Add the stdin items to the list of text blocks
        for item in stdin_items:
            self.state.pipeline.add_block(TextBlock(item, tags=apply_tags))

class HandlerStdout(Handler):
    """
    """
    def parse(self):
        self.prefix = pop_property(self.state.step_def, "prefix", template_map=self.state.vars)
        validate(isinstance(self.prefix, str) or self.prefix is None, "Step 'prefix' must be a string")

        self.suffix = pop_property(self.state.step_def, "suffix", template_map=self.state.vars)
        validate(isinstance(self.suffix, str) or self.suffix is None, "Step 'suffix' must be a string")

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"stdout: document tags: {self.state.block.tags}")
        logger.debug(f"stdout: document meta: {self.state.block.meta}")

        if self.prefix is not None:
            print(self.prefix)

        print(self.state.block.text)

        if self.suffix is not None:
            print(self.suffix)

class HandlerTemplate(Handler):
    """
    """
    def parse(self):
        self.vars = pop_property(self.state.step_def, "vars", template_map=self.state.vars)
        validate(isinstance(self.vars, dict) or self.vars is None, "Step 'vars' must be a dictionary or absent")

        self.merge_vars = pop_property(self.state.step_def, "merge_vars", template_map=self.state.vars, default=True)
        validate(isinstance(self.merge_vars, (str, bool)), "Step 'merge_vars' must be a bool, bool like string or absent")
        self.merge_vars = parse_bool(self.merge_vars)

    def is_per_block():
        return True

    def run(self):
        template_vars = self.state.vars.copy()

        if self.vars is not None:
            for key in self.vars:
                template_vars[key] = self.vars[key]

        environment = jinja2.Environment()

        logger.debug(f"template: document tags: {self.state.block.tags}")
        logger.debug(f"template: document meta: {self.state.block.meta}")

        template = environment.from_string(self.state.block.text)
        self.state.block.text = template.render(template_vars)
