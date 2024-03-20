
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
        validate((inspect.isclass(handler) and issubclass(handler, Handler)), "Invalid handler passed to PipelineStepInstance")
        validate(isinstance(block, TextBlock) or block is None, "Invalid block passed to PipelineStepInstance")

        self.step_def = step_def.copy()
        self.pipeline = pipeline
        self.handler = handler
        self.block = block

        # Create new vars for the instance, based on the pipeline vars, plus including
        # any block vars, if present
        self.vars = copy.deepcopy(self.pipeline.vars)
        if block is not None:
            self.vars["meta"] = copy.deepcopy(block.meta)
            self.vars["tags"] = copy.deepcopy(block.tags)

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

        instance = self.handler()
        instance.init(self)

        # Parse the step definition
        instance.parse()

        # The parse function should extract all of the relevant properties from the step_def, leaving any unknown properties.
        # Check that there are no properties left in the step definition
        validate(len(self.step_def.keys()) == 0, f"Unknown properties on step definition: {list(self.step_def.keys())}")

        # Perform processing for this handler
        instance.run()

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

class Handler:
    def is_per_block(self):
        raise PipelineRunException("is_per_block undefined in Handler")

    def init(self, step_instance):
        validate(isinstance(step_instance, PipelineStepInstance), "Invalid step_instance passed to Handler")

        self.step_instance = step_instance
        self.pipeline = self.step_instance.pipeline
        self.block = step_instance.block

    def parse(self):
        raise PipelineRunException("parse undefined in Handler")

    def run(self):
        raise PipelineRunException("run undefined in Handler")

class HandlerConfig(Handler):
    """
    """
    def parse(self):
        # Read the content from the file and use _process_config_content to do the work
        self.config_file = pop_property(self.step_instance.step_def, "file", template_map=self.step_instance.vars)
        validate(isinstance(self.config_file, str) or self.config_file is None, "Step 'config_file' must be a string or absent")
        validate(not isinstance(self.config_file, str) or self.config_file != "", "Step 'config_file' cannot be empty")

        # Extract the content var, which can be either a dict or yaml string
        self.config_content = pop_property(self.step_instance.step_def, "content", template_map=self.step_instance.vars)
        validate(isinstance(self.config_content, (str, dict)) or self.config_content is None, "Step 'config_content' must be a string, dict or absent")

        # Extract stdin bool, indicating whether to read config from stdin
        self.stdin = pop_property(self.step_instance.step_def, "stdin", template_map=self.step_instance.vars, default=False)
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
        config_vars = pop_property(content, "vars", template_map=self.step_instance.vars, default={})
        validate(isinstance(config_vars, dict), "Config 'vars' is not a dictionary")

        for config_var_name in config_vars:
            self.pipeline.vars[config_var_name] = config_vars[config_var_name]

        # Extract pipeline steps from the config
        config_pipeline = pop_property(content, "pipeline", template_map=None, default=[])
        validate(isinstance(config_pipeline, list), "Config 'pipeline' is not a list")

        for step in config_pipeline:
            validate(isinstance(step, dict), "Pipeline entry is not a dictionary")

            self.pipeline.add_step(step)

        # Validate config has no other properties
        validate(len(content.keys()) == 0, f"Found unknown properties in configuration: {content.keys()}")

class HandlerImport(Handler):
    """
    """
    def parse(self):
        self.import_files = pop_property(self.step_instance.step_def, "files", template_map=self.step_instance.vars)
        validate(isinstance(self.import_files, list), "Step 'files' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.import_files), "Step 'files' must be a list of strings")

        self.recursive = pop_property(self.step_instance.step_def, "recursive", template_map=self.step_instance.vars)
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

        for filename in filenames:
            logger.debug(f"import: reading file {filename}")
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                new_block = TextBlock(content, tags=self.step_instance.apply_tags)
                new_block.meta["import_filename"] = filename
                self.pipeline.blocks.append(new_block)

class HandlerMeta(Handler):
    """
    """
    def parse(self):
        self.vars = pop_property(self.step_instance.step_def, "vars", template_map=self.step_instance.vars)
        validate(isinstance(self.vars, dict), "Step 'vars' must be a dictionary of strings")
        validate(all(isinstance(x, str) for x in self.vars), "Step 'vars' must be a dictionary of strings")

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"meta: document tags: {self.block.tags}")
        logger.debug(f"meta: document meta: {self.block.meta}")

        for key in self.vars:
            self.block.meta[key] = self.vars[key]

class HandlerReplace(Handler):
    """
    """
    def parse(self):
        self.replace = pop_property(self.step_instance.step_def, "replace", template_map=self.step_instance.vars, default={})
        validate(isinstance(self.replace, list), "Step 'replace' must be a list")
        validate(all(isinstance(x, dict) for x in self.replace), "Step 'replace' items must be dictionaries")
        for item in self.replace:
            validate('key' in item and isinstance(item['key'], str), "Step 'replace' items must contain a string 'key' property")
            validate('value' in item and isinstance(item['value'], str), "Step 'replace' items must contain a string 'value' property")

        self.regex = pop_property(self.step_instance.step_def, "regex", template_map=self.step_instance.vars, default=False)
        validate(isinstance(self.regex, (bool, str)), "Step 'regex' must be a bool, bool like string or absent")
        self.regex = parse_bool(self.regex)

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"replace: document tags: {self.block.tags}")
        logger.debug(f"replace: document meta: {self.block.meta}")

        for replace_item in self.replace:
            # Copy the dictionary as we'll change it when removing values
            replace_item = replace_item.copy()

            replace_key = replace_item['key']
            replace_value = replace_item['value']

            replace_regex = pop_property(replace_item, "regex", template_map=self.step_instance.vars, default=False)
            validate(isinstance(replace_regex, (bool, str)), "Replace item 'regex' must be a bool, bool like string or absent")
            replace_regex = parse_bool(replace_regex)

            # replace_value isn't templated by pop_property as it is a list of dictionaries, so it
            # needs to be manually done here
            replace_value = template_if_string(replace_value, self.step_instance.vars)

            logger.debug(f"replace: replacing regex({self.regex or replace_regex}): {replace_key} -> {replace_value}")

            if self.regex or replace_regex:
                self.block.text = re.sub(replace_key, replace_value, self.block.text)
            else:
                self.block.text = self.block.text.replace(replace_key, replace_value)

class HandlerSplitYaml(Handler):
    """
    """
    def parse(self):
        self.strip = pop_property(self.step_instance.step_def, "strip", template_map=self.step_instance.vars, default=False)
        validate(isinstance(self.strip, (bool, str)), "Step 'strip' must be a bool or str value")
        self.strip = parse_bool(self.strip)

    def is_per_block():
        return True

    def run(self):
        # Read content from stdin
        logger.debug(f"split_yaml: document tags: {self.block.tags}")
        logger.debug(f"split_yaml: document meta: {self.block.meta}")

        lines = self.block.text.splitlines()
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
            new_block.meta = self.block.meta.copy()
            new_block.tags = self.block.tags.copy()

            self.pipeline.blocks.append(new_block)

        # Remove the original source block from the list
        self.pipeline.blocks.remove(self.block)

        logger.debug(f"split_yaml: output 1 document -> {len(documents)} documents")

class HandlerStdin(Handler):
    """
    """
    def parse(self):
        self.split = pop_property(self.step_instance.step_def, "split", template_map=self.step_instance.vars)
        validate(isinstance(self.split, str) or self.split is None, "Step 'split' must be a string")

        self.strip = pop_property(self.step_instance.step_def, "strip", template_map=self.step_instance.vars, default=False)
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

        # Add the stdin items to the list of text blocks
        for item in stdin_items:
            self.pipeline.blocks.append(TextBlock(item, tags=self.step_instance.apply_tags))

class HandlerStdout(Handler):
    """
    """
    def parse(self):
        self.prefix = pop_property(self.step_instance.step_def, "prefix", template_map=self.step_instance.vars)
        validate(isinstance(self.prefix, str) or self.prefix is None, "Step 'prefix' must be a string")

        self.suffix = pop_property(self.step_instance.step_def, "suffix", template_map=self.step_instance.vars)
        validate(isinstance(self.suffix, str) or self.suffix is None, "Step 'suffix' must be a string")

    def is_per_block():
        return True

    def run(self):
        logger.debug(f"stdout: document tags: {self.block.tags}")
        logger.debug(f"stdout: document meta: {self.block.meta}")

        if self.prefix is not None:
            print(self.prefix)

        print(self.block.text)

        if self.suffix is not None:
            print(self.suffix)

class HandlerTemplate(Handler):
    """
    """
    def parse(self):
        self.vars = pop_property(self.step_instance.step_def, "vars", template_map=self.step_instance.vars)
        validate(isinstance(self.vars, dict) or self.vars is None, "Step 'vars' must be a dictionary or absent")

        self.merge_vars = pop_property(self.step_instance.step_def, "merge_vars", template_map=self.step_instance.vars, default=True)
        validate(isinstance(self.merge_vars, (str, bool)), "Step 'merge_vars' must be a bool, bool like string or absent")
        self.merge_vars = parse_bool(self.merge_vars)

    def is_per_block():
        return True

    def run(self):
        template_vars = self.step_instance.vars.copy()

        if self.vars is not None:
            for key in self.vars:
                template_vars[key] = self.vars[key]

        environment = jinja2.Environment()

        logger.debug(f"template: document tags: {self.block.tags}")
        logger.debug(f"template: document meta: {self.block.meta}")

        template = environment.from_string(self.block.text)
        self.block.text = template.render(template_vars)
