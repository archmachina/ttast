#!/usr/bin/env python3
"""
"""

import argparse
import logging
import os
import sys
import glob
import re
import yaml

from string import Template

from .ttast_exception import *

logger = logging.getLogger(__name__)


def validate(val, message):
    if not val:
        raise ValidationException(message)


def template_if_string(val, mapping):
    if val is not None and isinstance(val, str):
        try:
            template = Template(val)
            return template.substitute(mapping)
        except KeyError as e:
            raise PipelineRunException(f"Missing key in template substitution: {e}") from e

    return val


def parse_bool(obj) -> bool:
    if obj is None:
        raise PipelineRunException("None value passed to parse_bool")

    if isinstance(obj, bool):
        return obj

    obj = str(obj)

    if obj.lower() in ["true", "1"]:
        return True

    if obj.lower() in ["false", "0"]:
        return False

    raise PipelineRunException(f"Unparseable value ({obj}) passed to parse_bool")


def extract_property(spec, key, *, template_map=None, default=None, required=False):
    validate(isinstance(spec, dict), f"Invalid spec passed to extract_property. Must be dict")
    validate(isinstance(template_map, dict) or template_map is None, "Invalid type passed as template_map")

    if key not in spec:
        # Raise exception is the key isn't present, but required
        if required:
            raise KeyError(f'Missing key "{key}" in spec or value is null')

        # If the key is not present, return the default
        return default

    # Retrieve value
    val = spec[key]

    # Template the value, depending on the type
    if val is not None and template_map is not None:
        if isinstance(val, str):
            val = template_if_string(val, template_map)
        elif isinstance(val, list):
            val = [template_if_string(x, template_map) for x in val]
        elif isinstance(val, dict):
            for val_key in val:
                val[val_key] = template_if_string(val[val_key], template_map)

    return val

class TextBlock:
    def __init__(self, block, *, tags=None, filename=""):
        validate(isinstance(tags, (list, set)) or tags is None, "Tags supplied to TextBlock must be a set, list or absent")

        self.block = block

        self.tags = set()
        if tags is not None:
            for tag in tags:
                self.tags.add(tag)

        self.filename = filename

class PipelineStep:
    def __init__(self, step_def, parent):
        validate(isinstance(step_def, dict), "Invalid step_def passed to PipelineStep")
        validate(isinstance(parent, Pipeline), "Invalid parent passed to PipelineStep")

        self.step_def = step_def
        self.parent = parent

        # Extract type
        self.step_type = extract_property(self.step_def, "type", template_map=self.parent.vars)
        validate(isinstance(self.step_type, str) and self.step_type != "", "Step 'type' is required and must be a non empty string")

        # Extract match any tags
        match_any_tags = extract_property(self.step_def, "match_any_tags", template_map=self.parent.vars, default=[])
        validate(isinstance(match_any_tags, list), "Step 'match_any_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_any_tags), "Step 'match_any_tags' must be a list of strings")
        self.match_any_tags = set(match_any_tags)

        # Extract match all tags
        match_all_tags = extract_property(self.step_def, "match_all_tags", template_map=self.parent.vars, default=[])
        validate(isinstance(match_all_tags, list), "Step 'match_all_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_all_tags), "Step 'match_all_tags' must be a list of strings")
        self.match_all_tags = set(match_all_tags)

        # Extract exclude tags
        exclude_tags = extract_property(self.step_def, "exclude_tags", template_map=self.parent.vars, default=[])
        validate(isinstance(exclude_tags, list), "Step 'exclude_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in exclude_tags), "Step 'exclude_tags' must be a list of strings")
        self.exclude_tags = set(exclude_tags)

        # Apply tags
        self.apply_tags = extract_property(self.step_def, "apply_tags", template_map=self.parent.vars, default=[])
        validate(isinstance(self.apply_tags, list), "Step 'apply_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.apply_tags), "Step 'apply_tags' must be a list of strings")

    def process(self):

        if self.step_type == "config":
            self._process_config()
        elif self.step_type == "import":
            self._process_import()
        elif self.step_type == "stdin":
            self._process_stdin()
        elif self.step_type == "stdin_yaml":
            self._process_stdin_yaml()
        elif self.step_type == "stdout":
            self._process_stdout()
        elif self.step_type == "replace":
            self._process_replace()
        else:
            raise PipelineRunException(f"Invalid step type in step {self.step_type}")

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

    def _process_stdin(self):
        split = extract_property(self.step_def, "split", template_map=self.parent.vars)
        validate(isinstance(split, str) or split is None, "Step 'split' must be a string")

        strip = extract_property(self.step_def, "strip", template_map=self.parent.vars, default=False)
        validate(isinstance(strip, (bool, str)), "Step 'strip' must be a bool or str value")
        strip = parse_bool(strip)

        # Read content from stdin
        logger.debug("stdin: reading document from stdin")
        stdin_content = sys.stdin.read()

        # Split if required and convert to a list of documents
        if split is not None and split != "":
            stdin_items = stdin_content.split(split)
        else:
            stdin_items = [stdin_content]

        # strip leading and trailing whitespace, if required
        if strip:
            stdin_items = [x.strip() for x in stdin_items]

        # Add the stdin items to the list of text blocks
        for item in stdin_items:
            self.parent.text_blocks.append(TextBlock(item, tags=self.apply_tags))

    def _process_stdin_yaml(self):
        strip = extract_property(self.step_def, "strip", template_map=self.parent.vars, default=False)
        validate(isinstance(strip, (bool, str)), "Step 'strip' must be a bool or str value")
        strip = parse_bool(strip)

        # Read content from stdin
        logger.debug("stdin_yaml: reading yaml document from stdin")
        stdin_lines = sys.stdin.read().splitlines()

        documents = []
        current = []

        for line in stdin_lines:

            # Determine if we have the beginning of a yaml document
            if line == "---" and len(current) > 0:
                documents.append("\n".join(current))
                current = []

            current.append(line)

        documents.append("\n".join(current))

        # Strip each document, if required
        if strip:
            documents = [x.strip() for x in documents]

        # Add all documents to the pipeline text block list
        for item in documents:
            self.parent.text_blocks.append(TextBlock(item, tags=self.apply_tags))

    def _process_import(self):
        import_files = extract_property(self.step_def, "files", template_map=self.parent.vars)
        validate(isinstance(import_files, list), "Step 'files' must be a list of strings")
        validate(all(isinstance(x, str) for x in import_files), "Step 'files' must be a list of strings")

        recursive = extract_property(self.step_def, "recursive", template_map=self.parent.vars)
        validate(isinstance(recursive, (bool, str)), "Step 'recursive' must be a bool or bool like string")
        recursive = parse_bool(recursive)

        filenames = set()
        for import_file in import_files:
            logger.debug(f"import: processing file glob: {import_file}")
            matches = glob.glob(import_file, recursive=recursive)
            for match in matches:
                filenames.add(match)

        for filename in filenames:
            logger.debug(f"import: reading file {filename}")
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                self.parent.text_blocks.append(TextBlock(content, tags=self.apply_tags, filename=filename))

    def _process_stdout(self):
        prefix = extract_property(self.step_def, "prefix", template_map=self.parent.vars)
        validate(isinstance(prefix, str) or prefix is None, "Step 'prefix' must be a string")

        suffix = extract_property(self.step_def, "suffix", template_map=self.parent.vars)
        validate(isinstance(suffix, str) or suffix is None, "Step 'suffix' must be a string")

        for block in self.parent.text_blocks:

            # Determine if we should be processing this document
            if not self._is_tag_match(block):
                continue

            logger.debug(f"stdout: document tags: {block.tags}")
            logger.debug(f"stdout: document filename: {block.filename}")

            if prefix is not None:
                print(prefix)

            print(block.block)

            if suffix is not None:
                print(suffix)

    def _process_replace(self):
        replace = extract_property(self.step_def, "replace", template_map=self.parent.vars, default={})
        validate(isinstance(replace, list), "Step 'replace' must be a list")
        validate(all(isinstance(x, dict) for x in replace), "Step 'replace' items must be dictionaries")
        for item in replace:
            validate('key' in item and isinstance(item['key'], str), "Step 'replace' items must contain a string 'key' property")
            validate('value' in item and isinstance(item['value'], str), "Step 'replace' items must contain a string 'value' property")

        regex = extract_property(self.step_def, "regex", template_map=self.parent.vars, default=False)
        validate(isinstance(regex, (bool, str)), "Step 'regex' must be a bool, bool like string or absent")
        regex = parse_bool(regex)

        for block in self.parent.text_blocks:

            # Determine if we should be processing this document
            if not self._is_tag_match(block):
                continue

            logger.debug(f"replace: document tags: {block.tags}")
            logger.debug(f"replace: document filename: {block.filename}")

            for replace_item in replace:
                replace_key = replace_item['key']
                replace_value = replace_item['value']

                replace_regex = extract_property(replace_item, "regex", template_map=self.parent.vars, default=False)
                validate(isinstance(replace_regex, (bool, str)), "Replace item 'regex' must be a bool, bool like string or absent")
                replace_regex = parse_bool(replace_regex)

                # replace_value isn't templated by extract_property as it is a list of dictionaries, so it
                # needs to be manually done here
                replace_value = template_if_string(replace_value, self.parent.vars)

                logger.debug(f"replace: replacing: {replace_key} -> {replace_value}")

                if regex or replace_regex:
                    block.block = re.sub(replace_key, replace_value, block.block)
                else:
                    block.block = block.block.replace(replace_key, replace_value)

    def _process_config(self):
        # Read the content from the file and use _process_config_content to do the work
        config_file = str(extract_property(self.step_def, "file", template_map=self.parent.vars))
        validate(isinstance(config_file, str) or config_file is None, "Step 'config_file' must be a string or absent")
        validate(not isinstance(config_file, str) or config_file != "", "Step 'config_file' cannot be empty")

        if config_file is not None:
            logger.debug(f"config: including config from file {config_file}")
            with open(config_file, "r", encoding='utf-8') as file:
                content = file.read()

            self._process_config_content(content)

        # Extract the content var, which can be either a dict or yaml string
        config_content = extract_property(self.step_def, "content", template_map=self.parent.vars)
        validate(isinstance(config_content, (str, dict)) or config_content is None, "Step 'config_content' must be a string, dict or absent")

        # Call _process_config_content, which can determine whether to process as string or dict
        if config_content is not None:
            logger.debug(f"config: including inline config")
            self._process_config_content(config_content)

    def _process_config_content(self, content):
        validate(isinstance(content, (str, dict)), "Included configuration must be a string or dictionary")

        # Parse yaml if it is a string
        if isinstance(content, str):
            content = yaml.safe_load(content)

        validate(isinstance(content, dict), "Parsed configuration is not a dictionary")

        # Extract vars from the config
        config_vars = dict(extract_property(content, "vars", template_map=self.parent.vars, default={}))
        validate(isinstance(config_vars, dict), "Config 'vars' is not a dictionary")

        for config_var_name in config_vars:
            self.parent.set_var(config_var_name, config_vars[config_var_name])

        # Extract pipeline steps from the config
        config_pipeline = extract_property(content, "pipeline", template_map=None, default=[])
        validate(isinstance(config_pipeline, list), "Config 'pipeline' is not a list")

        for step in config_pipeline:
            validate(isinstance(step, dict), "Pipeline entry is not a dictionary")

            self.parent.add_step(step)


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
            step = PipelineStep(self.steps[index], parent=self)

            step.process()

            index = index + 1


def process_args() -> int:
    """
    Processes ttast command line arguments, initialises and runs the pipeline to perform text processing
    """

    # Create parser for command line arguments
    parser = argparse.ArgumentParser(
        prog="ttast", description="Text Transform Assistant", exit_on_error=False
    )

    # Parser configuration
    parser.add_argument(
        "-c", action="append", dest="configs", help="Configuration files"
    )

    parser.add_argument(
        "-d", action="store_true", dest="debug", help="Enable debug output"
    )

    args = parser.parse_args()

    # Capture argument options
    debug = args.debug
    configs = args.configs

    # Logging configuration
    level = logging.WARNING
    if debug:
        level = logging.DEBUG

    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    try:
        # Create the pipeline state
        pipeline = Pipeline()

        # Add each config as a pipeline step, which will read and merge the config
        if configs is not None:
            for config_item in configs:
                step_def = {
                    "type": "config",
                    "file": config_item
                }

                pipeline.add_step(step_def)

        # Start processing the pipeline
        pipeline.process()
    except Exception as e:  # pylint: disable=broad-exception-caught
        if debug:
            logger.error(e, exc_info=True, stack_info=True)
        else:
            logger.error(e)
        return 1

    return 0


def main():
    """
    Entrypoint for the module.
    Minor exception handling is performed, along with return code processing and
    flushing of stdout on program exit.
    """
    try:
        ret = process_args()
        sys.stdout.flush()
        sys.exit(ret)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logging.getLogger(__name__).exception(e)
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
