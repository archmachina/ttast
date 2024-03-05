#!/usr/bin/env python3
"""
"""

import argparse
import logging
import os
import sys

import yaml

from string import Template

from .ttast_exception import PipelineRunException

logger = logging.getLogger(__name__)


def assert_type(obj, obj_type, message):
    if not isinstance(obj, obj_type):
        raise PipelineRunException(message)


def assert_not_none(obj, message):
    if obj is None:
        raise PipelineRunException(message)


def assert_not_emptystr(obj, message):
    if obj is None or (isinstance(obj, str) and obj == ""):
        raise PipelineRunException(message)


def template_if_string(val, mapping):
    if val is not None and isinstance(val, str):
        try:
            template = Template(val)
            return template.substitute(mapping)
        except KeyError as e:
            raise PipelineRunException(f"Missing key in template substitution: {e}") from e

    return val


def step_extract_property(spec, key, *, template_map, failemptystr=False, default=None):
    # Check that we have a valid spec
    if spec is None or not isinstance(spec, dict):
        raise PipelineRunException("spec is missing or is not a dictionary")

    # Check type for template_map
    if template_map is not None and not isinstance(template_map, dict):
        raise PipelineRunException("Invalid type passed as template_map")

    # Handle a missing key in the spec
    if key not in spec or spec[key] is None:
        # Key is not present or the value is null/None
        # Return the default, if specified
        if default is not None:
            return default

        # Key is not present or null and no default, so raise an exception
        raise KeyError(f'Missing key "{key}" in spec or value is null')

    # Retrieve value
    val = spec[key]

    # string specific processing
    if val is not None and isinstance(val, str):
        # Template the string
        if template_map is not None:
            val = template_if_string(val, template_map)

        # Check if we have an empty string and should fail
        if failemptystr and val == "":
            raise PipelineRunException(
                f'Value for key "{key}" is empty, but a value is required'
            )

    # Perform string substitution for other types
    if template_map is not None and val is not None:
        if isinstance(val, list):
            val = [template_if_string(x, template_map) for x in val]

        if isinstance(val, dict):
            for val_key in val.keys():
                val[val_key] = template_if_string(val[val_key], template_map)

    return val
    

class PipelineStep:
    def __init__(self, step_def, parent):
        assert_type(step_def, dict, "Invalid step_def passed to PipelineStep")
        assert_not_emptystr(step_def, "Empty step_def passed to PipelineStep")
        assert_type(parent, Pipeline, "Invalid parent passed to PipelineStep")

        self.step_def = step_def
        self.parent = parent
    
    def process(self):
        # Extract type
        step_type = str(step_extract_property(self.step_def, "type", template_map=self.parent.vars))

        if step_type == "config":
            self._process_config()
        elif step_type == "stdin":
            self._process_stdin()
        elif step_type == "stdout":
            self._process_stdout()
        elif step_type == "replace":
            self._process_replace()
        else:
            raise PipelineRunException(f"Invalid step type in step {step_type}")

    def _process_stdin(self):
        pass

    def _process_stdout(self):
        pass

    def _process_replace(self):
        pass

    def _process_config(self):
        config_file = str(step_extract_property(self.step_def, "file", template_map=self.parent.vars, default=""))
        if config_file is not None and config_file != "":
            with open(config_file, "r", encoding='utf-8') as file:
                content = file.read()

            self._process_config_content(content)

        config_content = str(step_extract_property(self.step_def, "content", template_map=self.parent.vars, default=""))
        if config_content is not None and config_content != "":
            self._process_config_content(content)

    def _process_config_content(self, content):
        config_content = yaml.safe_load(content)

        assert_type(config_content, dict, "Loaded config is not a dict")

        # Extract vars from the config
        config_vars = dict(step_extract_property(config_content, "vars", template_map=self.parent.vars, default={}))
        for config_var_name in config_vars:
            self.parent.set_var(config_var_name, config_vars[config_var_name])

        # Extract pipeline steps from the config
        config_pipeline = list(step_extract_property(config_content, "pipeline", template_map=None, default=[]))
        for step in config_pipeline:
            assert_type(step, dict, "Pipeline entry is not a dictionary")

            self.parent.add_step(step)

class Pipeline:
    def __init__(self):
        self.vars = os.environ.copy()
        self.steps = []

    def set_var(self, name, value):
        # if value is None:
        #     value = ""

        # template = Template(value)
        # value = template.substitute(self.vars)

        self.vars[name] = value

    def add_step(self, step_def):
        assert_type(step_def, dict, "Invalid step definition passed to add_step")

        if len(self.steps) > 100:
            raise PipelineRunException("Reached limit of 100 steps in pipeline. This is a safe guard to prevent infinite recursion")

        self.steps = self.steps + [step_def]

    def process(self):
        index = 0
        while index < len(self.steps):
            step = PipelineStep(self.steps[index], parent=self)

            step.process()

            index = index + 1


def process_args() -> int:
    """
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

    logger.info("Processing completed successfully")
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
