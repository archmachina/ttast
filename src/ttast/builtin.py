
import yaml
import sys
import re
import glob
import os
import inspect
import copy
import hashlib
import textwrap
import base64

from .util import *
from . import types

class HandlerConfig(types.Handler):
    """
    """
    def parse(self):
        # Read the content from the file and use _process_config_content to do the work
        self.config_file = self.state.templater.extract_property(self.state.step_def, "file")
        validate(isinstance(self.config_file, str) or self.config_file is None, "Step 'config_file' must be a string or absent")
        validate(not isinstance(self.config_file, str) or self.config_file != "", "Step 'config_file' cannot be empty")

        # Extract the content var, which can be either a dict or yaml string
        self.config_content = self.state.templater.extract_property(self.state.step_def, "content")
        validate(isinstance(self.config_content, (str, dict)) or self.config_content is None, "Step 'config_content' must be a string, dict or absent")

        # Extract stdin bool, indicating whether to read config from stdin
        self.stdin = self.state.templater.extract_property(self.state.step_def, "stdin", default=False)
        validate(isinstance(self.stdin, (bool, str)), "Step 'stdin' must be a bool, bool like string or absent")
        self.stdin = parse_bool(self.stdin)

    def is_per_block():
        return False

    def run(self, block):
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
        config_vars = self.state.templater.extract_property(content, "vars", default={})
        validate(isinstance(config_vars, dict), "Config 'vars' is not a dictionary")

        for config_var_name in config_vars:
            self.state.pipeline.set_var(config_var_name, config_vars[config_var_name])

        # Extract pipeline steps from the config
        # Don't template the pipeline steps - These will be templated when they are executed
        config_pipeline = self.state.templater.extract_property(content, "pipeline", default=[], recursive_template=False)
        validate(isinstance(config_pipeline, list), "Config 'pipeline' is not a list")

        for step in config_pipeline:
            validate(isinstance(step, dict), "Pipeline entry is not a dictionary")

            self.state.pipeline.add_step(step)

        # Validate config has no other properties
        validate(len(content.keys()) == 0, f"Found unknown properties in configuration: {content.keys()}")

class HandlerImport(types.Handler):
    """
    """
    def parse(self):
        self.import_files = self.state.templater.extract_property(self.state.step_def, "files")
        validate(isinstance(self.import_files, list), "Step 'files' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.import_files), "Step 'files' must be a list of strings")

        self.recursive = self.state.templater.extract_property(self.state.step_def, "recursive")
        validate(isinstance(self.recursive, (bool, str)), "Step 'recursive' must be a bool or bool like string")
        self.recursive = parse_bool(self.recursive)

    def is_per_block():
        return False

    def run(self, block):
        filenames = set()
        for import_file in self.import_files:
            logger.debug(f"import: processing file glob: {import_file}")
            matches = glob.glob(import_file, recursive=self.recursive)
            for match in matches:
                filenames.add(match)

        # Ensure consistency for load order
        filenames = list(filenames)
        filenames.sort()

        new_blocks = []
        for filename in filenames:
            logger.debug(f"import: reading file {filename}")
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()
                new_block = types.TextBlock(content)
                new_block.meta["import_filename"] = filename
                self.state.pipeline.add_block(new_block)
                new_blocks.append(new_block)

        return new_blocks

class HandlerMeta(types.Handler):
    """
    """
    def parse(self):
        self.vars = self.state.templater.extract_property(self.state.step_def, "vars")
        validate(isinstance(self.vars, dict), "Step 'vars' must be a dictionary of strings")
        validate(all(isinstance(x, str) for x in self.vars), "Step 'vars' must be a dictionary of strings")

    def is_per_block():
        return True

    def run(self, block):
        for key in self.vars:
            block.meta[key] = self.vars[key]

class HandlerReplace(types.Handler):
    """
    """
    def parse(self):
        self.replace = self.state.templater.extract_property(self.state.step_def, "replace", default={})
        validate(isinstance(self.replace, list), "Step 'replace' must be a list")
        validate(all(isinstance(x, dict) for x in self.replace), "Step 'replace' items must be dictionaries")
        for item in self.replace:
            validate('key' in item and isinstance(item['key'], str), "Step 'replace' items must contain a string 'key' property")
            validate('value' in item and isinstance(item['value'], str), "Step 'replace' items must contain a string 'value' property")

        self.regex = self.state.templater.extract_property(self.state.step_def, "regex", default=False)
        validate(isinstance(self.regex, (bool, str)), "Step 'regex' must be a bool, bool like string or absent")
        self.regex = parse_bool(self.regex)

    def is_per_block():
        return True

    def run(self, block):
        for replace_item in self.replace:
            # Copy the dictionary as we'll change it when removing values
            replace_item = replace_item.copy()

            replace_key = replace_item['key']
            replace_value = replace_item['value']

            replace_regex = self.state.templater.extract_property(replace_item, "regex", default=False)
            validate(isinstance(replace_regex, (bool, str)), "Replace item 'regex' must be a bool, bool like string or absent")
            replace_regex = parse_bool(replace_regex)

            logger.debug(f"replace: replacing regex({self.regex or replace_regex}): {replace_key} -> {replace_value}")

            if self.regex or replace_regex:
                block.text = re.sub(replace_key, replace_value, block.text)
            else:
                block.text = block.text.replace(replace_key, replace_value)

class HandlerSplitYaml(types.Handler):
    """
    """
    def parse(self):
        self.strip = self.state.templater.extract_property(self.state.step_def, "strip", default=False)
        validate(isinstance(self.strip, (bool, str)), "Step 'strip' must be a bool or str value")
        self.strip = parse_bool(self.strip)

    def is_per_block():
        return True

    def run(self, block):
        lines = block.text.splitlines()
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

        # If we have a single document and it's the same as the
        # original block, just exit
        if len(documents) == 1 and documents[0] == block.text:
            return

        # Add all documents to the pipeline text block list
        new_blocks = [types.TextBlock(item) for item in documents]
        for new_block in new_blocks:
            new_block.meta = copy.deepcopy(block.meta)
            new_block.tags = copy.deepcopy(block.tags)

            self.state.pipeline.add_block(new_block)

        # Remove the original source block from the list
        self.state.pipeline.remove_block(block)

        logger.debug(f"split_yaml: output 1 document -> {len(documents)} documents")

        return new_blocks

class HandlerStdin(types.Handler):
    """
    """
    def parse(self):
        self.split = self.state.templater.extract_property(self.state.step_def, "split")
        validate(isinstance(self.split, str) or self.split is None, "Step 'split' must be a string")

        self.strip = self.state.templater.extract_property(self.state.step_def, "strip", default=False)
        validate(isinstance(self.strip, (bool, str)), "Step 'strip' must be a bool or str value")
        self.strip = parse_bool(self.strip)

    def is_per_block():
        return False

    def run(self, block):
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
        new_blocks = [types.TextBlock(item) for item in stdin_items]
        for item in new_blocks:
            self.state.pipeline.add_block(item)

        return new_blocks

class HandlerStdout(types.Handler):
    """
    """
    def parse(self):
        self.prefix = self.state.templater.extract_property(self.state.step_def, "prefix")
        validate(isinstance(self.prefix, str) or self.prefix is None, "Step 'prefix' must be a string")

        self.suffix = self.state.templater.extract_property(self.state.step_def, "suffix")
        validate(isinstance(self.suffix, str) or self.suffix is None, "Step 'suffix' must be a string")

    def is_per_block():
        return True

    def run(self, block):
        if self.prefix is not None:
            print(self.prefix)

        print(block.text)

        if self.suffix is not None:
            print(self.suffix)

class HandlerTemplate(types.Handler):
    """
    """
    def parse(self):
        self.vars = self.state.templater.extract_property(self.state.step_def, "vars")
        validate(isinstance(self.vars, dict) or self.vars is None, "Step 'vars' must be a dictionary or absent")

    def is_per_block():
        return True

    def run(self, block):
        template_vars = self.state.vars.copy()

        if self.vars is not None:
            for key in self.vars:
                template_vars[key] = self.vars[key]

        block.text = self.state.templater.template_if_string(block.text)
        if not isinstance(block.text, str):
            raise PipelineRunException("Could not template source text")

def _block_sum(block):
    validate(isinstance(block, types.TextBlock), "Invalid text block passed to _block_sum")

    md5 = hashlib.md5()
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()

    encode = block.text.encode("utf-8")
    md5.update(encode)
    sha1.update(encode)
    sha256.update(encode)

    block.meta["md5sum"] = md5.hexdigest()
    block.meta["sha1sum"] = sha1.hexdigest()
    block.meta["sha256sum"] = sha256.hexdigest()

    short = 0
    short_wrap = textwrap.wrap(sha256.hexdigest(), 8)
    for item in short_wrap:
        short = short ^ int(item, 16)

    block.meta["shortsum"] = format(short, 'x')

class HandlerSum(types.Handler):
    """
    """
    def parse(self):
        pass

    def is_per_block():
        return True

    def run(self, block):
        _block_sum(block)

        logger.debug(f"sum: document short sum: {block.meta['shortsum']}")

class SupportHandlerSum(types.SupportHandler):
    """
    """
    def parse(self):
        pass

    def pre(self, block):
        pass

    def post(self, block):
        if block is None:
            return

        _block_sum(block)

        logger.debug(f"sum: document short sum: {block.meta['shortsum']}")

class SupportHandlerTags(types.SupportHandler):
    def parse(self):
        # Apply tags
        self.apply_tags = self.state.templater.extract_property(self.state.step_def, "apply_tags", default=[])
        validate(isinstance(self.apply_tags, list), "Step 'apply_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.apply_tags), "Step 'apply_tags' must be a list of strings")

    def pre(self, block):
        pass

    def post(self, block):
        if block is not None:
            for tag in self.apply_tags:
                block.tags.add(tag)

class SupportHandlerWhen(types.SupportHandler):
    def parse(self):
        # When condition
        self.when = self.state.templater.extract_property(self.state.step_def, "when", default=[])
        validate(isinstance(self.when, (list, str)), "Step 'when' must be a string or list of strings")
        if isinstance(self.when, str):
            self.when = [self.when]
        validate(all(isinstance(x, str) for x in self.when), "Step 'when' must be a string or list of strings")

    def pre(self, block):
        if len(self.when) > 0:
            for condition in self.when:
                result = self.state.templater.template_if_string("{{" + condition + "}}")
                if not parse_bool(result):
                    return []

    def post(self, block):
        pass

class SupportHandlerMatchTags(types.SupportHandler):
    def parse(self):
        # Extract match any tags
        self.match_any_tags = self.state.templater.extract_property(self.state.step_def, "match_any_tags", default=[])
        validate(isinstance(self.match_any_tags, list), "Step 'match_any_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.match_any_tags), "Step 'match_any_tags' must be a list of strings")
        self.match_any_tags = set(self.match_any_tags)

        # Extract match all tags
        match_all_tags = self.state.templater.extract_property(self.state.step_def, "match_all_tags", default=[])
        validate(isinstance(match_all_tags, list), "Step 'match_all_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in match_all_tags), "Step 'match_all_tags' must be a list of strings")
        self.match_all_tags = set(match_all_tags)

        # Extract exclude tags
        self.exclude_tags = self.state.templater.extract_property(self.state.step_def, "exclude_tags", default=[])
        validate(isinstance(self.exclude_tags, list), "Step 'exclude_tags' must be a list of strings")
        validate(all(isinstance(x, str) for x in self.exclude_tags), "Step 'exclude_tags' must be a list of strings")
        self.exclude_tags = set(self.exclude_tags)

    def pre(self, block):
        if len(self.match_any_tags) > 0:
            # If there are any 'match_any_tags', then at least one of them has to match with the document
            if len(self.match_any_tags.intersection(block.tags)) == 0:
                return []

        if len(self.match_all_tags) > 0:
            # If there are any 'match_all_tags', then all of those tags must match the document
            for tag in self.match_all_tags:
                if tag not in block.tags:
                    return []

        if len(self.exclude_tags) > 0:
            # If there are any exclude tags and any are present in the block, it isn't a match
            for tag in self.exclude_tags:
                if tag in block.tags:
                    return []

    def post(self, block):
        pass

def FilterHash(value, name="sha1"):

    # Get a reference to the object to use for hashing
    classref = getattr(hashlib, name)
    instance = classref()

    instance.update(str(value).encode("utf-8"))

    return instance.hexdigest()

def FilterBase64Encode(value, encoding="utf-8"):
    return base64.b64encode(str(value).encode(encoding))

def FilterBase64Decode(value, encoding="utf-8"):
    return base64.b64decode(value).decode(encoding)
