
import jinja2
import logging

from .exception import *

logger = logging.getLogger(__name__)

def validate(val, message):
    if not val:
        raise ValidationException(message)

def template_if_string(val, mapping):
    if val is not None and isinstance(val, str):
        try:
            environment = jinja2.Environment()

            template = environment.from_string(val)
            return template.render(mapping)
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

def pop_property(spec, key, *, template_map=None, default=None, required=False):
    validate(isinstance(spec, dict), f"Invalid spec passed to pop_property. Must be dict")
    validate(isinstance(template_map, dict) or template_map is None, "Invalid type passed as template_map")

    if key not in spec:
        # Raise exception is the key isn't present, but required
        if required:
            raise KeyError(f'Missing key "{key}" in spec or value is null')

        # If the key is not present, return the default
        return default

    # Retrieve value
    val = spec.pop(key)

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

def merge_meta_tags(vars, *, tags=None, meta=None):
    validate(isinstance(vars, dict), "Vars provided to merge_meta_tags is not a dictionary")
    validate(isinstance(tags, (set, list)) or tags is None, "Tags provided to merge_meta_tags is not a list, set or absent")
    validate(isinstance(meta, dict) or tags is None, "Tags provided to merge_meta_tags is not a list or absent")

    new_vars = vars.copy()

    new_tags = ",".join(set(tags))
    new_vars["ttast_tags"] = new_tags

    # Create vars for all of the metadata
    for meta_key in meta:
        new_vars[f"ttast_meta_{meta_key}"] = meta[meta_key]

    return new_vars
