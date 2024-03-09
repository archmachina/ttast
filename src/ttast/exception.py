"""
Contains common exceptions used by ttast
"""

class PipelineRunException(Exception):
    """
    Exception representing a runtime error while processing the pipeline
    """
    pass

class PipelineConfigException(Exception):
    """
    Exception representing a config error while processing the pipeline
    """
    pass

class ValidationException(Exception):
    """
    Validation of some condition failed
    """
