import os

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = {
    "name": "ttast",
    "version": os.environ["BUILD_VERSION"],
    "description": "Text Transform Assistant",
    "long_description_content_type": "text/markdown",
    "long_description": README,
    "license": "MIT",
    "packages": find_packages(where="src", include=["ttast", "ttast.*"]),
    "author": "Jesse Reichman",
    "keywords": ["Text", "Transform", "Assistant"],
    "url": "https://github.com/archmachina/ttast",
    "download_url": "https://pypi.org/project/ttast/",
    "entry_points": {"console_scripts": ["ttast = ttast:main"]},
    "package_dir": {"": "src"},
    "install_requires": ["PyYAML>=6.0.0", "Jinja2>=3.1.0"],
}

if __name__ == "__main__":
    setup(**setup_args, include_package_data=True)
