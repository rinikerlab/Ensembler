"""
Ensembler
Code to sample ensembles of simple (toy) models with various algorithms. 
"""
import sys
import versioneer

short_description = __doc__.split("\n")

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])

import io
import os
import sys

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = "ensembler-rinikerlab"
DESCRIPTION = "Ensembler is a tool for fast and efficient development of simulation methods or for teaching purposes."
URL = "https://github.com/rinikerlab/Ensembler"
DOWNLOAD_URL = "https://github.com/rinikerlab/Ensembler/archive/1.0.tar.gz"
EMAIL = "bschroed@ethz.ch"
AUTHOR = "Benjamin Ries; David Friedrich Hahn; Stephanie Linker"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None  # "1.0.0"
KEYWORDS = " ".join(
    [
        "teaching",
        "method_development",
        "statistical_mechanics",
        "statistical_thermodynamics",
        "physics",
        "chemistry",
        "free_energy_calculations",
        "free energy",
        "enhanced sampling",
        "RE-EDS",
        "EDS",
        "Conveyor_Belt_TI",
        "numerical_integration",
        "science",
    ]
)

# What packages are required for this module to be executed? - for minimal package version, please check the requirement files
REQUIRED = [
    "typing",  # Code: used for type declarations
    "pytest",  # Code: used for testing
    "pandas",  # Code: core functionality
    "numpy",  # Code: core functionality
    "sympy",  # Code: core functionality
    "scipy",  # Code: core functionality
    "matplotlib",  # Visualization
    "jupyter",  # Tutorial/Examples: basics
    "ipywidgets",  # Tutorial/Examples: Interactive widgets
    "tqdm",  # Tutorial/Examples: nice progressbar
    "ffmpeg",  # Visualizations: for animations in general
]

# What packages are optional?
EXTRAS = {
    "docs": [
        "sphinx",  # Documentation: autodocu tool
        "sphinx_rtd_theme",  # Documentation: style
        "nbsphinx",  # Documentation: for inclusion of jupyter notebooks
        "m2r",  # Documentation: converts markdown to rst
    ]
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README.md and use it as the long-description.
# Note: this will only work if 'README.md.md' is present in your MANIFEST.in file!
try:
    # with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    with io.open(os.path.join(here, "docs/sphinx_project/introduction.rst"), encoding="utf-8") as f:

        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, "ensembler", "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION
VERSION = about["__version__"]


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            import shutil

            shutil.rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(VERSION))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    # Self-descriptive entries which should always be present
    name=NAME,
    author=AUTHOR,
    author_email=EMAIL,
    description=short_description[0],
    long_description=long_description,
    # long_description_content_type="text/markdown",
    version=VERSION,  # versioneer.get_version(),
    # cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    keywords=KEYWORDS,
    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    # Allows `setup.py test` to work correctly with pytest
    install_requires=REQUIRED,
    setup_requires=REQUIRED,
    extras_require=EXTRAS,
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    url=URL,  # Website
    download_url=DOWNLOAD_URL,  # current version
    # install_requires=[REQUIRED],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    platforms=["Linux", "Mac OS-X", "Unix", "Windows"],  # Valid platforms your code works on, adjust to your flavor
    python_requires=">=3.5",  # Python version restrictions
    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,
)
