import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

DESCRIPTION = ("Mlflow plugin to use MongoDB as backend for MLflow tracking service")
VERSION = "0.1"
try:
    LONG_DESCRIPTION = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except Exception:
    LONG_DESCRIPTION = ""


def _read_reqs(relpath):
    fullpath = os.path.join(os.path.dirname(__file__), relpath)
    with open(fullpath) as f:
        return [s.strip() for s in f.readlines()
                if (s.strip() and not s.startswith("#"))]


REQUIREMENTS = _read_reqs("requirements.txt")


class ListDependencies(setuptools.Command):
    # `python setup.py <command name>` prints out "running <command name>" by default.
    # This logging message must be hidden by specifying `--quiet` (or `-q`) when piping the output
    # of this command to `pip install`.
    description = "List mlflow dependencies"
    user_options = [
        ("skinny", None, "List mlflow-skinny dependencies"),
    ]

    def initialize_options(self):
        self.skinny = False

    def finalize_options(self):
        pass

    def run(self):
        dependencies = REQUIREMENTS
        print("\n".join(dependencies))


class MinPythonVersion(setuptools.Command):
    description = "Print out the minimum supported Python version"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(MINIMUM_SUPPORTED_PYTHON_VERSION)


MINIMUM_SUPPORTED_PYTHON_VERSION = "3.8"

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    "Environment :: Console",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Topic :: Software Development :: Libraries"
]

setuptools.setup(
    name="mlflow-mongostore",
    version=VERSION,
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    install_requires=REQUIREMENTS,
    entry_points={
        # Define a Tracking Store plugin for tracking URIs with scheme 'file-plugin'
        "mlflow.tracking_store": ["mongodb=mlflow_mongostore.mongo_store:MongoStore",
                                  "mongodb+srv=mlflow_mongostore.mongo_store:MongoStore"],
    },
    cmdclass={
        "dependencies": ListDependencies,
        "min_python_version": MinPythonVersion,
    },
    tests_require=["pytest"],
    author="SatyamJay",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    classifiers=CLASSIFIERS,
    keywords="mlflow",
    url="https://github.com/satyamjay-iitd/mlflow-mongostore",
    python_requires=">=3.8",
    maintainer_email="satyamjay030@gmail.com",
)
