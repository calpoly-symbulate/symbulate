from pathlib import Path

from setuptools import setup, find_packages

# Renders as the PyPI project description; without this a PyPI listing
# shows a blank page below the summary line.
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="symbulate",
    version="0.5.5",
    description="A symbolic algebra for specifying simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dlsun/symbulate",
    author="Dennis Sun",
    author_email="dsun09@calpoly.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="probability simulation",
    packages=find_packages(),
    # The package style sheet is loaded at import time (see
    # symbulate/plot.py), so it must ship with the package.
    package_data={
        "symbulate": ["symbulate.mplstyle"],
    },
    # A floor, not a pin: old enough to be permissive, recent enough to
    # rule out versions that predate APIs this package already relies on
    # (e.g. np.random.default_rng, added in numpy 1.17).
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.4",
        "matplotlib>=3.2",
    ],
)
