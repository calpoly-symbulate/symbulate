from setuptools import setup, find_packages

setup(
    name="symbulate",
    version="0.5.5",
    description="A symbolic algebra for specifying simulations.",
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
    install_requires=["numpy", "scipy", "matplotlib"],
)
