#!/usr/bin/env python3
"""Setup script for the tripedal robot RL project."""

from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tripedal-robot-rl",
    version="0.1.0",
    author="Tripedal Robot Project",
    description="Reinforcement learning and control tools for a custom tripedal robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/tripedal-robot-rl",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 