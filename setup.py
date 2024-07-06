from setuptools import find_packages, setup

setup(
    name="neural_network",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
    ],
    python_requires=">=3.10",
)
