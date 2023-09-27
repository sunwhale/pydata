import setuptools
from src.pydata import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydata",
    version=__version__,
    author="Jingyu Sun",
    author_email="sun.jingyu@outlook.com",
    description="Python Database",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sunwhale/pydata",
    project_urls={
        "Bug Tracker": "https://github.com/sunwhale/pydata/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        'numpy',
        'pandas',
    ],
)