import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="takco-bennokr",
    version="0.0.1",
    author="Benno Kruit",
    author_email="bennokr@gmail.com",
    description="Extracti knowledge from tables",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/karma-research/takco",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    entry_points={'console_scripts': ['takco = takco.__main__:main']},
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)