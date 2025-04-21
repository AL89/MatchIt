from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(name="matchit",
      version="0.2.6",
      description="A package for Padel match making and padel player rankings",
      long_description=long_description,
      long_description_content_type="text/markdown",
      author="Andreas Løvgaard",
      packages=find_packages(),
      install_requires=[
          "pandas",
          "pydantic",
          "jupyter"
      ],
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.13",
        ],
        python_requires='>=3.13',
    )