import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as reqs:
    requirements = reqs.read().splitlines()

setuptools.setup(
    name="bayesian-inference",
    version="1.0.2",
    author="Berat Cankar",
    author_email="berat.cankar@gmail.com",
    description="Bayesian Inference library over network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yakuza8/bayesian-inference",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.3',
)
