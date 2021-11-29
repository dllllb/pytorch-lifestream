import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="embeddings-validation",
    version="0.0.3",
    author="Ivan Kireev",
    author_email="ivkireev@yandex.ru",
    description="Estimate your feature vector quality on downstream task",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sberbank-ai-lab/embeddings-valid",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
