from pathlib import Path

import setuptools

BASE_DIR = Path(__file__).parent

# Load packages from requirements.txt
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

VERSION = "0.0.1"
DESCRIPTION = "Convert PyTorch model to TFLite"

setuptools.setup(
    name="torch2tflite",
    version=VERSION,
    author="Rizky Nugroho",
    author_email="dummy@mail.com",
    description=DESCRIPTION,
    url="https://github.com/rpnugroho/torch2tflite",
    install_requires=required_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    # py_modules=['main'],
    entry_points={
        "console_scripts": [
            "torch2tflite = torch2tflite.main:cli",
        ],
    },
)
