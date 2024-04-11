# ICU Data Analysis

## About

This is a simple parser for ICU data. It reads the data from a given ICU data files directory and creates a final `.csv`
with the data.

## How to use

1. Clone the repository
2. Make sure you have Python 3 installed
3. Install the requirements by running `pip install -r requirements.txt`
4. Run the script by running `python src/main.py`
5. The final `.csv` will be created in `output/icu-data.csv`

## How to install Python

Based on the Operating System you are running on, you can follow the instructions on
the [official Python website](https://www.python.org/downloads/).

An easy solution is to use [Python Virtual Environments](https://docs.python.org/3/library/venv.html). This way you can
have multiple Python versions installed on your machine and you can create isolated environments for each project.
In order to do so, you can run the following commands:

```bash
python3 -m venv venv
```

This will create a virtual environment in the `venv` directory. You can activate it by running:

### For Linux

```bash
source venv/bin/activate
```

### For Windows

```bash
.\venv\Scripts\activate.bat
```

Another intuitive solution is to use [Anaconda](https://www.anaconda.com/products/distribution) which is a Python
distribution that comes with a lot of useful packages and tools.

## How to install the project dependencies

After you have activated the Python environment, you can install the dependencies by running:

```bash
pip install -r requirements.txt
```

This command will install all the dependencies listed in the `requirements.txt` file.

## About the data files directory

When running the script, you need to provide the path to the directory containing the ICU data files, in `.txt` format.
The directory should have the following structure:

``` text
data/
    CPP.txt
    Glucose.txt
    Haemoglobin.txt
    ...
```

**Notice 1**: the files should be named exactly as above.

**Notice 2**: the files should be in `.txt` format.

**Notice 3**: the files inside the `data` directory are not tracked by Git, so you need to add the files manually.

The algorithm will read all the files in the directory and parse the data into a final `.csv` file.
The file names are hardcoded in the script, so make sure the files are named as above.
If a file is missing, the program will raise a warning, but continue the execution.

## How to run the script

After you have installed the dependencies and have the data files in the correct directory, you can run the script by
running:

```bash
python src/main.py --dir=<path_to_icu_data_files> --dir-relative=True --log=DEBUG
```

Where:

If no `--dir` argument is provided, the default directory is the `data/` directory inside this repository.

If no `--dir-relative` argument is provided, the default value is `True`. This argument denotes if the path to the
directory is relative to the repository root or if it is an absolute path.

If no `--log` argument is provided, the default log level is `DEBUG`.

So you can also run the script with the default parameters, by running:

```bash
python src/main.py
```