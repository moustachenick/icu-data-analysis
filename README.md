# ICU Data Analysis

## About

This is a project that reads ICU data files and generates a final `.csv` file with the data.
It then creates a laggard dataset with the data from the previous day and calculates the difference between the two
days.
Then, it performs a regression analysis on the data and plots the results.

## How to use

1. Clone the repository
2. Make sure you have Python 3 installed (see below for the virtual env recommended option)
3. Install the requirements by running `pip install -r requirements.txt`
4. Run the script by running `python src/main.py`

## How to install Python

Based on the Operating System you are running on, you can follow the instructions on
the [official Python website](https://www.python.org/downloads/).

An easy solution is to use [Python Virtual Environments](https://docs.python.org/3/library/venv.html). This way you can
have multiple Python versions installed on your machine and you can create isolated environments for each project.
In order to do so, you can run the following commands:

```bash
python -m venv venv
```

This will create a virtual environment in the `venv` directory. You can activate it by running:

### For Linux

```bash
source venv/bin/activate
```

### For Windows

```bash
# Windows
.\venv\Scripts\activate.bat # or ./venv/Scripts/activate.bat (depending on your terminal configuration)
```

Another intuitive solution is to use [Anaconda](https://www.anaconda.com/products/distribution) which is a Python
distribution that comes with a lot of useful packages and tools.

## How to install the project dependencies

After you have activated the Python environment, you can install the dependencies by running:

```bash
# Windows
venv/Scripts/pip install -r requirements.txt

# Linux
venv/bin/pip install -r requirements.txt
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

## How to run the main script

After you have installed the dependencies and have the data files in the correct directory, you can run the script by
running:

```bash
# Windows
venv/Scripts/python src/main.py

# Linux
venv/bin/python src/main.py
```

## Testing

The project includes some basic testing using the `pytest` package and the `unittest` module.

You can run the tests by running:

```bash
# Windows
venv/Scripts/pytest tests/

# Linux
venv/bin/pytest tests/
```
