# ICU Data Analysis

## About

This is a project that reads ICU data files, generates train/test datasets, and runs a Regression or Classification pipeline.

## How to use

1. Clone the repository
2. Make sure you have Python 3 installed (see below for the virtual env recommended option)
3. Install the requirements by running `pip install -r requirements.txt`
4. Run the script by running `python src/main.py`

## How to install Python

### Using Miniconda/Anaconda (Recommended)

You can install Python using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
After installing Miniconda/Anaconda, you can create a new environment with the desired Python version by running:

```shell
conda env create -f environment.yml

conda activate icu-data-analysis
```

### Python Native Installation

Based on the Operating System you are running on, you can follow the instructions on
the [official Python website](https://www.python.org/downloads/).

### Using Virtual Environment

An easy solution is to use [Python Virtual Environments](https://docs.python.org/3/library/venv.html). This way you can
have multiple Python versions installed on your machine and you can create isolated environments for each project.
In order to do so, you can run the following commands:

```shell
py -3.12 -m venv venv
```

This will create a virtual environment in the `venv` directory. You can activate it by running:

#### For Linux

```shell
source venv/bin/activate
```

#### For Windows

```shell
# Windows
.\venv\Scripts\activate.bat # or ./venv/Scripts/activate.bat (depending on your terminal configuration)
```

After you have activated the Python environment, you can install the dependencies by running:

```shell
# Windows
venv/Scripts/pip install -r requirements.txt

# Linux
venv/bin/pip install -r requirements.txt
```

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

## Running the main script

After you have installed the dependencies and have the data files in the correct directory, you can run the script by
running:

### Miniconda/Anaconda Option

```shell
conda activate icu-data-analysis

python src/main.py

# or, more verbose:
conda run -n icu-data-analysis python src/main.py
```

### Virtual Environment Option

```shell
# Windows
venv/Scripts/python src/main.py

# clean generated files & run:
./clean_generated_files.sh && venv/Scripts/python.exe src/main.py

# Linux
venv/bin/python src/main.py
```

### Native Python Option

```shell
python src/main.py
```

## Configuration

The pipeline is fully configuration-driven — there are **no interactive prompts**.
All run decisions live in a `config.toml` file at the project root. This file is **not tracked by
Git**; a tracked template, `config.example.toml`, holds the documented defaults. On the
first run, if `config.toml` does not exist, the program creates it by copying
`config.example.toml`. Edit `config.toml` to change behaviour, then re-run.

Because the run is driven entirely by config (no typed input), every run is reproducible —
the full console output is also mirrored to `output/{mode}_{timestamp}.txt`, so you can diff
outputs across changes.

```toml
[run]
mode  = "regression"   # "regression" or "classification"
hours = 5              # lag lookback window; drives icp_lag_1 ... icp_lag_N

[preprocessing]
add_pathology_one_hot     = false  # add one-hot encoded pathology columns
drop_high_missing_columns = true   # drop respiration_rate + rows with >1 missing value
impute_missing_values     = true   # KNN-impute (n_neighbors=1) rows with exactly 1 missing value
drop_lagged_null_rows     = true   # drop rows with null values in lagged columns

[parsing]
filter_by_pathology      = ""      # pathology code to keep; "" = include all pathologies
apply_instance_filtering = true    # restrict dataset to patients in pathologies_filtered.csv

[classification]
run_cross_validation = false       # run the 10-fold cross-validation after the main pipeline
```

The CLI flags `--mode` and `--hours` override the corresponding `[run]` values, e.g.:

```shell
python src/main.py --mode classification --hours 8
```

After the preprocessing steps, the lagged datasets will be created and saved to the `data/` directory.

For Regression, it would be:

```text
data/train_data.csv

data/test_data.csv
```

For Classification, it would be:

```text
data/train_data_classification.csv

data/test_data_classification.csv
```

**Notice**: If you want the pre-precessing steps to run again, you should delete the `data/train_data.csv`,
`data/test_data.csv`, `data/train_data_classification.csv`, `data/test_data_classification.csv`, `data/final_data.csv`,
and `data/cleaned_df_lagged` files. The script will then re-run the preprocessing steps using your `config.toml`.

You can also do that by running the provided `clean_generated_files.sh` script, which will delete all the aforementioned
generated
files.

## Testing

The project includes some basic testing using the `pytest` package and the `unittest` module.

You can run the tests by running:

```shell
# Windows
venv/Scripts/pytest tests/

# Linux
venv/bin/pytest tests/
```
