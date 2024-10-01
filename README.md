# Radio AstroNomy Cubes Handler (RANCH)

[![PyPI version](https://badge.fury.io/py/astro-ranch.svg)](https://badge.fury.io/py/astro-ranch)
[![Documentation Status](https://readthedocs.org/projects/astro-ranch/badge/?version=latest)](https://astro-ranch.readthedocs.io/en/latest/?badge=latest)
![test coverage](./coverage.svg)

An API to easily handle radio astronomy FITS files with Python.


## Installation

*(optional)* Create a virtual environment and activate it:

```shell
python -m venv .venv
source .venv/bin/activate
```

**Note 1:** to deactivate the virtual env :

```shell
deactivate
```

**Note 2:** To delete the virtual environment:

```shell
rm -r .venv
```

### From PyPI (recommanded)

To install `ranch`:

```shell
pip install astro-ranch
```

### From local package

To get the source code:

```shell
git clone git@github.com:einigl/ranch.git
```

To install `ranch`:

```shell
pip install -e .
```


## Get started

To get started, check out the Jupyter notebooks provided in the `examples` folder.


## Tests

To test, run:

```shell
pytest --cov && coverage-badge -o coverage.svg -f
```

## Documentation

```bash
cd docs
sphinx-apidoc -o . ../ranch
make html
```

Outputs are in `docs/_build/html`.


## Features

TODO
