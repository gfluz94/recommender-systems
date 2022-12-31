# recommender-systems
This repository aims at building recommender systems - from classical to most advanced approaches - with the full movielens dataset.

## Recommender Systems

![](./media/scheduler.gif)


This repository gathers all the code required for building recommender systems - from classical to most advanced approaches - with the full movielens dataset. The following techniques will be covered:

* Collaborative Filtering;
* Matrix Factorization;
* Content-Based Filtering.

First of all, we used **Python** and Tensorflow Deep Learning framework to write our scripts. Important packages within this environment are listed below:

* `tensorflow` so we could work on matrix factorization and deep neural networks;
* `numpy` and `scipy` so we could handle the huge dimensionality by taking advantage of sparse matrices;
* `pandas` so we could read raw data.

Finally, we used GitHub actions to build CI pipeline, with the help of a `Makefile`:

* __Installing packages__: we used `pip` and a `requirements.txt` file to list all required packages (`make install`);
* __Formatting__: `black` was used (`make format`);
* __Linting__: `pylint` was used (`make lint`);
* __Testing__: `pytest` was used (`make test`).

____

This project is structured as follows:

#### main.py

Main script where the the recommender system is designed, built and evaluated - according to which technique is chosen by the user.

```py
python main.py
```

#### tests

Folder where all unit tests are located.

#### recsys

Project folder structure, where all classes and methods are contained.

```sh
recsys
├── __init__.py
├── collaborative_filtering.py
├── content_based_filtering.py
├── matrix_factorization.py
└── utils
    ├── __init__.py
    ├── errors.py
    └── logging.py
    └── aws.py
```
