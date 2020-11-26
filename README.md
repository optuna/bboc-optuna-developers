# Optuna Developers' Solution for Black-Box Optimization Challenge

First of all, we would like to thank BBO Challenge Organizers for this interesting competition. And congratulations to all winners.
Here is the code of Optuna Developers' solution for [NeurIPS 2020 Black-Box Optimization Challenge](https://bbochallenge.com/).

## Solution

### Results

Our solution achieved 96.939 for public and also 91.806 for private.
We ranked 9th place in public and 5th place in private.


### Building the final submission

The final code is placed to `./submissions/mksturbo`.
You can prepare the submission using the `prepare_upload.sh` script.

```
$ ./prepare_upload.sh ./submissions/mksturbo/
```


### Running local benchmarks using Bayesmark

You can run local benchamarks on publicly available problems using [Bayesmark](https://github.com/uber/bayesmark) library.
These problems using scikit-learn's classifiers/regressors and its built-in datasets.
See the [Bayesmark documentation](https://bayesmark.readthedocs.io/en/latest/index.html) for the details.

```
$ python3 -m venv venv  # Please use Python 3.6.10.
$ source venv/bin/activate
$ pip install -r environment.txt -r submissions/mksturbo/requirements.txt
$ ./run_local.sh ./submissions/mksturbo/ 3
```

<details>

<summary>Faster local benchmarking</summary>

You can also use [run_benchmark.py](./run_benchmark.py) to run local benchmarks.
This script is faster than `run_local.sh` because it runs benchmarks in parallel.

```
$ python run_benchmark.py --task large --repeat 3 --parallelism 16 --out ./output/ --optimizer ./submissions/mksturbo/
```

</details>


## References

TODO: Add link to the paper

## LICENSE

[Apache License 2.0](./LICENSE)

