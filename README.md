## Falconn++ - A Locality-sensitive Filtering Approach for ANNS with Inner Product

Falconn++ is a locality-sensitive filtering (LSF) approach, built on top of cross-polytope LSH (FalconnLib) to answer approximate nearest neighbor search with inner product. 
The filtering mechanism of Falconn++  bases on the asymptotic property of the concomitant of extreme order statistics where the projections of $x$ onto closest or furthest vector to $q$ preserves the dot product $x^T q$.
Similar to FalconnLib, Falconn++ utilizes many random projection vectors and uses the Fast Hadamard Transform to speed up the hashing evaluation.
Apart from many hashing-based approaches, Falconn++ has multi-probes on both indexing and querying to improve the quality of candidates.
Falconn++ also supports multi-threading for both indexing and querying by adding only ```#pragma omp parallel for```.

We call EigenLib that supports SIMD dot product computation.
We have not engineered Falconn++ much with other techniques, e.g. prefetching.
## Prerequisites

* A compiler with C++17 support
* CMake >= 3.27 (test on Ubuntu 20.04 and Python3)
* Ninja >= 1.10 
* Eigen >= 3.3
* Boost >= 1.71
* Pybinding11 (https://pypi.org/project/pybind11/) 

## Installation

Just clone this repository and run

```bash
python3 setup.py install
```

or 

```bash
mkdir build && cd build && cmake .. && make
```


## Test call

Data and query must be d x n matrices.

```
import FalconnPP
index = FalconnPP.FalconnPP(n_points, n_features)
index.setIndexParam(n_tables, n_proj, bucketLimit, alpha, iProbes, n_threads)
index.build(dataset_t)  # add vectors to the index, must transpose to D x N

index.set_qProbes(qProbes) # set multi-probes for querying
fal_answers = index.query(queries_t, k)
```

See test/ for Python example and src/main.cpp for C++ example.

## Authors

It is mainly developed by Ninh Pham. It grew out of a master research project of Tao Liu.

