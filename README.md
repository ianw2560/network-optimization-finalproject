# Analyzing a Congressional Twitter Interaction Network Using MCL, Spectral Clustering, and PageRank

This repository is the source code for the paper of the same name. It was created for the COP5337 class.

Here is a brief overview of the structure of this repository:

- `results/`: This folder contains outputs from running the programs
- `congress_network_data.json`: This is the dataset from the SNAP platform in JSON format
- `congress.edgelist`: This is the dataset from the SNAP platform as a list of edges with weights.
- `clustering.py`: Source code for running MCL and spectral clustering
- `pagerank.py`:  Source code for running PageRank

## Usage

To generate the results for yourself, see the following sections:

To run the MCL and spectral clustering algorithms, run

```
python clustering.py > results/clustering.out
```

To run the MCL and spectral clustering algorithms, run

```
python pagerank.py > results/pagerank.out
```

This code was tested on Python 3.11.5 on Manjaro Linux.

## Results

The results mentioned in the paper have been committed to this repository and are found in the `results/` directory.

## References

The dataset for this code can be found on the [SNAP website](https://snap.stanford.edu/data/congress-twitter.html).
