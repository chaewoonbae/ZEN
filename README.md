# Official code implementation of ZEN

This repository is the official implementation of Parameter-Free Hypergraph Neural Network for Few-Shot Node Classification. 


## Dataset description

We provide 10 hypergraph benchmark datasets for evaluating accuracy, and one additional dataset for interpretability analysis. Their overall statistics are as follows:

### Benchmark datasets

| Name       | Command name        | # of nodes | # of edges | # of classes | # of features |
|------------|---------------------|------------|------------|--------------|----------------|
| Cora       | `cora`              | 2,708      | 1,579      | 7            | 1,433          |
| Citeseer   | `citeseer`          | 3,312      | 1,079      | 6            | 3,703          |
| Pubmed     | `pubmed`            | 19,717     | 7,963      | 3            | 500            |
| Cora-CA    | `coauthor_cora`     | 2,708      | 1,072      | 7            | 1,433          |
| 20News     | `20newsW100`        | 16,242     | 100        | 4            | 100            |
| MN40       | `ModelNet40`        | 12,311     | 12,311     | 40           | 100            |
| Congress   | `congress-bills`    | 1,718      | 83,105     | 2            | 100            |
| Walmart    | `walmart-trips`     | 88,860     | 69,906     | 11           | 100            |
| Senate     | `senate-committees` | 282        | 315        | 2            | 100            |
| House      | `house-committees`  | 1,290      | 340        | 2            | 100            |

### Interpretability dataset

| Name | Command name | # of nodes | # of edges | # of classes | # of features |
|------|--------------|------------|------------|--------------|----------------|
| Zoo  | `zoo`        | 101        | 43         | 7            | 16             |



## Evaluation

To evaluate ZEN on Cora, run:

```eval
python3 main.py
```

### Additional arguments

| Argument  | Description                                                                                                         |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| `-data`   | Name of the dataset to use. Options: `cora`, `citeseer`, `pubmed`, etc. Default: `cora`.                            |
| `-n`      | Grid resolution for 2-simplex search during hyperparameter tuning. Total grid points: $(n+2)(n+1)/2$. Default: `9`. |
| `-k`      | Number of labeled nodes per class used for training and validation in few-shot classification. Default: `5`.        |
| `-run`    | Number of independent random splits (train/val/test) for evaluation. Default: `10`.                                 |
| `-device` | Computation device. Options: `cuda:0`, `cuda:1`, `cpu`, etc. Default: `cuda:0`.                                        |
