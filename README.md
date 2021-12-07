# BiksUP - An Ablation Study of the SR-GNN

BiksUP is an ablation study conducted on the SR-GNN model in order to find a superior mode, with scalability in focus.

The work has been conducted by a group of 7th semester students at Aalborg University.

## Paper data and code

The following project is an ablation study based on the work by Wu et al in *'Session-based Recommendation with Graph Neural Networks'* found [here](https://ojs.aaai.org//index.php/AAAI/article/view/3804).

The ablation study has has been made on the codebase from [github](https://github.com/CRIPAC-DIG/SR-GNN).


<!-- 
This is the code for the AAAI 2019 Paper: [Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855). We have implemented our methods in both **Tensorflow** and **Pytorch**. -->

The code can be run on two datasets, who should be placed in the folder `datasets/`. The datasets are as following:

 * YooChoose: download [here](https://www.kaggle.com/chadgostopp/recsys-challenge-2015)
 * Dignetica: Download [here](https://competitions.codalab.org/competitions/11161)
<!-- 
Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:

- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup> or <https://competitions.codalab.org/competitions/11161> -->

There is a small dataset `sample` included in the folder `datasets/`, which can be used to test the correctness of the code.

For additional information from the authers of the original paper, a [blog](https://sxkdz.github.io/research/SR-GNN) was written to explain the paper.
<!-- 
We have also written a [blog](https://sxkdz.github.io/research/SR-GNN) explaining the paper. -->

## Usage

You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=sample`

```bash
usage: preprocess.py [-h] [--dataset DATASET]

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  dataset name: diginetica/yoochoose/sample
```

Then you can run the file `pytorch_code/main.py` or `tensorflow_code/main.py` to train the model.

For example: `cd pytorch_code; python main.py --dataset=sample`

You can add the suffix `--nonhybrid` to use the global preference of a session graph to recommend instead of the hybrid preference.

You can also change other parameters according to the usage:

```bash
usage main.py [-h] [--dataset DATASET [DATASET ...]] [--batchSize BATCHSIZE] [--hiddenSize HIDDENSIZE]
               [--epoch EPOCH] [--lr LR] [--lr_dc LR_DC] [--lr_dc_step LR_DC_STEP] [--l2 L2] [--step STEP]
               [--patience PATIENCE] [--nonhybrid] [--validation] [--valid_portion VALID_PORTION]
               [--keys KEYS [KEYS ...]] [--runall RUNALL] [--runlast RUNLAST] [--iterations ITERATIONS]
               [--big_o BIG_O] [--exp_graph EXP_GRAPH]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET [DATASET ...]
                        dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample
  --batchSize BATCHSIZE
                        input batch size
  --hiddenSize HIDDENSIZE
                        hidden state size
  --epoch EPOCH         the number of epochs to train for
  --lr LR               learning rate
  --lr_dc LR_DC         learning rate decay rate
  --lr_dc_step LR_DC_STEP
                        the number of steps after which the learning rate decay
  --l2 L2               l2 penalty
  --step STEP           gnn propogation steps
  --patience PATIENCE   the number of epoch to wait before early stop
  --nonhybrid           only use the global preference to predict
  --validation          validation
  --valid_portion VALID_PORTION
                        split the portion of training set as validation set
  --keys KEYS [KEYS ...]
                        List of boolean keys of what permutation to execute, '1' = True, '0'=False, '_' = True and
                        False. Example: ['1110_00']
  --runall RUNALL       Run all permutations of key combinations
  --runlast RUNLAST     Run the last executed variation of the --keys argument
  --iterations ITERATIONS
                        How many times the experiments should run
  --big_o BIG_O         Find the complexity with one key and one dataset
  --exp_graph EXP_GRAPH
                        Run the exponential graph experiment, Default='False'
```

## Requirements

- Python 3
- PyTorch 0.4.0 or Tensorflow 1.9.0

## Other Implementation for Reference
There are other implementation available for reference:
- Implementation based on PaddlePaddle by Baidu [[Link]](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gnn)
- Implementation based on PyTorch Geometric [[Link]](https://github.com/RuihongQiu/SR-GNN_PyTorch-Geometric)
- Another implementation based on Tensorflow [[Link]](https://github.com/jimanvlad/SR-GNN)
- Yet another implementation based on Tensorflow [[Link]](https://github.com/loserChen/TensorFlow-In-Practice/tree/master/SRGNN)

## Citation

Please cite our paper if you use the code:

```
@inproceedings{biksup:2021,
title = {{Improving the Scalability of Session-Recommendation Systems}},
author = {Roni Horo, Mads H. Kusk, Jeppe J. Holt, Bjarke S. Baltzersen, Milad Samim and Jamie B. Pedersen},
year = 2021,
location = {Aalborg, DK},
month = dec,
pages = {346--353},
url = {https://aaai.org/ojs/index.php/AAAI/article/view/3804},
editor = {Peter Dolog},
}
```

