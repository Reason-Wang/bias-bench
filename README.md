# An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models
> Nicholas Meade, Elinor Poole-Dayan, Siva Reddy

[![arxiv](https://img.shields.io/badge/arXiv-2110.00768-b31b1b.svg)](https://arxiv.org/abs/2110.08527)

## 使用

+ 对crows-pairs，一个示例命令如下：

```bash
python experiments/crows.py \
  --model GPT2LMHeadModel \
  --model_name_or_path gpt2 \
  --bias_type gender
```

+ 期中，model为模型结构，model_name_or_path为模型名称或目录，bias_type为偏见种类。具体选项可再experiments/crows.py中查看
+ 对stereoset，一个示例命令如下：

```bash
python experiments/stereoset.py \
  --model GPT2LMHeadModel \
  --model_name_or_path gpt2
```

+ 然后用下面的命令计算总的结果：

```bash
python experiments/stereoset_evaluation.py \
  --predictions_dir results/stereoset/
```

+ 这些模型结构目前是固定的，我还没有探索怎么加不同模型结构，但是相同结构的不同模型应该可以直接测

This repository contains the official source code for [An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models](https://arxiv.org/abs/2110.08527) presented at ACL 2022.

## Bias Bench Leaderboard
For tracking progress on the intrinsic bias benchmarks evaluated in this work, we created [**Bias Bench**](https://mcgill-nlp.github.io/bias-bench/). We plan to update Bias Bench in the future with additional bias benchmarks. To make a submission to Bias Bench, please contact nicholas.meade@mila.quebec.

## Install
```bash
git clone https://github.com/mcgill-nlp/bias-bench.git
cd bias-bench 
python -m pip install -e .
```

## Required Datasets
Below, a list of the external datasets required by this repository is provided:

Dataset | Download Link | Notes | Download Directory
--------|---------------|-------|-------------------
Wikipedia-2.5 | [Download](https://drive.google.com/file/d/1JSlm8MYDbNjpMPnKbb91T-xZnlWAZmZl/view?usp=sharing) | English Wikipedia dump used for SentenceDebias and INLP. | `data/text`
Wikipedia-10 | [Download](https://drive.google.com/file/d/1boQTn44RnHdxWeUKQAlRgQ7xrlQ_Glwo/view?usp=sharing) | English Wikipedia dump used for CDA and Dropout. | `data/text`

Each dataset should be downloaded to the specified path, relative to the root directory of the project.

## Experiments
We provide scripts for running all of the experiments presented in the paper.
Generally, each script takes a `--model` argument and a `--model_name_or_path` argument.
We briefly describe the script(s) for each experiment below:

* **CrowS-Pairs**: Two scripts are provided for evaluating models against CrowS-Pairs: `experiments/crows.py` evaluates non-debiased
  models against CrowS-Pairs and `experiments/crows_debias.py` evaluates debiased models against CrowS-Pairs.
* **INLP Projection Matrix**: `experiments/inlp_projection_matrix.py` is used to compute INLP projection matrices.
* **SEAT**: Two scripts are provided for evaluating models against SEAT: `experiments/seat.py` evaluates non-debiased models against SEAT and
  `experiments/seat_debias.py` evaluates debiased models against SEAT.
* **StereoSet**: Two scripts are provided for evaluating models against StereoSet: `experiments/stereoset.py` evaluates non-debiased models against StereoSet and
  `experiments/stereoset_debias.py` evaluates debiased models against StereoSet.
* **SentenceDebias Subspace**: `experiments/sentence_debias_subspace.py` is used to compute SentenceDebias subspaces.
* **GLUE**: `experiments/run_glue.py` is used to run the GLUE benchmark.
* **Perplexity**: `experiments/perplexity.py` is used to compute perplexities on WikiText-2.

For a complete list of options for each experiment, run each experiment script with the `--h` option.
For example usages of these experiment scripts, refer to `batch_jobs`.
The commands used in `batch_jobs` produce the results presented in the paper.

### Notes
* To run SentenceDebias models against any of the benchmarks, you will first need to run `experiments/sentence_debias_subspace.py`.
* To run INLP models against any of the benchmarks, you will first need to run `experiments/inlp_projection_matrix.py`.
* `export` contains a collection of scripts to format the results into the tables presented in the paper.

## Running on an HPC Cluster
We provide scripts for running all of the experiments presented in the paper on a SLURM cluster in `batch_jobs`.
If you plan to use these scripts, make sure you customize `python_job.sh` to run the jobs on your cluster.
In addition, you will also need to change both the output (`-o`) and error (`-e`) paths.

## Acknowledgements
This repository makes use of code from the following repositories:

* [Towards Debiasing Sentence Representations](https://github.com/pliang279/sent_debias)
* [StereoSet: Measuring Stereotypical Bias in Pre-trained Language Models](https://github.com/moinnadeem/stereoset)
* [CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models](https://github.com/nyu-mll/crows-pairs)
* [On Measuring Social Biases in Sentence Encoders](https://github.com/w4ngatang/sent-bias)
* [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://github.com/shauli-ravfogel/nullspace_projection)
* [Towards Understanding and Mitigating Social Biases in Language Models](https://github.com/pliang279/lm_bias)
* [Self-Diagnosis and Self-Debiasing: A Proposal for Reducing Corpus-Based Bias in NLP](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00434/108865/Self-Diagnosis-and-Self-Debiasing-A-Proposal-for)

We thank the authors for making their code publicly available.

## Citation
If you use the code in this repository, please cite the following paper:

    @inproceedings{meade_2022_empirical,
        title = "An Empirical Survey of the Effectiveness of Debiasing Techniques for Pre-trained Language Models",
        author = "Meade, Nicholas  and Poole-Dayan, Elinor  and Reddy, Siva",
        booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
        month = may,
        year = "2022",
        address = "Dublin, Ireland",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.acl-long.132",
        doi = "10.18653/v1/2022.acl-long.132",
        pages = "1878--1898",
    }
