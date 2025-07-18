# Hyper-CL: Conditioning Sentence Representations with Hypernetworks

<div align=center>
  <img alt="Static Badge" src="https://img.shields.io/badge/HyperCL-1.0-blue">
  <img alt="Github Created At" src="https://img.shields.io/github/created-at/HYU-NLP/Hyper-CL">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/HYU-NLP/Hyper-CL">
  <a href="https://arxiv.org/abs/2403.09490">
      <img class="img-concert" src="https://img.shields.io/badge/arXiv-2403.09490-B21A1B"/>
  </a>
  <br>
</div>

#### Official Repository for "Hyper-CL: Conditioning Sentence Representations with Hypernetworks" [[Paper(arXiv)]](https://arxiv.org/abs/2403.09490))

##### Young Hyun Yoo, Jii Cha, Changhyeon Kim and Taeuk Kim. _Accepted to ACL2024 long paper_.

---

### Table of Contents

- [C-STS](#c-sts)
  - [Requirements](#requirements_csts)
  - [Data](#data)
  - [Training](#train_csts)
  - [Hyperparameter](#hyperparameters)
- [SimKGC](#SimKGC)
  - [Data](#dataset_simkgc)
  - [Training](#train_simkgc)
  - [Evaluation](#evaluation)
- [Citation](#citation)

## C-STS <a name="c-sts"></a>

In this section, we describe how to train a Hyper-CL model by using our code. This code based on [C-STS](https://github.com/princeton-nlp/c-sts/tree/main)

#### Requirements <a name="requirements_csts"></a>

Run the following script, the requirements are the same as C-STS.

#### Data <a name="data"></a>

Download the C-STS dataset and locate the file at data/ (reference the [C-STS repository](https://github.com/princeton-nlp/c-sts/tree/main) for more details.)

```bash
pip install -r requirements.txt
```

#### Training <a name="train_csts"></a>

We provide example training scripts for finetuning and evaluating the models in the paper. Go to C-STS/ and execute the following command

```bash
bash run_sts.sh
```

Following the arguments of [C-STS](https://github.com/princeton-nlp/c-sts/tree/main), we explain the additional arguments in following :

- `--objective`: (If you train Hyper-CL, you should use `triplet_cl_mse`)
- `--cl_temp`: Temperature for contrastive loss
- `--cl_in_batch_neg`: Add in-batch negative loss to main loss
- `--hypernet_scaler`: To set the value of K for low-rank implemented Hyper-CL _(i.e., hyper64-cl, hyper85-cl)_, we determine the divisor of the embedding size. For instance, in the base model, 'K=64' for hyper64-cl means the embedding size 768 is divided by 12. Thus, the hypernet_scaler is set to `12`.

- `--hypernet_dual`: Dual encoding that uses separate 2 encoders for sentences 1 and 2 and for the condition.

#### Hyperparameters <a name="hyperparameters"></a>

We use the following hyperparamters for training Hyper-CL:
|Emb.Model | Learning rate (lr) | Weight decay (wd) | Temperature (temp) |
|:--------------|:-----------:|:--------------:|:---------:|
| DiffCSE_base+hyper-cl | 3e-5 | 0.1 | 1.5 |
| DiffCSE_base+hyper64-cl | 1e-5 | 0.0 | 1.5 |
| SimCSE_base+hyper-cl | 3e-5 | 0.1 | 1.9 |
| SimCSE_base+hyper64-cl | 2e-5 | 0.1 | 1.7 |
| SimCSE_large+hyper-cl | 2e-5 | 0.1 | 1.5 |
| SimCSE_large+hyper85-cl | 1e-5 | 0.1 | 1.9 |

## SimKGC <a name="SimKGC"></a>
We provide example training scripts for finetuning and evaluating the models in the paper. Go to sim-kcg/ and execute the following command.
This code is based on [SimKCG](https://github.com/intfloat/SimKGC)

#### Preprocessing WN18RR dataset <a name="dataset_simkgc"></a>

```bash
bash scripts/preprocess.sh WN18RR
```

#### Training <a name="train_simkgc"></a> 

```bash
bash scripts/train_wn.sh
```

We explain the arguments in following:

- `--pretrained-model`: Backbone model checkpoint (`bert-base-uncased` or `bert-large-uncased`)
- `--encoding_type`: Encoding type (`bi_encoder` or `tri_encoder`)
- `--triencoder_head`: Triencoder head (`concat`, `hadamard` or `hypernet`)
- Refer to `config.py` for other arguments.

#### Evaluation for Perfomance and Inference Time <a name="evaluate"></a>

```bash
bash scripts/eval.sh ./checkpoint/WN18RR/model_best.mdl WN18RR
```

## Citation

Please cite our paper if you use Hyper-CL in your work:

```bash
@article{yoo2024hyper,
  title={Hyper-CL: Conditioning Sentence Representations with Hypernetworks},
  author={Yoo, Young Hyun and Cha, Jii and Kim, Changhyeon and Kim, Taeuk},
  journal={arXiv preprint arXiv:2403.09490},
  year={2024}
}
```
