# ThinkMatch Paddlization: PaddlePaddle Implementation of ThinkMatch

This repository provides the non-official PaddlePaddle implementation of deep graph matching models in [ThinkMatch](https://github.com/Thinklab-SJTU/ThinkMatch).

## Contents

- [Getting Started](#get-started): Setting up the environment
  - [Base Env Setup](#basic-environment-for-thinkmatch): Set up the base environment for ThinkMatch
  - [Install PaddlePaddle](#installing-paddlepaddle): Install PaddlePaddle 2.1.0
  - [Prepare Datasets](#preparing-dataset): Prepare datasets for evaluation
- [Parameter Conversion](#parameter-conversion): Convert PyTorch models to PaddlePaddle
  - [Downloading Pretrained and Preconverted Models](#pretrained-models)
  - [Parameter Conversion](#running-parameter-conversion)
- [Evaluation](#evaluation-on-paddlepaddle): Running evaluation with the new PaddlePaddle implementation.
- [Support Table](#support-table): List of models and datasets that are supported by this implementation.

## Get Started

Please follow the steps to set up the environment.

### Basic Environment for ThinkMatch

We recommend using a docker.

- The docker maintained by ThinkLab of SJTU is available [here](https://hub.docker.com/r/runzhongwang/thinkmatch).
- Please use `runzhongwang/thinkmatch:torch1.6.0-cuda10.1-cudnn7-pyg1.6.3-pygmtools0.1.14` image.

For other ways to set up the basic environment, please visit the [upstream](https://github.com/Thinklab-SJTU/ThinkMatch) and check the README there.

### Installing PaddlePaddle

After setting up the basic environment, please install PaddlePaddle version `2.1.0`.

It can be done by

```sh
conda install paddlepaddle-gpu==2.1.0 cudatoolkit=10.1 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/
```

or

```sh
python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

#### Note

1. Always install **PaddlePaddle 2.1.0**.
2. If you are using a docker, please **install PaddlePaddle inside the docker**
3. If you are using the docker image provided by ThinkMatch, please **DO install PaddlePaddle with CUDA 10.1**.

### Preparing Dataset

Currently we support evaluation on 3 visual datasets: [PascalVOC](#pascalvoc), [WillowObject](#willow-object-class) and [IMC-PT-SparseGM](#imc-pt-sparsegm). Since this implementation uses `pytmtools` for loading datasets, it should be able to automatically download the datasets. However, you can still prepare the datasets manually if errors occur during download.

The dataset preparation is the same as the upstream PyTorch implementation.

#### PascalVOC

1. Download [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2011/index.html).
   - The file structure should look like `data/PascalVOC/TrainVal/VOCdevkit/VOC2011`
2. Download [PascalVOC Keypoint Annotation](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/voc2011_keypoints_Feb2012.tgz).
   - The file structure should look like `data/PascalVOC/annotations`
3. Download train-test split specification. This file should already be included in the repository, but if it does not exist, it must be downloaded manually from upstream repository.
   - The file structure should look like `data/PascalVOC/voc2011_pairs.npz`

#### Willow-Object-Class

1. Download [Willow-Object-Class](http://www.di.ens.fr/willow/research/graphlearning/WILLOW-ObjectClass_dataset.zip) dataset
2. Extract the data
   - The file structure should look like `data/WillowObject/WILLOW-ObjectClass`

#### IMC-PT-SparseGM

1. Download IMC-PT-SparseGM from [google drive](https://drive.google.com/file/d/1Po9pRMWXTqKK2ABPpVmkcsOq-6K_2v-B/view?usp=sharing)
2. Extract the data
   - The file structure should look like `data/IMC_PT_SparseGM/annotations`

## Parameter Conversion

Since training a model is currently not supported in this PaddlePaddle implementation, you have to convert a set of pretrained parameters of a PyTorch model to parameters for a PaddlePaddle model.

### Pretrained Models

- The pretrained PyTorch models can be found on the [google drive](https://drive.google.com/drive/folders/11xAQlaEsMrRlIVc00nqWrjHf8VOXUxHQ). These models are maintained by ThinkLab of SJTU.
- We also provide pre-converted PaddlePaddle parameters. They are available at [jbox](https://jbox.sjtu.edu.cn/l/31Ll1T) and [google drive](https://drive.google.com/drive/folders/1qO93jTgYgVi9N2FZzZXTMw2QF4HLkOzx?usp=sharing).

### Running Parameter Conversion

The `convert_params.py` provides basic functionality of converting a PyTorch model to a PaddlePaddle model.

For example, to convert a CIE model trained on PascalVOC, you can run

```sh
python convert_params.py --cfg ./experiments/vgg16_cie_voc.yaml -m CIE -i ./pretrained/pretrained_params_vgg16_cie_voc.pt -o ./pretrained/paddle_vgg16_cie_voc
```

#### Commandline Arguments for Parameter Conversion

`convert_params.py` accepts the following commandline arguments

- `--cfg`: The experiment config. This config is required because it is used for constructing the model to load the parameters.
  - Please make sure the model architecture specified in the config matches the model architecture required by the paramters.
- `--model_arch -m`: Model architecture.
  - Available options are: `GMN` `PCA` `ICPA` `CIE` `NGMv1` and `SSStM`
  - Note: This argument is **sensitive to capital letters**.
- `--input_path -i`: Path to a pretrained PyTorch model parameter
- `--output_path -o`: Path where the output PaddlePaddle model parameter will be stored to.
  - Note: Do not append `.pdparams` at the end of this argument, because PaddlePaddle automatically appends one when saving state dict.
    - e.g. use `-o my_paddle_model` instead of `-o my_paddle_model.pdparams`

#### Cheatsheet

We provide a [cheetsheet](./convert_params_cheatsheet.md) which lists all commands to convert supported models from PyTorch to PaddlePaddle. You can simple copy and paste them.

#### Parameter Conversion Support List

Currently, we support converting the following PyTorch models to PaddlePaddle

- [GMN](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#gmn)
- [PCA-GM and IPCA-GM](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#pca-gm)
- [CIE](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#cie-h)
- [NGMv1](https://thinkmatch.readthedocs.io/en/latest/guide/models.html#ngm)
- [SSStM](https://github.com/eliphatfs/ThinkMatch)

## Evaluation on PaddlePaddle

Running evaluation in this PaddlePaddle implementation is the same as running evaluation on the original PyTorch implementation. After setting up environment and downloading datasets, it can be summarized into the following steps.

1. Put converted PaddlePaddle paramters under `./pretrained` directory.
   - By default, the config yaml specifies that the pretrained parameter is located in `./pretrained` directory.
   - If your parameters are not in this directory, either move them to `./pretrained` or modify the config yaml in `./experiments`
2. Execute `eval_pdl_retrofit.py`.
   - We entirely refined the original evaluation script to support more models and more datasets, so the original evaluation script is deprecated.
   - Since this implementation introduces a lot of changes, we do NOT guarantee that the original `eval_pdl.py` is still runnable. **Always execute `eval_pdl_retrofit`**.

For example, the evaluation of CIE on PascalVOC can be done by

```sh
python eval_pdl_retrofit.py --cfg ./experiments/vgg16_cie_voc.yaml
```

## Support Table

This table lists the models and datasets that we have supported. Note that we currently only support running evaluation and do not support running training.

|   Models    |     PascalVOC      |     WillowObject     |   IMC-PT-SparseGM    |
| :---------: | :----------------: | :------------------: | :------------------: |
|   **GMN**   | :heavy_check_mark: |  :heavy_check_mark:  |         :x:          |
| **PCA-GM**  | :heavy_check_mark: |  :heavy_check_mark:  |  :heavy_check_mark:  |
| **IPCA-GM** | :heavy_check_mark: |  :heavy_check_mark:  | No Pretrained Params |
|  **CIE-H**  | :heavy_check_mark: |  :heavy_check_mark:  |  :heavy_check_mark:  |
|  **GANN**   |  Not Implemented   |   Not Implemented    |   Not Implemented    |
|  **NGMv1**  | :heavy_check_mark: |  :heavy_check_mark:  |         :x:          |
|  **NGMv2**  |  Not Implemented   |   Not Implemented    |   Not Implemented    |
|  **BBGM**   |  Not Implemented   |   Not Implemented    |   Not Implemented    |
|  **SSStM**  | :heavy_check_mark: | No Pretrained Params | No Pretrained Params |
