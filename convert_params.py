# Tribute to https://github.com/wuyang556/paddlevision
from tqdm import tqdm
from collections import OrderedDict
import paddle.fluid as fluid
from torchvision import models
from paddle import vision

from models.PCA.model import Net as tchPCA
from models.PCA.model_pdl import Net as pdlPCA
from models.CIE.model import Net as TorchCIE
from models.CIE.model_pdl import Net as PaddleCIE
from models.IGM.model import Net as TorchIGM
from models.IGM.model_pdl import Net as PaddleIGM
from src.utils.model_sl import load_model

from src.utils.config import cfg


def convert_params(model_th, model_pd, model_path):
    """
    convert pytorch model's parameters into paddlepaddle model
    then save the converted model params as .pdparams
    :param model_th: pytorch model which has loaded pretrained parameters.
    :param model_pd: paddlepaddle dygraph model
    :param model_path: paddlepaddle dygraph model path
    """
    state_dict_th = model_th.state_dict()
    state_dict_pd = model_pd.state_dict()
    state_dict = OrderedDict()
    num_batches_tracked_th = 0
    for key_th in state_dict_th.keys():
        if "num_batches_tracked" in key_th:
            num_batches_tracked_th += 1

    for key_pd in tqdm(state_dict_pd.keys()):
        if key_pd in state_dict_th.keys():
            key_th = key_pd
        # Following tailor to our Graph Match work
        elif ('gnn_layer_list' in key_pd):
            key_th = key_pd.replace('list.', '')
        elif ('aff_layer_list' in key_pd):
            key_th = key_pd.replace('aff_layer_list.', 'affinity_')
        elif ('cross_layer' in key_pd):
            # noww only support **one** cross layer
            key_th = key_pd.replace('cross_layer', 'cross_graph_0')

        if "_mean" in key_pd:
            key_th = key_pd.replace("_mean", "running_mean")
        elif "_variance" in key_pd:
            key_th = key_pd.replace("_variance", "running_var")

        # if "fc" in key_th or "classifier":
        if "fc" in key_th or "cross_graph" in key_th:  # ad-hoc fix for CIE
            if len(state_dict_pd[key_pd].shape) < 4:
                state_dict[key_pd] = state_dict_th[key_th].numpy().astype(
                    "float32").transpose()
            else:
                state_dict[key_pd] = state_dict_th[key_th].numpy().astype(
                    "float32")
        else:
            state_dict[key_pd] = state_dict_th[key_th].numpy().astype(
                "float32")

    assert len(state_dict_pd.keys()) == len(state_dict.keys())
    if (len(state_dict.keys()) + num_batches_tracked_th
            != len(state_dict_th.keys())):
        raise ValueError(
            f'Statedict key mismatch: Paddle got {len(state_dict.keys())} keys'
            + f' and {num_batches_tracked_th} batch-norm params.'
            + f' But torch model has {len(state_dict_th.keys())} keys'
        )

    model_pd.set_dict(state_dict)

    fluid.dygraph.save_dygraph(model_pd.state_dict(), model_path=model_path)
    print("model converted successfully.")
    print(f'  Torch model has {len(state_dict_th.keys())} keys', end='. ')
    print(f'Including {num_batches_tracked_th} keys not used during inference')
    print(f'  Paddle model has {len(state_dict.keys())} keys')

def vgg_convert(paddle_param_path):
    with fluid.dygraph.guard():
        model_th = models.vgg16_bn(pretrained=True)
        model_pd = vision.models.vgg16(pretrained=False, batch_norm=True)
        model_path = paddle_param_path
        print(model_th.state_dict().keys(), sep='\n')
        print(len(model_th.state_dict().keys()))
        print(model_pd.state_dict().keys(), sep='\n')
        print(len(model_pd.state_dict().keys()))
        convert_params(model_th, model_pd, model_path)


def igm_convert(torch_param_path, paddle_param_path):
    with fluid.dygraph.guard():
        model_torch = TorchIGM()
        model_paddle = PaddleIGM()
        load_model(model_torch, torch_param_path)
        print('## Torch State Dict:', len(model_torch.state_dict().keys()))
        print(*model_torch.state_dict().keys(), sep='\n')
        print('## Paddle State Dict:', len(model_paddle.state_dict().keys()))
        print(*model_paddle.state_dict().keys(), sep='\n')
        convert_params(model_torch, model_paddle, paddle_param_path)


def cie_convert(torch_param_path, paddle_param_path):
    with fluid.dygraph.guard():
        model_torch = TorchCIE()
        model_paddle = PaddleCIE()
        load_model(
            model_torch,
            torch_param_path)
        model_path = paddle_param_path
        print('## Torch State Dict:', len(model_torch.state_dict().keys()))
        print(*model_torch.state_dict().keys(), sep='\n')
        print('## Paddle State Dict:', len(model_paddle.state_dict().keys()))
        print(*model_paddle.state_dict().keys(), sep='\n')
        convert_params(model_torch, model_paddle, model_path)


def pca_convert(torch_param_path, paddle_param_path):
    '''
    If u want to convert PCA
    please move this file to the parent dir
    '''
    with fluid.dygraph.guard():
        model_th = tchPCA()
        model_pd = pdlPCA()
        load_model(model_th, torch_param_path)
        model_path = paddle_param_path
        print('Torch State Dict:', len(model_th.state_dict().keys()))
        print(*model_th.state_dict().keys(), sep='\n')
        print('Paddle State Dict:', len(model_pd.state_dict().keys()))
        print(*model_pd.state_dict().keys(), sep='\n')

        convert_params(model_th, model_pd, model_path)


if __name__ == '__main__':
    from src.utils.parse_args import parse_args

    args = parse_args(
        'Deep learning of graph matching training & evaluation code.')
    INPUT_PATH = cfg.PRETRAINED_PATH
    OUTPUT_PATH = args.output_path
    ARCH = args.model_arch

    if OUTPUT_PATH is None or len(OUTPUT_PATH) == 0:
        print('No output path specified')
        print('Default to `./paddle_model.pdparams`')
        OUTPUT_PATH = './paddle_model.pdparams'

    if ARCH is None:
        raise ValueError('Please specify model architecture by `-m`')
    elif 'CIE' in ARCH:
        cie_convert(INPUT_PATH, OUTPUT_PATH)
    elif 'PCA' in ARCH:
        pca_convert(INPUT_PATH, OUTPUT_PATH)
    elif 'VGG16BN' in ARCH:
        vgg_convert(OUTPUT_PATH)
    elif 'IGM' in ARCH:
        igm_convert(INPUT_PATH, OUTPUT_PATH)
    else:
        raise ValueError(f'? {ARCH}')
