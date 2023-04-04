from .setr import SETR
from .upernet import UperNet
from .dpt import DPTSeg
from .segmentor import Segmentor
from .trans2seg import Trans2Seg
from .segformer import Segformer
from .fcn import FCN
from .paddle_EMRT import EMRT
from .paddle_EMRT_cswin_backbone import EMRT_CSwin
from .paddle_EMRT_hrnet_backbone import EMRT_HRNet
from src.models.backbones import segformer_paddleSeg


def get_model(config):
    if "SETR" in config.MODEL.NAME:
        model = SETR(config)
    elif "UperNet" in config.MODEL.NAME:
        model = UperNet(config)
    elif "DPT" in config.MODEL.NAME:
        model = DPTSeg(config)
    elif "Segmenter" in config.MODEL.NAME:
        model = Segmentor(config)
    elif 'Trans2Seg' in config.MODEL.NAME:
        model = Trans2Seg(config)
    elif "Segformer" == config.MODEL.NAME:
        model = Segformer(config)
    elif "PaddleSeg_Segformer" == config.MODEL.NAME:
        model = segformer_paddleSeg.SegFormer_B4(num_classes=config.DATA.NUM_CLASSES,
                                                 pretrained=config.MODEL.PRETRAINED)
    elif "FCN" in config.MODEL.NAME:
        model = FCN(config)

    elif "EMRT_CSwin" in config.MODEL.NAME or "EMRT_ViT" in config.MODEL.NAME:
        model = EMRT_CSwin(config)
    elif "EMRT_HRNet" in config.MODEL.NAME:
        model = EMRT_HRNet(config)
    elif "EMRT" in config.MODEL.NAME or "EMRT_Segformer" in config.MODEL.NAME:
        model = EMRT(config)

    return model
