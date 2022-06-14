"""Implement pre-trained histopathology model from Ciga et al. 2022."""

import timm
import torch


@timm.models.register_model
def resnet18_ciga_ssl(pretrained=False, **kwargs):
    """Create model from https://arxiv.org/abs/2011.13971.

    See https://github.com/ozanciga/self-supervised-histopathology.
    """
    from timm.models.resnet import BasicBlock, ResNet

    def _filter_checkpoint_fn(state_dict):
        # Make timm compatible: pytorchnative_tenpercent_resnet18.ckpt
        state_dict = state_dict["state_dict"]
        state_dict = {k.replace("model.resnet.", ""): v for k, v in state_dict.items()}
        # Remove keys not present in the timm resnet18 model.
        del state_dict["fc.1.weight"], state_dict["fc.1.bias"], state_dict["fc.3.weight"], state_dict["fc.3.bias"]
        # These weights get removed anyway, because we change the number of output classes in the model.
        # So in effect, we are using the pre-trained feature extractor part of resnet18.
        # Definitely do not use this model with 1000 classes!
        state_dict["fc.weight"] = torch.zeros(1000, 512, dtype=torch.float32)
        state_dict["fc.bias"] = torch.zeros(1000, dtype=torch.float32)
        return state_dict

    if kwargs["pretrained_cfg"] is None:
        kwargs["pretrained_cfg"] = {
            'url': "https://github.com/ozanciga/self-supervised-histopathology/releases/download/nativetenpercent/pytorchnative_tenpercent_resnet18.ckpt",
            # The number of classes it not actually 1000...
            # We set it here because that's what imagenet uses.
            'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
            'crop_pct': 0.875, 'interpolation': 'bilinear',
            # https://github.com/ozanciga/self-supervised-histopathology/issues/2#issuecomment-794469448
            'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
            'first_conv': 'conv1', 'classifier': 'fc',
        }

    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    return timm.models.helpers.build_model_with_cfg(
        model_cls=ResNet,
        variant="resnet18",
        pretrained=pretrained,
        pretrained_filter_fn=_filter_checkpoint_fn,
        **model_args)
