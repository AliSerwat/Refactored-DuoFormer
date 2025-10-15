from .model import MyModel, HybridModel, ViTBase16, count_parameters
from .model_wo_extra_params import MyModel_no_extra_params
from .resnet50ssl import (
    ResNetTrunk,
    ResNetTrunkByScale,
    get_pretrained_url,
    resnet50FeatureExtractor,
    resnet50,
    ResNet50withFC,
)


def build_model(
    depth=12,
    patch_size=49,
    embed_dim=256,
    num_heads=6,
    init_values=1e-5,
    num_classes=100,
    num_layers=4,
    proj_dim=384,
    model_ver="scaleformer",
    pretrained=True,
    freeze=True,
):
    return MyModel(
        depth=depth,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        init_values=init_values,
        num_layers=num_layers,
        proj_dim=proj_dim,
        model_ver=model_ver,
        pretrained=pretrained,
        freeze=freeze,
    )


def build_model_no_extra_params(
    depth=12,
    embed_dim=256,
    num_heads=6,
    num_classes=100,
    num_layers=4,
    num_patches=49,
    proj_dim=384,
    mlp_ratio=4.0,
    attn_drop_rate=0.0,
    proj_drop_rate=0.0,
    freeze_backbone=True,
    backbone="r50",
    pretrained=True,
):

    return MyModel_no_extra_params(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_classes=num_classes,
        num_layers=num_layers,
        num_patches=num_patches,
        proj_dim=proj_dim,
        mlp_ratio=mlp_ratio,
        attn_drop_rate=attn_drop_rate,
        proj_drop_rate=proj_drop_rate,
        freeze_backbone=freeze_backbone,
        backbone=backbone,
        pretrained=pretrained,
    )


def build_hybrid(num_classes=100, num_blocks=12, proj_dim=768, num_heads=12):
    """
    Build hybrid ResNet-ViT model.

    Args:
        num_classes: Number of output classes
        num_blocks: Number of transformer blocks
        proj_dim: Projection dimension
        num_heads: Number of attention heads

    Returns:
        Hybrid model instance
    """
    return HybridModel(
        num_classes=num_classes, num_blocks=num_blocks, proj_dim=proj_dim
    )
