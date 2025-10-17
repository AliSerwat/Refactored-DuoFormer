import torch
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.layers import Mlp, DropPath
from timm.models.resnetv2 import ResNetV2
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from .transformer.multi_vision_transformer import MultiscaleTransformer
from .attention.multiscale_attn import MultiScaleAttention, MultiscaleBlock
from .projection.projection_head import (
    Projection,
    Channel_Projector_layer1,
    Channel_Projector_layer2,
    Channel_Projector_layer3,
    Channel_Projector_All,
)
from .attention.scale_attention import (
    AttentionForScale,
    ScaleBlock,
    ScaleFormer,
    AttentionForPatch,
    PatchBlock,
    MultiscaleFormer,
)
from .backbone.backbone import Backbone, Backbone2


class MyModel_no_extra_params(nn.Module):
    def __init__(
        self,
        depth=None,
        embed_dim=768,
        num_heads=12,
        init_values=1e-5,
        num_classes=2,
        num_layers=4,
        num_patches=49,
        mlp_ratio=4.0,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        proj_dim=768,
        freeze_backbone=True,
        backbone="r50",
        scale_token="random",
        patch_attn=True,
        pretrained=True,
    ):
        super().__init__()

        # Input validation
        assert depth is not None and depth > 0, "depth must be a positive integer"
        assert (
            embed_dim > 0 and embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be positive and divisible by num_heads ({num_heads})"
        assert num_heads > 0, "num_heads must be positive"
        assert num_layers in [
            2,
            3,
            4,
        ], f"num_layers must be 2, 3, or 4, got {num_layers}"
        assert backbone in [
            "r50",
            "r18",
            "r50_Swav",
        ], f"backbone must be 'r50', 'r18', or 'r50_Swav', got {backbone}"
        assert scale_token in [
            "random",
            "channel",
        ], f"scale_token must be 'random' or 'channel', got {scale_token}"

        self.num_layers = num_layers
        self.proj_dim = proj_dim
        self.backbone = backbone
        self.scale_token = scale_token
        self.patch_attn = patch_attn
        self.name = (
            f"DuoFormer_{backbone}_L{num_layers}"  # Add name attribute for saving
        )

        if backbone == "r50":
            if pretrained:
                self.resnet_projector = nn.Sequential(
                    *list(models.resnet50(weights=ResNet50_Weights.DEFAULT).children())[
                        :-2
                    ]
                )
                print("✅ ResNet-50 pretrained weights loaded (IMAGENET1K_V1)")
            else:
                self.resnet_projector = nn.Sequential(
                    *list(models.resnet50().children())[:-2]
                )
                print("⚠️  ResNet-50 initialized with random weights")
        elif backbone == "r18":
            if pretrained:
                self.resnet_projector = nn.Sequential(
                    *list(models.resnet18(weights=ResNet18_Weights.DEFAULT).children())[
                        :-2
                    ]
                )
                print("✅ ResNet-18 pretrained weights loaded (IMAGENET1K_V1)")
            else:
                self.resnet_projector = nn.Sequential(
                    *list(models.resnet18().children())[:-2]
                )
                print("⚠️  ResNet-18 initialized with random weights")
        elif backbone == "r50_Swav":
            # Use standard ResNet-50 instead of legacy SwAV implementation
            self.resnet_projector = nn.Sequential(
                *list(
                    models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).children()
                )[:-2]
            )
            print("✅ ResNet-50 loaded with ImageNet weights (SwAV not available)")

        if freeze_backbone:
            for param in self.resnet_projector.parameters():
                param.requires_grad = False
            print("🔒 Backbone frozen during training")

        if self.scale_token == "random":
            self.channel_token = torch.nn.Parameter(torch.randn(1, 1, 1, self.proj_dim))
            nn.init.normal_(self.channel_token, std=0.036)  # 0.036,1e-6
        elif self.scale_token == "channel":  # 0.036,1e-6
            self.channel_proj1 = Channel_Projector_layer1(backbone=backbone)
            self.channel_proj2 = Channel_Projector_layer2(backbone=backbone)
            self.channel_proj3 = Channel_Projector_layer3()
            self.channel_proj_all = Channel_Projector_All(backbone=backbone)

        self.projection = Projection(
            num_layers=self.num_layers, proj_dim=self.proj_dim, backbone=backbone
        )
        self.vision_transformer = MultiscaleFormer(
            depth=depth,
            scales=self.num_layers,
            num_heads=num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            qk_norm=False,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=None,
            act_layer=None,
            init_values=None,
            num_classes=num_classes,
            num_patches=num_patches,
        )
        print("✅ Multi-scale transformer initialized")

        # self.scale_former = ScaleFormer(depth=12,scales=self.num_layers,num_heads=12,embed_dim=self.proj_dim)   # only scale attentions, consistent with pretrained hybrid

        # Index mapping for multi-scale features
        # Scale 3: 7x7   -> 49 patches (original resolution)
        # Scale 2: 14x14 -> 196 patches (2x upsampled)
        # Scale 1: 28x28 -> 784 patches (4x upsampled)
        # Scale 0: 56x56 -> 3136 patches (8x upsampled)
        self.index = {}
        for i in range(4):
            self.index[f"{4 - i - 1}"] = torch.empty([49, 4**i], dtype=torch.int64)
        for r in range(7):
            for c in range(7):
                p = r * 7 + c
                self.index["3"][p, :] = p
                self.index["2"][p, :] = torch.IntTensor(
                    [
                        2 * r * 14 + 2 * c,
                        (2 * r + 1) * 14 + 2 * c,
                        2 * r * 14 + (2 * c + 1),
                        (2 * r + 1) * 14 + (2 * c + 1),
                    ]
                )
                self.index["1"][p, :] = torch.IntTensor(
                    [
                        4 * r * 28 + 4 * c,
                        4 * r * 28 + 4 * c + 1,
                        4 * r * 28 + 4 * c + 2,
                        4 * r * 28 + 4 * c + 3,
                        (4 * r + 1) * 28 + 4 * c,
                        (4 * r + 1) * 28 + 4 * c + 1,
                        (4 * r + 1) * 28 + 4 * c + 2,
                        (4 * r + 1) * 28 + 4 * c + 3,
                        (4 * r + 2) * 28 + 4 * c,
                        (4 * r + 2) * 28 + 4 * c + 1,
                        (4 * r + 2) * 28 + 4 * c + 2,
                        (4 * r + 2) * 28 + 4 * c + 3,
                        (4 * r + 3) * 28 + 4 * c,
                        (4 * r + 3) * 28 + 4 * c + 1,
                        (4 * r + 3) * 28 + 4 * c + 2,
                        (4 * r + 3) * 28 + 4 * c + 3,
                    ]
                )
                self.index["0"][p, :] = torch.IntTensor(
                    [
                        8 * r * 56 + 8 * c,
                        8 * r * 56 + 8 * c + 1,
                        8 * r * 56 + 8 * c + 2,
                        8 * r * 56 + 8 * c + 3,
                        8 * r * 56 + 8 * c + 4,
                        8 * r * 56 + 8 * c + 5,
                        8 * r * 56 + 8 * c + 6,
                        8 * r * 56 + 8 * c + 7,
                        (8 * r + 1) * 56 + 8 * c,
                        (8 * r + 1) * 56 + 8 * c + 1,
                        (8 * r + 1) * 56 + 8 * c + 2,
                        (8 * r + 1) * 56 + 8 * c + 3,
                        (8 * r + 1) * 56 + 8 * c + 4,
                        (8 * r + 1) * 56 + 8 * c + 5,
                        (8 * r + 1) * 56 + 8 * c + 6,
                        (8 * r + 1) * 56 + 8 * c + 7,
                        (8 * r + 2) * 56 + 8 * c,
                        (8 * r + 2) * 56 + 8 * c + 1,
                        (8 * r + 2) * 56 + 8 * c + 2,
                        (8 * r + 2) * 56 + 8 * c + 3,
                        (8 * r + 2) * 56 + 8 * c + 4,
                        (8 * r + 2) * 56 + 8 * c + 5,
                        (8 * r + 2) * 56 + 8 * c + 6,
                        (8 * r + 2) * 56 + 8 * c + 7,
                        (8 * r + 3) * 56 + 8 * c,
                        (8 * r + 3) * 56 + 8 * c + 1,
                        (8 * r + 3) * 56 + 8 * c + 2,
                        (8 * r + 3) * 56 + 8 * c + 3,
                        (8 * r + 3) * 56 + 8 * c + 4,
                        (8 * r + 3) * 56 + 8 * c + 5,
                        (8 * r + 3) * 56 + 8 * c + 6,
                        (8 * r + 3) * 56 + 8 * c + 7,
                        (8 * r + 4) * 56 + 8 * c,
                        (8 * r + 4) * 56 + 8 * c + 1,
                        (8 * r + 4) * 56 + 8 * c + 2,
                        (8 * r + 4) * 56 + 8 * c + 3,
                        (8 * r + 4) * 56 + 8 * c + 4,
                        (8 * r + 4) * 56 + 8 * c + 5,
                        (8 * r + 4) * 56 + 8 * c + 6,
                        (8 * r + 4) * 56 + 8 * c + 7,
                        (8 * r + 5) * 56 + 8 * c,
                        (8 * r + 5) * 56 + 8 * c + 1,
                        (8 * r + 5) * 56 + 8 * c + 2,
                        (8 * r + 5) * 56 + 8 * c + 3,
                        (8 * r + 5) * 56 + 8 * c + 4,
                        (8 * r + 5) * 56 + 8 * c + 5,
                        (8 * r + 5) * 56 + 8 * c + 6,
                        (8 * r + 5) * 56 + 8 * c + 7,
                        (8 * r + 6) * 56 + 8 * c,
                        (8 * r + 6) * 56 + 8 * c + 1,
                        (8 * r + 6) * 56 + 8 * c + 2,
                        (8 * r + 6) * 56 + 8 * c + 3,
                        (8 * r + 6) * 56 + 8 * c + 4,
                        (8 * r + 6) * 56 + 8 * c + 5,
                        (8 * r + 6) * 56 + 8 * c + 6,
                        (8 * r + 6) * 56 + 8 * c + 7,
                        (8 * r + 7) * 56 + 8 * c,
                        (8 * r + 7) * 56 + 8 * c + 1,
                        (8 * r + 7) * 56 + 8 * c + 2,
                        (8 * r + 7) * 56 + 8 * c + 3,
                        (8 * r + 7) * 56 + 8 * c + 4,
                        (8 * r + 7) * 56 + 8 * c + 5,
                        (8 * r + 7) * 56 + 8 * c + 6,
                        (8 * r + 7) * 56 + 8 * c + 7,
                    ]
                )

    def get_features(self, x):
        """Extract multi-scale features from backbone."""
        features = {}

        if self.backbone == "r18":
            # ResNet-18 sliced model: 0=conv1, 1=bn1, 2=relu, 3=maxpool, 4=layer1, 5=layer2, 6=layer3, 7=layer4
            # For num_layers=2, we want layer3 (6) and layer4 (7) -> keys "2" and "3"
            layer_mapping = {"6": "2", "7": "3"}
        else:
            # ResNet-50 layer mapping: layer1->0, layer2->1, layer3->2, layer4->3
            layer_mapping = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}

        for name, module in self.resnet_projector.named_children():
            x = module(x)
            if name in layer_mapping:
                features[layer_mapping[name]] = x

        return features

    def forward(self, x):
        if self.backbone == "r50_Swav":
            # x = self.get_features(x)  # feature extraction
            x = self.resnet_projector(
                x
            )  # feature extraction for resnet 50 pretrained on TCGA, output is a list contains all scales
            x = {str(i): output for i, output in enumerate(x)}
        else:
            x = self.get_features(x)

        # Move index tensors to same device as input (only once)
        if not hasattr(self, "_indices_device_moved"):
            device = x[next(iter(x.keys()))].device
            for key in self.index:
                self.index[key] = self.index[key].to(device)
            self._indices_device_moved = True

        if self.scale_token == "channel":
            channel_fuse = {}
            channel_fuse["0"] = self.channel_proj1(x["0"])
            channel_fuse["1"] = self.channel_proj2(x["1"])
            channel_fuse["2"] = self.channel_proj3(x["2"])
            channel_fuse["3"] = x["3"]
            channel_fuse_all = torch.cat(
                [channel_fuse[key] for key in sorted(channel_fuse.keys())], dim=1
            )  # gather channel-wise information
            channel_token = (
                self.channel_proj_all(channel_fuse_all)
                .unsqueeze(-1)
                .permute(0, 2, 3, 1)
            )  # 49,1,768
            B, _, _, _ = channel_token.shape
        else:
            # Get batch size from first available feature map
            first_key = next(iter(x.keys()))
            B = x[first_key].shape[0]
        C = self.proj_dim
        if self.num_layers == 2:
            x = self.projection({"2": x["2"], "3": x["3"]})
            x["3"] = x["3"].reshape(B, C, -1)
            x["2"] = x["2"].reshape(B, C, -1)
            # print(self.index['3'])
            # print(x['3'])
            x["3"] = x["3"][
                :, :, self.index["3"]
            ]  # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x["2"] = x["2"][
                :, :, self.index["2"]
            ]  # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x = torch.cat((x["3"], x["2"]), dim=-1).permute(
                0, 2, 3, 1
            )  # [64, 768, 49, 5] -> [64, 49, 5, 768]
        elif self.num_layers == 4:
            x = self.projection({"0": x["0"], "1": x["1"], "2": x["2"], "3": x["3"]})
            x["3"] = x["3"].reshape(B, C, -1)
            x["2"] = x["2"].reshape(B, C, -1)
            x["3"] = x["3"][
                :, :, self.index["3"]
            ]  # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x["2"] = x["2"][
                :, :, self.index["2"]
            ]  # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x["1"] = x["1"].reshape(B, C, -1)
            x["0"] = x["0"].reshape(B, C, -1)
            x["1"] = x["1"][:, :, self.index["1"]]
            x["0"] = x["0"][:, :, self.index["0"]]
            x = torch.cat((x["3"], x["2"], x["1"], x["0"]), dim=-1).permute(0, 2, 3, 1)
        elif self.num_layers == 3:
            x = self.projection({"1": x["1"], "2": x["2"], "3": x["3"]})
            x["3"] = x["3"].reshape(B, C, -1)
            x["2"] = x["2"].reshape(B, C, -1)
            x["3"] = x["3"][
                :, :, self.index["3"]
            ]  # [64, 768, 7, 7] -> [64, 49, 1, 7, 7]
            x["2"] = x["2"][
                :, :, self.index["2"]
            ]  # [64, 768, 14, 14] -> [64, 49, 4, 14, 14]
            x["1"] = x["1"].reshape(B, C, -1)
            x["1"] = x["1"][:, :, self.index["1"]]
            x = torch.cat((x["3"], x["2"], x["1"]), dim=-1).permute(0, 2, 3, 1)

        if self.scale_token == "channel":
            x = torch.cat((channel_token, x), dim=2)
        elif self.scale_token == "random":
            x = torch.cat((self.channel_token.expand(B, 49, -1, -1), x), dim=2)

        output = self.vision_transformer(x)  # multiscale transformer
        return output
