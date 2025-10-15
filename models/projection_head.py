import torch
from torch import nn


class Projection(nn.Module):
    """
    Multi-scale feature projection layer for DuoFormer.

    Projects features from different ResNet layers to a common dimension
    for multi-scale transformer processing. Supports 1-4 scale configurations
    with different backbone architectures.

    Args:
        num_layers (int): Number of feature scales to use (1, 2, 3, or 4).
            - 1: Only final layer (2048 channels for ResNet-50)
            - 2: Final + penultimate layers (2048 + 1024 channels)
            - 3: Final + penultimate + third layers (2048 + 1024 + 512 channels)
            - 4: All four layers (2048 + 1024 + 512 + 256 channels)
        proj_dim (int): Target projection dimension (typically 384 or 768)
        backbone (str): Backbone architecture ('r50' for ResNet-50, 'r18' for ResNet-18)

    Attributes:
        proj_heads: Conv2d layers for projecting each scale to proj_dim
        num_layers: Number of scales being used
        proj_dim: Target projection dimension

    Example:
        >>> # 2-scale projection for ResNet-50
        >>> proj = Projection(num_layers=2, proj_dim=768, backbone='r50')
        >>> # Forward pass with multi-scale features
        >>> output = proj({'3': feat_2048, '2': feat_1024})
    """

    def __init__(self, num_layers=2, proj_dim=768, backbone="r50"):
        super().__init__()
        if backbone == "r50":
            if num_layers == 1:
                self.proj_heads = nn.Conv2d(
                    2048, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                # self.proj_heads = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self._initialize_weights(self.proj_heads)
            elif num_layers == 2:
                self.proj_heads3 = nn.Conv2d(
                    2048, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads2 = nn.Conv2d(
                    1024, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
            elif num_layers == 3:
                self.proj_heads3 = nn.Conv2d(
                    2048, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads2 = nn.Conv2d(
                    1024, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads1 = nn.Conv2d(
                    512, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
            elif num_layers == 4:
                self.proj_heads3 = nn.Conv2d(
                    2048, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads2 = nn.Conv2d(
                    1024, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads1 = nn.Conv2d(
                    512, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads0 = nn.Conv2d(
                    256, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
                self._initialize_weights(self.proj_heads0)
        elif backbone == "r18":
            if num_layers == 1:
                self.proj_heads = nn.Conv2d(
                    512, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self._initialize_weights(self.proj_heads)
            elif num_layers == 2:
                # self.proj_heads3 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads2 = nn.Conv2d(
                    256, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads1 = nn.Conv2d(
                    128, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                # self.proj_heads0 = nn.Conv2d(64, proj_dim, kernel_size=(1,1),stride=(1,1))
                # self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
                # self._initialize_weights(self.proj_heads0)
            elif num_layers == 3:
                # self.proj_heads3 = nn.Conv2d(512, proj_dim, kernel_size=(1,1),stride=(1,1))
                self.proj_heads0 = nn.Conv2d(
                    64, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads2 = nn.Conv2d(
                    256, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads1 = nn.Conv2d(
                    128, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                # self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads0)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
            elif num_layers == 4:
                self.proj_heads3 = nn.Conv2d(
                    512, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads2 = nn.Conv2d(
                    256, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads1 = nn.Conv2d(
                    128, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self.proj_heads0 = nn.Conv2d(
                    64, proj_dim, kernel_size=(1, 1), stride=(1, 1)
                )
                self._initialize_weights(self.proj_heads3)
                self._initialize_weights(self.proj_heads2)
                self._initialize_weights(self.proj_heads1)
                self._initialize_weights(self.proj_heads0)

        # else:
        #     self.proj_heads = nn.ModuleDict()
        #     for i in range(num_layers):
        #         # self.proj_heads[f'{i}'] = nn.Linear(256 * (2 ** i), proj_dim)
        #         self.proj_heads[f'{3-i}'] = nn.Conv2d(256 * (2 ** (3-i)), proj_dim, kernel_size=(1,1),stride=(1,1))
        #         self._initialize_weights(self.proj_heads[f'{i}'])

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # nn.init.xavier_uniform_(module.weight)  # nn.init.xavier_uniform_() or nn.init.xavier_normal_()
            nn.init.kaiming_normal_(
                module.weight
            )  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                # nn.init.constant_(module.bias, 0)
                nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        """
        Forward pass through projection layers.

        Args:
            x (dict): Dictionary of multi-scale features with keys '0', '1', '2', '3'
                corresponding to different ResNet layers. Each value is a tensor of
                shape (B, C, H, W) where C varies by layer.

        Returns:
            dict or torch.Tensor:
                - If multiple scales: Dictionary with same keys as input, containing
                  projected features of shape (B, proj_dim, H, W)
                - If single scale: Tensor of shape (B, proj_dim, H, W)

        Example:
            >>> # Multi-scale input
            >>> features = {'3': feat_2048, '2': feat_1024}
            >>> output = proj(features)
            >>> # Single scale input
            >>> single_feat = {'3': feat_2048}
            >>> output = proj(single_feat)  # Returns tensor directly
        """
        if len(x) != 1:
            proj_features = {}
            for k, fea in x.items():
                N, C, H, W = fea.shape
                if k == "3":
                    proj_features[k] = self.proj_heads3(fea)
                elif k == "2":
                    proj_features[k] = self.proj_heads2(fea)
                elif k == "1":
                    proj_features[k] = self.proj_heads1(fea)
                elif k == "0":
                    proj_features[k] = self.proj_heads0(fea)
        else:
            proj_features = self.proj_heads(x)
        return proj_features


class Channel_Projector_layer1(nn.Module):
    """
    Channel projection layer for the first ResNet layer (layer 0).

    Reduces spatial dimensions of features from the first ResNet layer
    through convolutional downsampling and projects to a common dimension.

    Args:
        backbone (str): Backbone architecture ('r50' or 'r18')
            - 'r50': Input 256 channels, projects to 768
            - 'r18': Input 64 channels, projects to 384

    Architecture:
        - Conv2d layers for spatial downsampling (stride=2)
        - BatchNorm + ReLU activation
        - Final projection to target dimension
    """

    def __init__(self, backbone="r50"):
        super().__init__()
        # Convolutional layers to reduce spatial dimensions
        if backbone == "r50":
            self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        elif backbone == "r18":
            self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        # Pooling layer to downsample to 7x7
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights(self.conv1)
        self._initialize_weights(self.conv2)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.activation1(x)
        x = self.conv2(x)
        # x = self.norm2(x)
        # x = self.activation2(x)
        x = self.pool(x)
        return x

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight
            )  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)


class Channel_Projector_layer2(nn.Module):
    """
    Channel projection layer for the second ResNet layer (layer 1).

    Processes features from the second ResNet layer with minimal processing
    since spatial dimensions are already suitable for multi-scale attention.

    Args:
        backbone (str): Backbone architecture ('r50' or 'r18')
            - 'r50': Input 512 channels, projects to 768
            - 'r18': Input 128 channels, projects to 384

    Architecture:
        - Direct projection without spatial downsampling
        - BatchNorm + ReLU activation
        - Final projection to target dimension
    """

    def __init__(self, backbone="r50"):
        super().__init__()
        # Convolutional layers to reduce spatial dimensions
        if backbone == "r50":
            self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        elif backbone == "r18":
            self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._initialize_weights(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.norm1(x)
        # x = self.activation1(x)
        x = self.pool(x)
        return x

    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight
            )  # nn.init.kaiming_uniform_ ()or nn.init.kaiming_normal_()
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6)


class Channel_Projector_layer3(nn.Module):
    """
    Channel projection layer for the third ResNet layer (layer 2).

    Processes features from the third ResNet layer with minimal processing
    since spatial dimensions are already suitable for multi-scale attention.

    Args:
        backbone (str): Backbone architecture ('r50' or 'r18')
            - 'r50': Input 1024 channels, projects to 768
            - 'r18': Input 256 channels, projects to 384

    Architecture:
        - Direct projection without spatial downsampling
        - BatchNorm + ReLU activation
        - Final projection to target dimension
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation="ReLU"):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Channel_Projector_All(nn.Module):
    """
    Channel projection layer for fusing all multi-scale features.

    Takes concatenated features from all scales and projects them to a single
    channel token for integration with the transformer architecture.

    Args:
        backbone (str): Backbone architecture ('r50' or 'r18')
            - 'r50': Input 4*768 channels, projects to 768
            - 'r18': Input 4*384 channels, projects to 384

    Architecture:
        - Global average pooling to reduce spatial dimensions
        - Linear projection to target dimension
        - BatchNorm + ReLU activation
    """

    def __init__(self, backbone="r50", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if backbone == "r50":
            self.nConvs = _make_nConv(3840, 768, 4)
        elif backbone == "r18":
            self.nConvs = _make_nConv(
                384, 768, 4
            )  # self.nConvs = _make_nConv(960, 768, 4)

    def forward(self, x):
        return torch.flatten(self.nConvs(x), start_dim=2)
