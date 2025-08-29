import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class MSINET(nn.Module):
    def __init__(self, input_size=(240, 320)):
        super().__init__()
        self._output = None
        self._mapping = {}

        # 明确设置数据格式
        self._data_format = "channels_first"
        self._channel_axis = 1
        self._dims_axis = (2, 3)
        self.input_size = input_size

        # 计算预期特征图尺寸
        self._compute_expected_sizes(input_size)

        # encoder参数列表
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 调整conv4和conv5的池化参数以确保尺寸匹配
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # 移除池化层以保持与layer3相同的尺寸
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU()
            # 移除池化层以保持与layer3相同的尺寸
        )

        # aspp参数列表 - 修改输入通道数以匹配encoder输出(256+512+512=1280)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.ReLU()
        )

        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # decoder参数列表
        self.convTr1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.convTr2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.convTr3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def _compute_expected_sizes(self, input_size):
        """计算各层预期输出尺寸"""
        h, w = input_size
        # conv1: 2倍下采样
        h, w = h // 2, w // 2
        # conv2: 2倍下采样
        h, w = h // 2, w // 2
        # conv3: 2倍下采样
        h, w = h // 2, w // 2
        # conv4和conv5: 保持与conv3相同的尺寸(无池化)
        self.expected_size = (h, w)

    def _encoder(self, images):
        # 定义均值张量 - 使用RGB顺序
        imagenet_mean = torch.tensor([123.675, 116.28, 103.53],
                                     dtype=torch.float32,
                                     device=config.PARAMS["device"])
        # 调整均值张量的形状为NCHW
        imagenet_mean = imagenet_mean.view(1, 3, 1, 1)

        # 如果输入是[0,1]范围，转换为[0,255]
        if images.max() <= 1.0:
            images = images * 255.0

        # 对图像进行均值归一化
        images = images - imagenet_mean

        # 卷积池化操作
        layer1 = self.conv1(images)
        layer2 = self.conv2(layer1)
        layer3 = self.conv3(layer2)
        layer4 = self.conv4(layer3)
        layer5 = self.conv5(layer4)

        # 检查特征图尺寸
        expected_h, expected_w = self.expected_size
        if layer3.shape[2:] != (expected_h, expected_w):
            print(f"Warning: layer3尺寸不匹配, 预期: {(expected_h, expected_w)}, 实际: {layer3.shape[2:]}")
        if layer4.shape[2:] != (expected_h, expected_w):
            print(f"Warning: layer4尺寸不匹配, 预期: {(expected_h, expected_w)}, 实际: {layer4.shape[2:]}")
        if layer5.shape[2:] != (expected_h, expected_w):
            print(f"Warning: layer5尺寸不匹配, 预期: {(expected_h, expected_w)}, 实际: {layer5.shape[2:]}")

        # 全连接层
        encoder_output = torch.cat([layer3, layer4, layer5],
                                   dim=self._channel_axis)

        return encoder_output

    def _aspp(self, feature):
        layer1 = self.branch1(feature)
        layer2 = self.branch2(feature)
        layer3 = self.branch3(feature)
        layer4 = self.branch4(feature)
        layer5 = self.branch5(feature)

        # 上采样全局平均池化分支到特征图尺寸
        layer5 = F.interpolate(layer5, size=feature.shape[2:],
                               mode='bilinear', align_corners=True)

        context = torch.cat([layer1, layer2, layer3, layer4, layer5],
                            dim=self._channel_axis)

        aspp_output = self.branch(context)

        return aspp_output

    def _decoder(self, feature):
        # 上采样2倍
        layer1 = F.interpolate(feature, scale_factor=2, mode='bilinear', align_corners=True)
        layer1 = self.convTr1(layer1)

        # 上采样2倍
        layer2 = F.interpolate(layer1, scale_factor=2, mode='bilinear', align_corners=True)
        layer2 = self.convTr2(layer2)

        # 上采样2倍
        layer3 = F.interpolate(layer2, scale_factor=2, mode='bilinear', align_corners=True)
        layer3 = self.convTr3(layer3)

        return layer3

    def _normalize(self, maps, eps=1e-7):
        dims = tuple(range(1, maps.dim()))  # (1,2,3) 对于 NCHW
        min_per_image = maps.amin(dims, keepdim=True)
        maps = maps - min_per_image

        max_per_image = maps.amax(dims, keepdim=True)
        maps = maps / (max_per_image + eps)

        return maps

    def forward(self, x):
        x = self._encoder(x)
        x = self._aspp(x)
        x = self._decoder(x)
        x = self._normalize(x)

        return x