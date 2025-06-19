import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config

class MSINET(nn.Module):
    def __init__(self):
        super().__init__()
        self._output = None
        self._mapping = {}

        self._channel_axis = 1
        self._dims_axis = (2, 3)

        # encoder参数列表
        self.enc_conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.enc_conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.enc_conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.enc_conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.enc_conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.enc_conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.enc_conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.enc_conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.enc_conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.enc_conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.enc_conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.enc_conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.enc_conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)

        # aspp参数列表
        self.aspp_conv1_1 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)
        self.aspp_conv1_2 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)
        self.aspp_conv1_3 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)
        self.aspp_conv1_4 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)
        self.aspp_conv1_5 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=1, padding=1)
        self.aspp_conv2 = nn.Conv2d(in_channels=1280, out_channels=256, kernel_size=3, padding=1)

        # decoder参数列表
        self.dec_conv1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.dec_conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dec_conv4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def _encoder(self, images):
        # 定义均值张量
        imagenet_mean = torch.tensor([103.939, 116.779, 123.68],
                                     dtype=torch.float32,
                                     device=config.PARAMS["device"])
        # 调整均值张量的形状
        imagenet_mean = imagenet_mean.view(1, 3, 1, 1)
        # 如果输入是[0,1]范围，转换为[0,255]
        if images.max() <= 1.0:
            images = images * 255.0
        # 对图像进行均值归一化
        images -= imagenet_mean
        # 调整维度顺序，可以使用 permute
        # images = images.permute(0, 3, 1, 2)

        # 卷积池化操作
        # 第一层   channels: 3 -> 64   height, weight: /= 2
        layer01 = self.enc_conv1_1(images)
        layer01 = self.relu(layer01)
        layer02 = self.enc_conv1_2(layer01)
        layer02 = self.relu(layer02)
        layer03 = self.max_pool1(layer02)

        # 第二层   channels: 64 -> 128   height, weight: /= 2
        layer04 = self.enc_conv2_1(layer03)
        layer04 = self.relu(layer04)
        layer05 = self.enc_conv2_2(layer04)
        layer05 = self.relu(layer05)
        layer06 = self.max_pool2(layer05)

        # 第三层   channels: 128 -> 256   height, weight: /= 2
        layer07 = self.enc_conv3_1(layer06)
        layer07 = self.relu(layer07)
        layer08 = self.enc_conv3_2(layer07)
        layer08 = self.relu(layer08)
        layer09 = self.enc_conv3_3(layer08)
        layer09 = self.relu(layer09)
        layer10 = self.max_pool3(layer09)

        # 第四层   channels: 256 -> 512
        layer11 = self.enc_conv4_1(layer10)
        layer11 = self.relu(layer11)
        layer12 = self.enc_conv4_2(layer11)
        layer12 = self.relu(layer12)
        layer13 = self.enc_conv4_3(layer12)
        layer13 = self.relu(layer13)
        # layer14 = self.max_pool4(layer13)
        layer14 = layer13

        # 第五层   channels: 512 -> 512
        layer15 = self.enc_conv5_1(layer14)
        layer15 = self.relu(layer15)
        layer16 = self.enc_conv5_2(layer15)
        layer16 = self.relu(layer16)
        layer17 = self.enc_conv5_3(layer16)
        layer17 = self.relu(layer17)
        # layer18 = self.max_pool5(layer17)
        layer18 = layer17

        # 在_encoder方法中拼接前添加
        assert layer10.shape[2:] == (30, 40), f"layer10尺寸错误: {layer10.shape}"
        assert layer14.shape[2:] == (30, 40), f"layer14尺寸错误: {layer14.shape}"
        assert layer18.shape[2:] == (30, 40), f"layer18尺寸错误: {layer18.shape}"

        # 全连接层
        encoder_output = torch.cat([layer10, layer14, layer18],
                                   dim=self._channel_axis)

        return encoder_output

    def _aspp(self, feature):
        branch1 = self.aspp_conv1_1(feature)
        branch1 = self.relu(branch1)

        branch2 = self.aspp_conv1_2(feature)
        branch2 = self.relu(branch2)

        branch3 = self.aspp_conv1_3(feature)
        branch3 = self.relu(branch3)

        branch4 = self.aspp_conv1_4(feature)
        branch4 = self.relu(branch4)

        # 将height, weight维度压缩为1，类似那个自适应池化的用法
        branch5 = torch.mean(feature, dim=self._dims_axis, keepdim=True)
        branch5 = self.aspp_conv1_5(branch5)
        branch5 = self.relu(branch5)

        shape = feature.shape
        # 计算目标大小
        factor = 1
        target_height = shape[self._dims_axis[0]] * factor
        target_width = shape[self._dims_axis[1]] * factor
        # 双线性插值上采样
        branch5 = F.interpolate(
            branch5,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )

        context = torch.cat([branch1, branch2, branch3, branch4, branch5],
                                   dim=self._channel_axis)

        aspp_output = self.aspp_conv2(context)
        aspp_output = self.relu(aspp_output)

        return aspp_output

    def _decoder(self, feature):
        # 第一层
        layer1 = feature
        shape = layer1.shape
        factor = 2
        target_height = shape[self._dims_axis[0]] * factor
        target_width = shape[self._dims_axis[1]] * factor
        layer1 = F.interpolate(
            layer1,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        layer2 = self.dec_conv1(layer1)
        layer2 = self.relu(layer2)

        # 第二层
        layer3 = layer2
        shape = layer3.shape
        factor = 2
        target_height = shape[self._dims_axis[0]] * factor
        target_width = shape[self._dims_axis[1]] * factor
        layer3 = F.interpolate(
            layer3,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        layer4 = self.dec_conv2(layer3)
        layer4 = self.relu(layer4)

        # 第三层
        layer5 = layer4
        shape = layer5.shape
        factor = 2
        target_height = shape[self._dims_axis[0]] * factor
        target_width = shape[self._dims_axis[1]] * factor
        layer5 = F.interpolate(
            layer5,
            size=(target_height, target_width),
            mode='bilinear',
            align_corners=False
        )
        layer6 = self.dec_conv3(layer5)
        layer6 = self.relu(layer6)

        decoder_output = self.dec_conv4(layer6)

        return decoder_output

    def _normalize(self, map):
        min_per_image = torch.min(map.view(map.size(0), -1), dim=1, keepdim=True)[0]
        min_per_image = min_per_image.view(map.size(0), 1, 1, 1)
        map -= min_per_image

        max_per_image = torch.max(map.view(map.size(0), -1), dim=1, keepdim=True)[0]
        max_per_image = max_per_image.view(map.size(0), 1, 1, 1)
        map = map / (1e-7 + max_per_image)

        return map

    def forward(self, x):
        x = self._encoder(x)
        x = self._aspp(x)
        x = self._decoder(x)
        x = self._normalize(x)

        return x