import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SaliconDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None, target_transform=None):
        super(SaliconDataset, self).__init__()

        # 构建完整路径
        self.stimuli_dir = os.path.join(data_root, 'salicon', 'stimuli', mode)
        self.saliency_dir = os.path.join(data_root, 'salicon', 'saliency', mode)

        # 验证目录存在性
        if not os.path.exists(self.stimuli_dir):
            raise FileNotFoundError(f"Stimuli目录不存在: {self.stimuli_dir}")
        if not os.path.exists(self.saliency_dir):
            raise FileNotFoundError(f"Saliency目录不存在: {self.saliency_dir}")

        # 获取图像文件名列表 (确保匹配)
        self.image_files = []
        # 遍历stimuli目录中的所有.jpg文件
        for file in os.listdir(self.stimuli_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                # 检查对应的saliency文件是否存在
                base_name = os.path.splitext(file)[0]  # 去掉扩展名
                saliency_file = f"{base_name}.png"  # 创建对应的.png文件名
                saliency_path = os.path.join(self.saliency_dir, saliency_file)

                if os.path.exists(saliency_path):
                    self.image_files.append(file)
                else:
                    print(f"警告：缺失显著图文件 {saliency_file}，跳过图像 {file}")

        # 如果没有找到匹配的文件对
        if not self.image_files:
            raise RuntimeError(f"在 {self.stimuli_dir} 和 {self.saliency_dir} 中没有找到匹配的图像和显著图文件对")

        # 设置变换
        self.transform = transform
        self.target_transform = target_transform

        # 如果没有提供变换，使用默认变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((240, 320)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        if self.target_transform is None:
            self.target_transform = transforms.Compose([
                transforms.Resize((240, 320)),
                transforms.ToTensor()
            ])

    def __len__(self):
        """返回数据集大小"""
        return len(self.image_files)

    def __getitem__(self, idx):
        """获取单个样本"""
        img_name = self.image_files[idx]

        # 构建完整文件路径
        img_path = os.path.join(self.stimuli_dir, img_name)

        # 创建对应的显著图文件名（将.jpg替换为.png）
        base_name = os.path.splitext(img_name)[0]
        saliency_name = f"{base_name}.png"
        saliency_path = os.path.join(self.saliency_dir, saliency_name)

        # 打开图像
        image = Image.open(img_path).convert('RGB')
        saliency = Image.open(saliency_path).convert('L')  # 转换为灰度图

        # 应用变换
        image = self.transform(image)
        saliency = self.target_transform(saliency)

        return image, saliency