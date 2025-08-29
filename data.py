import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import zipfile
import gdown


def download_salicon(data_path):
    """Downloads the SALICON dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.

    .. seealso:: The code for downloading files from google drive is based
                 on the solution provided at [https://bit.ly/2JSVgMQ].
    """

    print(">> Downloading SALICON dataset...", end="", flush=True)

    default_path = data_path + "salicon/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    ids = ["1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5",
           "1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj",
           "1PnO7szbdub1559LfjYHMy65EDC4VhJC8"]

    urls = ["https://drive.google.com/uc?id=" +
            i + "&export=download" for i in ids]

    save_paths = [default_path, fixations_path, saliency_path]

    for count, url in enumerate(urls):
        gdown.download(url, data_path + "tmp.zip", quiet=True)

        with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
            for file in zip_ref.namelist():
                if "test" not in file:
                    zip_ref.extract(file, save_paths[count])

    os.rename(default_path + "images", default_path + "stimuli")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


class SaliconDataset(Dataset):
    def __init__(self, data_root, mode='train', transform=None, target_transform=None):
        super(SaliconDataset, self).__init__()

        # 构建完整路径
        self.stimuli_dir = os.path.join(data_root, 'salicon', 'stimulisalicon', 'stimuli', mode)
        self.saliency_dir = os.path.join(data_root, 'salicon', 'stimulisalicon', 'saliency', mode)

        # 如果子目录都不存在，那么就说明数据集没有下载，那么我们创建目录并下载数据集
        if not os.path.exists(self.stimuli_dir):
            download_salicon(os.path.join(data_root, 'salicon', 'stimuli'))

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


class TestDataset(Dataset):
    def __init__(self, data_root, transform=None):
        super(TestDataset, self).__init__()
        # 测试图片所在的目录：data_root/sence/origin/test
        self.test_dir = os.path.join(data_root, 'sence', 'origin', 'test')
        if not os.path.exists(self.test_dir):
            raise FileNotFoundError(f"Test directory not found: {self.test_dir}")
        self.image_files = [f for f in os.listdir(self.test_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not self.image_files:
            raise RuntimeError(f"No images found in {self.test_dir}")

        # 设置变换
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((240, 320)),  # 与训练时保持一致
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        # 保存原始尺寸，以便后续恢复
        original_size = image.size  # (width, height)
        # 应用变换
        image_tensor = self.transform(image)
        return image_tensor, img_name, original_size