import argparse
import os
import torch
import torchvision.transforms as transforms
from torch.utils.hipify.hipify_python import preprocessor
import numpy as np

import config
import data
import utils
from model import MSINET
from loss import KLDivLossWrapper

def define_paths(current_path, args):
    if os.path.isfile(args.path):
        data_path = args.path
    else:
        data_path = os.path.join(args.path, "")

    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"

    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path
    }

    return paths



def train_model(dataset, paths, device):
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.PARAMS["batch_size"],
        shuffle=True,
        num_workers=2
    )

    # 初始化模型、损失函数和优化器
    model = MSINET()
    criterion = KLDivLossWrapper()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.PARAMS["learning_rate"])

    # 计算训练数据相关信息
    n_train_data = len(dataset)
    n_train_batches = len(dataloader)

    # 初始化历史和进度条工具
    history = utils.History(n_train_batches,
                            "salicon",
                            paths["history"],
                            device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)

    # 将模型和损失函数移到指定设备
    model.to(device)

    # 训练循环
    for epoch in range(config.PARAMS["n_epochs"]):
        model.train()

        # 训练阶段
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images)
            if targets.dim() == 3:
                targets = targets.unsqueeze(1)
            loss = criterion(targets, outputs)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新历史记录和进度条
            history.update_train_step(loss.item())
            progbar.update_train_step(batch_idx)

        # 保存历史记录并更新进度条
        history.save_history()
        mean_train_loss = history.get_mean_train_error(reset=True)
        progbar.update_valid_step()
        progbar.write_summary(mean_train_loss)

    # 保存最终模型
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_path, f"msinet_salicon_model.pth")
    torch.save(model.state_dict(), model_path)

    print("训练完成！")



def test_model(dataset, paths, device):
    print("11111")



def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    # 选择训练或者测试
    phases_list = ["train", "test"]

    # 默认salicon数据集
    datasets_list = ["salicon", "mit1003", "cat2000",
                     "dutomron", "pascals", "osie", "fiwi"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    # 可选参数，数据集选取，不加入该参数则默认salicon数据集
    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")

    # 可选参数，自定义数据集目录，不加入该参数则默认当前目录的data目录
    parser.add_argument("-p", "--path", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    args = parser.parse_args()

    # 创建目录存放实验结果
    paths = define_paths(current_path, args)

    # 初始化数据集
    train_dataset = data.SaliconDataset(
        data_root='data',
        mode='train'
    )

    if args.phase == "train":
        train_model(train_dataset, paths, config.PARAMS["device"])
    elif args.phase == "test":
        test_model(args.data, paths, config.PARAMS["device"])



if __name__ == "__main__":
    main()
