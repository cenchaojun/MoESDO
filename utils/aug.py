import os
import shutil
import random

# 定义源路径和目标路径
# image_dir = '/data2/CCJ/ccj/dataset/RGBSOD/DUTS-TE/images'
# mask_dir = '/data2/CCJ/ccj/dataset/RGBSOD/DUTS-TE/masks'
# image_dir = '/data2/CCJ/ccj/dataset/RGBSOD/DUT-O/images'
# mask_dir = '/data2/CCJ/ccj/dataset/RGBSOD/DUT-O/masks'
# image_dir = '/data2/CCJ/ccj/dataset/RGBSOD/ECSSD/images'
# mask_dir = '/data2/CCJ/ccj/dataset/RGBSOD/ECSSD/masks'
datasets = ["DUTS-TE", "PASCAL-S", "DUT-O", "ECSSD"]
for dataset in datasets:
    image_dir = '/data2/CCJ/ccj/dataset/RGBSOD/{0}/images'.format(dataset)
    mask_dir = '/data2/CCJ/ccj/dataset/RGBSOD/{0}/masks'.format(dataset)
    target_image_dir = '/data2/CCJ/ccj/dataset/SOD2A/DUTS/train/images'  # 目标图像文件夹路径
    target_mask_dir = '/data2/CCJ/ccj/dataset/SOD2A/DUTS/train/masks'    # 目标掩模文件夹路径

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_mask_dir, exist_ok=True)

    # 获取图片文件列表
    images = os.listdir(image_dir)

    # 计算抽取数量（20%）
    num_samples = max(1, int(len(images) * 0.02))  # 至少抽取1张

    # 随机选择一定数量的图片
    random_samples = random.sample(images, num_samples)

    # 复制选中的图片和对应的掩模到目标文件夹
    for image in random_samples:
        # 复制图片
        image_path = os.path.join(image_dir, image)
        shutil.copy(image_path, target_image_dir)

        # 生成对应的掩模名称
        mask_name = image.replace('.jpg', '.png')  # 假设掩模文件名为相同的基本名
        mask_path = os.path.join(mask_dir, mask_name)

        # 复制对应的掩模
        shutil.copy(mask_path, target_mask_dir)

    print(f"已随机抽取 {num_samples} 张图片及其掩模到 {target_image_dir} 和 {target_mask_dir}")
