import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from tqdm import tqdm

from dataset.sod_dataset import getSODDataloader
from model.moesod import MoESOD
from utils.AvgMeter import AvgMeter
from utils.loss import LossFunc

# Ensure albumentations is not updated to avoid potential compatibility issues
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def reshapePos(pos_embed, img_size):
    """
    Reshapes the positional embedding to match the target image size.

    Args:
        pos_embed (torch.Tensor): The original positional embedding.
        img_size (int): The target image size.

    Returns:
        torch.Tensor: The reshaped positional embedding.
    """
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # Resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(
            pos_embed, (token_size, token_size), mode="bilinear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed


def reshapeRel(k, rel_pos_params, img_size):
    """
    Reshapes relative positional parameters for specific layers.

    Args:
        k (str): The key of the parameter.
        rel_pos_params (torch.Tensor): The relative positional parameters.
        img_size (int): The target image size.

    Returns:
        torch.Tensor: The reshaped relative positional parameters.
    """
    if not ("2" in k or "5" in k or "8" in k or "11" in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(
        rel_pos_params, (token_size * 2 - 1, w), mode="bilinear", align_corners=False
    )
    return rel_pos_params[0, 0, ...]


def load(net, ckpt, img_size):
    """
    Loads pre-trained weights into the network, reshaping positional embeddings if necessary.

    Args:
        net (torch.nn.Module): The network to load weights into.
        ckpt (str): Path to the checkpoint file.
        img_size (int): The image size used for training.

    Returns:
        OrderedDict: The state dictionary that was loaded.
    """
    ckpt = torch.load(ckpt, map_location="cpu")
    dict_ordered = OrderedDict()
    for k, v in ckpt.items():
        # Rename pe_layer
        if "pe_layer" in k:
            dict_ordered[k[15:]] = v
            continue
        # Reshape positional embeddings
        if "pos_embed" in k:
            dict_ordered[k] = reshapePos(v, img_size)
            continue
        # Reshape relative positional embeddings
        if "rel_pos" in k:
            dict_ordered[k] = reshapeRel(k, v, img_size)
        # Handle image encoder weights
        elif "image_encoder" in k:
            if "neck" in k:
                # Add original final neck layer to layers 3, 6, and 9 with same initialization
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict_ordered[new_key] = v
            else:
                dict_ordered[k] = v
        # Handle mask decoder weights
        if "mask_decoder.transformer" in k:
            dict_ordered[k] = v
        if "mask_decoder.iou_token" in k:
            dict_ordered[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict_ordered[k] = v
    state = net.load_state_dict(dict_ordered, strict=False)
    return state


def trainer(net, dataloader, loss_func, optimizer, local_rank):
    """
    Performs one epoch of training.

    Args:
        net (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        loss_func (callable): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        local_rank (int): The local rank of the current process.
    """
    net.train()
    loss_avg = AvgMeter()
    mae_avg = AvgMeter()
    if local_rank == 0:
        print("Starting training...")

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    if local_rank == 0:
        data_generator = tqdm(dataloader)
    else:
        data_generator = dataloader

    for data in data_generator:
        img = data["img"].cuda().to(torch.float32)
        label = data["mask"].cuda().unsqueeze(1)

        optimizer.zero_grad()

        coarse_out, moe_loss = net(img)

        coarse_out = sigmoid(coarse_out)

        loss_coarse = loss_func(coarse_out, label)
        loss = loss_coarse + moe_loss

        loss_avg.update(loss.item(), img.shape[0])

        img_mae = torch.mean(torch.abs(coarse_out - label))
        mae_avg.update(img_mae.item(), n=img.shape[0])

        loss.backward()
        optimizer.step()

    temp_cost = time.time() - start
    print(
        "local_rank:{}, loss:{:.4f}, mae:{:.4f}, cost_time:{:.0f}m:{:.0f}s".format(
            local_rank, loss_avg.avg, mae_avg.avg, temp_cost // 60, temp_cost % 60
        )
    )


def valer(net, dataloader, local_rank):
    """
    Performs validation on the validation set.

    Args:
        net (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        local_rank (int): The local rank of the current process.

    Returns:
        float: The mean absolute error on the validation set.
    """
    net.eval()
    if local_rank == 0:
        print("Starting validation...")

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    mae_avg = AvgMeter()
    with torch.no_grad():
        if local_rank == 0:
            data_generator = tqdm(dataloader)
        else:
            data_generator = dataloader
        for data in data_generator:
            img = data["img"].cuda().to(torch.float32)
            ori_label = data["ori_mask"].cuda()

            coarse_out, moe_loss = net(img)

            coarse_out = sigmoid(coarse_out)
            coarse_out = torch.nn.functional.interpolate(
                coarse_out,
                [ori_label.shape[1], ori_label.shape[2]],
                mode="bilinear",
                align_corners=False,
            )

            # Since float values are converted to int when saving the mask,
            # multiple decimals will be lost, which may result in minor deviations from the evaluation code.
            img_mae = torch.mean(torch.abs(coarse_out - ori_label))
            mae_avg.update(img_mae.item(), n=1)

    temp_cost = time.time() - start
    print(
        "local_rank:{}, val_mae:{:.4f}, cost_time:{:.0f}m:{:.0f}s".format(
            local_rank, mae_avg.avg, temp_cost // 60, temp_cost % 60
        )
    )

    return mae_avg.avg


def ddp_setup(rank, world_size):
    """
    Sets up distributed data parallel training.

    Args:
        rank (int): Unique identifier of each process.
        world_size (int): Total number of processes.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(local_rank, num_gpus, args):
    """
    Main training function.

    Args:
        local_rank (int): The local rank of the current process.
        num_gpus (int): The total number of GPUs available.
        args (argparse.Namespace): Command-line arguments.
    """
    # Initialize distributed training
    dist.init_process_group(
        backend="nccl", init_method=args.init_method, world_size=num_gpus, rank=local_rank
    )
    torch.cuda.set_device(local_rank)

    # Initialize the model
    net = MoESOD(args.img_size).cuda()

    # Load pre-trained weights if specified
    if args.resume == "":
        state = load(net, args.sam_ckpt, args.img_size)
        if local_rank == 0:
            print(state)

    # Get data loaders
    trainLoader = getSODDataloader(
        args.data_path, args.batch_size, args.num_workers, "train", img_size=args.img_size
    )
    valLoader = getSODDataloader(
        args.data_path,
        1,
        args.num_workers,
        "test",
        local_rank,
        img_size=args.img_size,
        max_rank=dist.get_world_size(),
    )

    loss_func = LossFunc

    # Define different learning rates for different layers
    hungry_param = []
    full_param = []
    for k, v in net.named_parameters():
        if "image_encoder" in k:
            if "adapter" in k:
                hungry_param.append(v)
            elif "neck" in k:
                full_param.append(v)
            else:
                v.requires_grad = False
        else:
            if "transformer" in k:
                full_param.append(v)
            elif "iou" in k:
                full_param.append(v)
            elif "mask_tokens" in k:
                hungry_param.append(v)
            elif "pe_layer" in k:
                full_param.append(v)
            elif "output_upscaling" in k:
                full_param.append(v)
            else:
                hungry_param.append(v)

    optimizer = torch.optim.AdamW(
        [
            {"params": hungry_param, "lr": args.lr_rate},
            {"params": full_param, "lr": args.lr_rate * 0.1},
        ],
        weight_decay=1e-5,
    )

    for name, param in net.named_parameters():
        print(f"Parameter name: {name} - {'Frozen' if not param.requires_grad else 'Trainable'}")

    best_mae = 1
    best_epoch = 0

    start_epoch = 1

    # Resume training from checkpoint
    if args.resume != "":
        start_epoch = int(args.resume.split("/")[-1].split(".")[0][11:]) + 1
        resume_dict = torch.load(args.resume, map_location="cpu")
        optimizer.load_state_dict(resume_dict["optimizer"])
        net_dict = OrderedDict()
        for k, v in resume_dict["model"].items():
            if "module." in k:
                net_dict[k[7:]] = v
            else:
                net_dict[k] = v
        state = net.load_state_dict(net_dict)
        if local_rank == 0:
            print(state)

    # Wrap the model with DistributedDataParallel
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    for i in range(start_epoch, args.epochs + 1):
        # Learning rate scheduling
        if i <= args.warmup_period:
            _lr = args.lr_rate * i / args.warmup_period
        else:
            _lr = args.lr_rate * (0.98 ** (i - args.warmup_period))

        t = 0
        for param_group in optimizer.param_groups:
            if t == 0:
                param_group["lr"] = _lr
            else:
                param_group["lr"] = _lr * 0.1
            t += 1

        if local_rank == 0:
            print("Epoch {} starting...".format(i))

        # Train and validate
        trainer(net, trainLoader, loss_func, optimizer, local_rank=local_rank)
        if i > 0:
            local_mae = valer(net, valLoader, local_rank=local_rank)

            # Average results from multi-GPU inference
            sum_result = torch.tensor(local_mae).cuda()
            dist.reduce(sum_result, dst=0, op=dist.ReduceOp.SUM)

            if local_rank == 0:
                mae = sum_result.item() / dist.get_world_size()
                print("Current MAE: {:.4f}".format(mae))

                # Save the best result
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = i
                    print(
                        "Saving epoch {} in {}".format(
                            i, "{}/model_epoch{}.pth".format(args.save_dir, i)
                        )
                    )
                    if not os.path.exists(args.save_dir):
                        os.makedirs(args.save_dir)
                    torch.save(
                        {"model": net.state_dict(), "optimizer": optimizer.state_dict()},
                        "{}/model_epoch{}.pth".format(args.save_dir, i),
                    )
                print("Best epoch:{}, MAE:{}".format(best_epoch, best_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--init_method",
        default="tcp://127.0.0.1:33519",
        type=str,
        help="init_method for distributed training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_period", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr_rate", type=float, default=0.0005)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset path",
        help="the postfix must to be DUTS",
    )
    parser.add_argument("--sam_ckpt", type=str, default="./sam_vit_b_01ec64.pth")
    parser.add_argument("--save_dir", type=str, default="./output/")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="If you need to train from begining, make sure 'resume' is empty str. If you want to continue training, set it to the previous checkpoint.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()

    # Set visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    num_gpus = torch.cuda.device_count()
    mp.spawn(main, args=(num_gpus, args), nprocs=num_gpus)