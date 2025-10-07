import argparse
import os
import time  # Although not used in eval, good practice to keep if needed later
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import py_sod_metrics  # Assuming this is a custom module for SOD metrics
from dataset.sod_dataset import getSODDataloader
from model.moesod import MoESOD
from utils.AvgMeter import AvgMeter  # Assuming this is a custom module for averaging


# Ensure albumentations is not updated to avoid potential compatibility issues
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


def load_model_checkpoint(net: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Loads the model checkpoint and moves the model to the specified device.

    Args:
        net (torch.nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto.

    Returns:
        torch.nn.Module: The model with loaded weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    print(f"Loading model checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    # Handle potential 'module.' prefix from DistributedDataParallel saving
    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."):
            state_dict[k[7:]] = v
        else:
            state_dict[k] = v

    load_state = net.load_state_dict(state_dict, strict=False)
    print(f"Model state after loading checkpoint: {load_state}")
    return net.to(device)


def evaluate_model(
    net: torch.nn.Module, dataloader: torch.utils.data.DataLoader, output_path: str, device: torch.device
):
    """
    Evaluates the model on the dataset and saves predictions and metrics.

    Args:
        net (torch.nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        output_path (str): Directory to save predicted saliency maps and evaluation results.
        device (torch.device): The device to perform inference on.
    """
    net.eval()

    # Initialize metrics
    mae_meter = AvgMeter()  # For tracking average MAE during iteration
    mae_metric = py_sod_metrics.MAE()
    wfm_metric = py_sod_metrics.WeightedFmeasure()
    sm_metric = py_sod_metrics.Smeasure()
    em_metric = py_sod_metrics.Emeasure()
    fm_metric = py_sod_metrics.Fmeasure()

    sigmoid = torch.nn.Sigmoid()

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    print(f"Starting evaluation. Predictions will be saved to: {output_path}")

    with torch.no_grad():
        # Use tqdm for progress bar, only showing on rank 0 if distributed (though this script seems single GPU)
        data_generator = tqdm(dataloader, ncols=100, desc="Evaluating")
        for data in data_generator:
            img = data["img"].to(device).to(torch.float32)
            original_label = data["ori_mask"].to(device)  # Ground truth mask
            mask_name = data["mask_name"][0]  # Get the name of the current image/mask

            # Forward pass
            # moe_loss is ignored during evaluation
            predicted_map, _ = net(img)

            # Post-process the output
            predicted_map = sigmoid(predicted_map)
            predicted_map = F.interpolate(
                predicted_map,
                [original_label.shape[1], original_label.shape[2]],
                mode="bilinear",
                align_corners=False,
            )

            # Calculate MAE for iteration tracking
            img_mae = torch.mean(torch.abs(predicted_map - original_label))
            mae_meter.update(img_mae.item(), n=1)

            # Prepare prediction for metrics and saving
            # Scale to 0-255, convert to uint8
            pred_uint8 = (predicted_map * 255).squeeze().cpu().data.numpy().astype(np.uint8)
            # Ground truth is already expected as float (0-1) by py_sod_metrics usually,
            # but if it's uint8, it needs to be handled. Assuming original_label is 0-1 float.

            # Update SOD metrics
            fm_metric.step(pred=pred_uint8, gt=original_label)
            wfm_metric.step(pred=pred_uint8, gt=original_label)
            sm_metric.step(pred=pred_uint8, gt=original_label)
            em_metric.step(pred=pred_uint8, gt=original_label)
            mae_metric.step(pred=pred_uint8, gt=original_label)

            # Save the predicted saliency map
            # Convert to BGR for saving with cv2 if needed, though grayscale is standard for SOD
            # pred_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)
            output_filepath = os.path.join(output_path, mask_name)
            cv2.imwrite(output_filepath, pred_uint8)  # Saving as grayscale

    # --- Final Metric Calculation and Reporting ---
    print(f"\nAverage MAE (from iterator): {mae_meter.avg:.5f}")

    # Get aggregated results from metrics
    fm_results = fm_metric.get_results()
    wfm_results = wfm_metric.get_results()
    sm_results = sm_metric.get_results()
    em_results = em_metric.get_results()
    mae_results = mae_metric.get_results()

    # Extract relevant metrics
    # The exact keys might vary based on the py_sod_metrics implementation.
    # Commonly used metrics are:
    # 'mae': mean absolute error
    # 'sm': structural similarity
    # 'em': enhanced alignment measure
    # 'wfm': weighted f-measure
    # 'fm': f-measure (adp - adaptive threshold, mf - max f-measure)

    mae_final = mae_results["mae"]
    max_fm_final = fm_results["mf"]  # Max F-measure
    adp_fm_final = fm_results["fm"]["adp"]  # Adaptive F-measure
    sm_final = sm_results["sm"]
    em_final = em_results["em"]
    wfm_final = wfm_results["wfm"]

    print("\n--- Evaluation Results ---")
    print(f"MAE: {mae_final:.5f}")
    print(f"MaxFm: {max_fm_final:.5f}")
    print(f"AdpFm: {adp_fm_final:.5f}")
    print(f"SM: {sm_final:.5f}")
    print(f"EM: {em_final:.5f}")
    print(f"WFM: {wfm_final:.5f}")

    # Store results for Excel
    results_dict = {
        "Metric": ["MAE", "MaxFm", "AdpFm", "SM", "EM", "WFM"],
        "Value": [
            mae_final,
            max_fm_final,
            adp_fm_final,
            sm_final,
            em_final,
            wfm_final,
        ],
    }

    df_results = pd.DataFrame(results_dict)
    excel_filename = os.path.join(output_path, "evaluation_results.xlsx")
    df_results.to_excel(excel_filename, index=False)
    print(f"Results saved to Excel file: {excel_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saliency Object Detection Model Evaluation Script.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./output/model_epoch136.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset path",
        help="Path to the evaluation dataset (e.g., DUTS test set).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output_saliencymap/model_epoch136",
        help="Directory to save predicted saliency maps and evaluation results.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=384,
        help="Input image size for the model.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="ID of the GPU to use for evaluation. Set to -1 for CPU.",
    )

    args = parser.parse_args()

    # --- Device Setup ---
    if args.gpu_id >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)} on device {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU for evaluation.")
        args.gpu_id = -1  # Ensure gpu_id is -1 if CPU is used

    # --- Model Initialization ---
    # Initialize the model architecture
    model = MoESOD(args.img_size)

    # Load the trained weights
    model = load_model_checkpoint(model, args.checkpoint, device)

    # --- DataLoader Initialization ---
    print(f"Loading evaluation dataset from: {args.data_path}")
    eval_dataloader = getSODDataloader(
        data_path=args.data_path,
        batch_size=1,  # Batch size is usually 1 for evaluation to process each image individually
        num_workers=args.num_workers,
        split="test",  # Assuming 'test' split is for evaluation
        img_size=args.img_size,
    )

    # --- Run Evaluation ---
    evaluate_model(model, eval_dataloader, args.output_path, device)

    print("\nEvaluation script finished.")