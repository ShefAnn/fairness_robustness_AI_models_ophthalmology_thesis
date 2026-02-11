#!/usr/bin/env python3
#The big part of code in main_finetune.py, extract_f_p_RETFound_noise.py, extract_f_p_RETFound.py is inspired by: https://github.com/msayhan/ICL-Ophthalmology-Public

# =========================
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm

# =========================
import numpy as np
import torch

# =========================
import models_vit as models
from util.datasets import build_dataset

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Inference + filename, emb, pred, label "
    )
    # ---- Core training
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU (effective batch size = batch_size * accum_iter * #gpus)")

    # ---- Model parameters
    parser.add_argument("--model", default="vit_large_patch16", type=str, metavar="MODEL",
                        help="Model entry in models_vit.py")
    parser.add_argument("--input_size", default=256, type=int, help="Image size")

    # ---- Finetuning & adaptation
    parser.add_argument("--task", default="", type=str, help="Task name for logging/output grouping")

    # ---- Dataset & paths
    parser.add_argument("--data_path", default="./data/", type=str)
    parser.add_argument("--nb_classes", default=8, type=int)
    parser.add_argument("--output_dir", default="./output_dir")

    # ---- Runtime
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", default="", help="Resume full state (optimizer, scaler, etc.)")
    parser.add_argument("--num_workers", default=10, type=int)

    # ---- NOISE !!!
    parser.add_argument("--noise", default='small')

    return parser.parse_args()


def add_noise (images, noise_level, seed=42):
    if noise_level == 'small':
        std = 0.02
    elif noise_level == 'medium':
        std = 0.05
    elif noise_level == 'large':
        std = 0.15
    else:
        std = 0
        
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(images) * std

    images = images + noise
    return torch.clamp(images, 0.0, 1.0)

# =========================
# Extract
# =========================

def extract (model, data, dataloader, device, noise):
    out_data = OrderedDict()
    out_data['file_names'] = []
    out_data['labels'] = []
    out_data['features'] = []
    out_data['logits'] = []
    out_data['predictions'] = []

    model.eval()
    start_index = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if noise is not None:
                images = add_noise(images, noise_level=noise)

            features = model.forward_features(images).flatten(1)
            logits = model.head(features)
            predictions = torch.argmax(logits, dim=1)

            images_set = images.size(0)

            for i in range(images_set):
                path = data.samples[start_index][0]
                out_data['file_names'].append(os.path.basename(path))
                start_index += 1
                    
            out_data['labels'].append(labels.cpu().numpy())
            out_data['features'].append(features.cpu().numpy())
            out_data['logits'].append(logits.cpu().numpy())
            out_data['predictions'].append(predictions.cpu().numpy())

    for r in ['labels', 'features', 'logits', 'predictions']:
        out_data[r] = np.concatenate(out_data[r], axis=0)
    out_data['file_names'] = np.array(out_data['file_names'])
    
    return out_data


# =========================
# Main
# =========================
def main():
    args = get_args_parser()

    device = torch.device(args.device)
    print(f'Device : {device}')

    dataset_test  = build_dataset(is_train="test",  args=args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size, shuffle=False, pin_memory=True, 
        num_workers=args.num_workers, drop_last=False,
    )
    model = models.__dict__[args.model](
            img_size=args.input_size,
            num_classes=args.nb_classes,
            global_pool=True,
        )

    ckpt =  torch.load(args.resume, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)

    out = extract(model, dataset_test, data_loader_test, device, noise=args.noise)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.task}_ev+path+pred+emb_noise{args.noise}.npz")
    np.savez_compressed(save_path, **out)
    print(f"SAVED to {save_path}")
    for k, v in out.items():
        print(f"{k}:{v.shape if hasattr(v, 'shape') else len(v)}")


if __name__ == "__main__":
    main()


    
