import argparse
import os
from pathlib import Path

import numpy as np
import tifffile as tif
import torch
import tqdm
import ttach as tta
from catalyst import utils
from catalyst.dl import SupervisedRunner
from skimage.transform import resize

from dataset import load_data
from model_utils import build_model
from transforms import hysteresis_thresholding, single_level_thresholding

def run_inference(images_path, model_path, output_path, model_str, encoder_str,
    in_channels, do_tta, device, scaling, save_prob, thresholding_method):

    def predict_im(im, scaling, runner, device):
    
        def nearest_multiple_of_32(x):
            return np.ceil(x / 32) * 32

        original_shape = im['image'].shape[-2:]  # Assuming the shape is [B, C, H, W]

        inference_height = nearest_multiple_of_32(original_shape[0] * scaling) 
        inference_width = nearest_multiple_of_32(original_shape[1] * scaling)

        image_np = im['image'].squeeze().cpu().numpy() 
        resized_image = resize(image_np, (inference_height, inference_width), anti_aliasing=True) 
        resized_image = torch.from_numpy(resized_image[None, None, ...]).to(device) 
        
        pred = runner.predict_batch({'image': resized_image})

        logits_np = torch.sigmoid(pred['logits']).squeeze().cpu().numpy()  
        return resize(logits_np, original_shape, anti_aliasing=True)  

    # Load data without masks (inference mode)
    loader = load_data(images_path_str=images_path, batch_size=1)

    # Build and load the trained model
    model = build_model(model_str, encoder_str, in_channels)
    
    checkpoint = utils.load_checkpoint(model_path)
    utils.unpack_checkpoint(checkpoint=checkpoint, model=model)

    # Move the model to the specified device and set it to evaluation mode
    model.to(device)
    model.eval()

    if do_tta:
        tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.flip_transform(), merge_mode='mean')
        runner = SupervisedRunner(model=tta_model, input_key="image", input_target_key="mask", device=device)
    else:
        runner = SupervisedRunner(model=model, input_key="image", input_target_key="mask", device=device)

    os.makedirs(output_path, exist_ok=True)

    for im in tqdm.tqdm(iter(loader)):
        filename = im['filename']

        if not filename[0].endswith('_prob.tif') or not filename[0].endswith('_mask.tif'):
        
            try:
                prob = predict_im(im, scaling, runner, device)
                
                if save_prob:
                    tif.imwrite(os.path.join(output_path, filename[0].replace('.tif', '_prob.tif')), np.array(prob).astype(np.float32))

                if thresholding_method == 'hysteresis':
                    mask = hysteresis_thresholding(prob)

                elif thresholding_method == 'single_level':
                    mask = single_level_thresholding(prob)

                tif.imwrite(os.path.join(output_path, filename[0].replace('.tif', '_mask.tif')), np.array(mask).astype(np.float32))

            except RuntimeError:
                print(f'RuntimeError with {filename}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a segmentation model on provided images.")
    
    parser.add_argument("--images_dir_path", type=str, required=True,
                        help="Path to the directory containing images for inference.")
    parser.add_argument("--model_checkpoint_path", type=str, required=True,
                        help="Path to the model checkpoint (.ckpt)")
    parser.add_argument("--output_dir_path", type=str, required=True,
                        help="Path to the directory where masks will be saved.")
    parser.add_argument("--model_architecture", type=str, default="Unet",
                        help="Model architecture to use for segmentation. Default is 'Unet'.")
    parser.add_argument("--encoder_architecture", type=str, default="mit_b5",
                        help="Encoder architecture to use within the segmentation model. Default is 'mit_b5'.")
    parser.add_argument("--input_channels", type=int, default=1,
                        help="Number of input channels for the model. Default is 1.")
    parser.add_argument("--test_time_augmentation", action='store_true',
                        help="Enable test time augmentation.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device for inference, either 'cuda' or 'cpu'. Default is 'cuda'.")
    parser.add_argument("--resolution_scale", type=float, default=1,
                        help="Scale factor to apply to the image prior to inference, closer to 0 means lower resolution. Default is 1")
    parser.add_argument("--save_prob", action='store_true',
                        help="Whether to save the probability map along with the mask.")    
    parser.add_argument("--thresholding_method", type=str, default='hysteresis',
                        help="Thresholding method applied to the probability map. Default is 'hysteresis'")   

    args = parser.parse_args()

    run_inference(
        images_path=args.images_dir_path,
        model_path=args.model_dir_path,
        output_path=args.output_dir_path,
        model_str=args.model_architecture,
        encoder_str=args.encoder_architecture,
        in_channels=args.input_channels,
        do_tta=args.test_time_augmentation,
        device=args.device, 
        scaling=args.resolution_scale,
        save_prob=args.save_prob,
        thresholding_method=args.thresholding_method
    )