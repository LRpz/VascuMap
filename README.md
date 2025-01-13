# VascuMap
Label-free phenotyping of human microvessel networks using deep learning segmentation and graph-based analysis.
## Installation
Create and activate conda environment
```bash
conda create -n vascumap python=3.8
conda activate vascumap
```
Install PyTorch (adjust cuda version as needed)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install the package
```bash
pip install -e .
```

## Training
Train a new model using the following command:
```bash
python src/training.py \
    --images_dir_path "/path/to/training/images" \
    --masks_dir_path "/path/to/training/masks" \
    --model_dir_path "/path/to/save/model" \
    --model_architecture "Unet" \
    --encoder_architecture "mit_b5" \
    --input_channels 1 \
    --weights "imagenet" \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --fp16
```

Training Arguments
* ```--images_dir_path```: Directory containing training images
* ```--masks_dir_path```: Directory containing binary mask images
* ```--model_dir_path```: Directory to save model checkpoints
* ```--model_architecture```: Model architecture (default: 'Unet')
* ```--encoder_architecture```: Encoder backbone (default: 'mit_b5')
* ```--input_channels```: Number of input channels (default: 1)
* ```--weights```: Pretrained weights (default: 'imagenet')
* `--batch_size`: Batch size for training (default: 16)
* `--epochs`: Number of epochs to train the model (default: 200)
* `--learning_rate`: Learning rate for optimizer (default: 0.001)
* `--fp16`: Enable mixed precision training using FP16 (optional flag)

## Inference
Run inference on new images using a trained model:
```bash
python src/inference.py \
    --images_dir_path "/path/to/test/images" \
    --model_checkpoint_path "/path/to/model/best.pth" \
    --output_dir_path "/path/to/output" \
    --model_architecture "Unet" \
    --encoder_architecture "mit_b5" \
    --input_channels 1 \
    --test_time_augmentation \
    --device "cuda" \
    --resolution_scale 1.0 \
    --save_prob \
    --thresholding_method "hysteresis"
```

Inference Arguments
- `--images_dir_path`: Path to the directory containing images for inference (required)
- `--model_checkpoint_path`: Path to the model checkpoint (.ckpt) (required)
- `--output_dir_path`: Path to the directory where masks will be saved (required)
- `--model_architecture`: Model architecture to use for segmentation (default: 'Unet')
- `--encoder_architecture`: Encoder architecture to use within the model (default: 'mit_b5')
- `--input_channels`: Number of input channels for the model (default: 1)
- `--test_time_augmentation`: Enable test time augmentation (optional flag)
- `--device`: Device for inference, either 'cuda' or 'cpu' (default: 'cuda')
- `--resolution_scale`: Scale factor to apply to the image prior to inference (default: 1.0)
- `--save_prob`: Whether to save the probability map along with the mask (optional flag)
- `--thresholding_method`: Thresholding method applied to probability map (default: 'hysteresis')

## Network Analysis
```bash
python src/graph_metrics.py \
--masks_dir_path "/path/to/masks/directory" \
--file_suffix "_mask" \
--save_local_metrics \
--graph_visualization
```
Network Analysis Arguments
* `--masks_dir_path`: Path to the directory containing segmentation masks (required)
* `--file_suffix`: Suffix of the mask files (default: '_mask')
* `--save_local_metrics`: Save edge and node metrics for each mask (optional flag)
* `--graph_visualization`: Save graph visualization (optional flag)

## Data Format
* Input images should be in TIFF format
* For training, corresponding mask images should be binary (0 or 1) and in TIFF format
* Images and masks should have matching filenames
## License
This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ . See the `LICENSE` file for details.