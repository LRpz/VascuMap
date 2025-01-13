import argparse
from model_utils import build_model, adapt_input_model, compile_runner
from dataset import load_data

def train(images_path, masks_path, model_path, model_str, encoder_str, 
          weights, in_channels, batch_size, epochs, learning_rate, fp16):
          
    # Load data and prepare loaders
    loaders = load_data(images_path, masks_path, batch_size)

    model, runner, criterion, optimizer, scheduler, callbacks = compile_runner(
        learning_rate=learning_rate,
        model_str=model_str,
        encoder_str=encoder_str,
        encoder_weights=weights,
        loss_weights=[1.0, 1.0],  # Modify as needed for your application
        n_epochs=epochs,
        in_channels=in_channels  # Update if using different number of input channels
    )

    # Train the model using Catalyst's SupervisedRunner
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=callbacks,
        logdir=model_path,
        num_epochs=epochs,
        main_metric="iou",  # Change metric if needed
        minimize_metric=False,
        verbose=True,
        fp16=fp16,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a segmentation model on provided images and masks.")
    
    parser.add_argument("--images_dir_path", type=str, required=True,
                        help="Path to the directory containing training images.")
    parser.add_argument("--masks_dir_path", type=str, required=True,
                        help="Path to the directory containing corresponding mask images.")
    parser.add_argument("--model_dir_path", type=str, required=True,
                        help="Path to the directory where model checkpoints and logs will be saved.")
    parser.add_argument("--model_architecture", type=str, default="Unet",
                        help="Model architecture to use for segmentation. Default is 'Unet'.")
    parser.add_argument("--encoder_architecture", type=str, default="mit_b5",
                        help="Encoder architecture to use within the segmentation model. Default is 'mit_b5'.")
    parser.add_argument("--input_channels", type=int, default=1,
                        help="Number of input channels for the model. Default is 1.")
    parser.add_argument("--weights", type=str, default="imagenet",
                        help="Pretrained weights to initialize the encoder. Default is 'imagenet'.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training. Default is 16.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of epochs to train the model. Default is 200.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer during training. Default is 0.001 (1e-3).")
    parser.add_argument("--fp16", action='store_true',
                        help="Enable mixed precision training using FP16. Reduces memory usage and can improve performance.")

    args = parser.parse_args()

    train(
      images_path=args.images_dir_path,
      masks_path=args.masks_dir_path,
      model_path=args.model_dir_path,
      model_str=args.model_architecture,
      encoder_str=args.encoder_architecture,
      in_channels=args.input_channels,
      weights=args.weights,
      batch_size=args.batch_size,
      epochs=args.epochs,
      learning_rate=args.learning_rate, 
      fp16=args.fp16
   )