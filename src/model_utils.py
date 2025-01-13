import math
import segmentation_models_pytorch as smp
import torch
from catalyst import utils
from catalyst.contrib.nn import DiceLoss, RAdam
from catalyst.dl import (CriterionCallback, DiceCallback, IouCallback,
                         MetricAggregationCallback, SupervisedRunner)

def adapt_input_conv(in_chans, conv_weight):
    """
    This function adapts the input channels of a convolutional layer's weights based on the number of input channels 
    provided. It handles cases where the input channels are 1 (grayscale), 3 (RGB), or other values. 
    The function ensures that the weight tensor is in the correct format for the given number of input channels and 
    adjusts the weights accordingly.

    Args:
    in_chans (int): The number of input channels.
    conv_weight (torch.Tensor): The convolutional layer's weights.

    Returns:
    torch.Tensor: The adapted convolutional layer's weights.
    """
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)

    return conv_weight

def adapt_input_model(model):
    """
    Adapts first layer to take a specified number of input channels.
    
     Args:
        model: The segmentation model to be adapted.

     Returns:
        The adapted segmentation model.
     """
    # Adapt first layer to take 1 channel as input - timm approach = sum weights
    new_weights = adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
    model.encoder.patch_embed1.proj = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))

    with torch.no_grad():
        model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
    
    return model

def build_model(model_str, encoder_str, in_channels=1, encoder_weights=None):
   """
   Provide a segmentation model with specified architecture and encoder.

   Args:
       model_str (str): Name of the segmentation model architecture.
       encoder_str (str): Name of the encoder used in the model.
       encoder_weights (str): Pretrained weights for the specified encoder.
       in_channels (int): Number of input channels.

   Returns:
       A PyTorch model instance with specified architecture and encoder.
   """

   # Create a dictionary mapping from string to actual SMP function
   ARCHITECTURES = {
      'Unet': smp.Unet,
      'Unet++': smp.UnetPlusPlus,
      'MAnet': smp.MAnet,
      'Linknet': smp.Linknet,
      'FPN': smp.FPN,
      'PSPNet': smp.PSPNet,
      'PAN': smp.PAN,
      'DeepLabV3': smp.DeepLabV3,
      'DeepLabV3+': smp.DeepLabV3Plus
   }

   try:
        # Get constructor method for desired architecture
        constructor = ARCHITECTURES[model_str]
       
        if encoder_str.startswith('mit') and in_channels == 1:
            model = constructor(encoder_name=encoder_str, classes=1, in_channels=3, encoder_weights=encoder_weights)
            model = adapt_input_model(model)
        else: 
            model = constructor(encoder_name=encoder_str, classes=1, in_channels=in_channels, encoder_weights=encoder_weights)

        return model
   
   except KeyError:
       raise ValueError(f'Model type "{model_str}" not understood. Valid options are: {list(ARCHITECTURES.keys())}')


def compile_runner(learning_rate, model_str, encoder_str, encoder_weights, loss_weights, n_epochs, in_channels):

    criterion = {
        "dice": DiceLoss(),
        "bce": torch.nn.BCEWithLogitsLoss() 
    }

    device = utils.get_device()
    print(f"Using device: {device}")

    callbacks = [

        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),

        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum", 
            metrics={
                "loss_dice": loss_weights[0],
                "loss_bce": loss_weights[1]
                },
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask", threshold=0.5)

     ]

    model = build_model(model_str, encoder_str, in_channels, encoder_weights)
    model_params = utils.process_model_params(model)

    optimizer = RAdam(model_params, lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(n_epochs/2), gamma=0.5, last_epoch=- 1, verbose=False)
    runner = SupervisedRunner(device=device, input_key="image", input_target_key="mask")

    return model, runner, criterion, optimizer, scheduler, callbacks