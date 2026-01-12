import torch
import numpy as np

from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Boxes, Masks, Probs

def jsonify_ultra_result(result: Boxes | Masks | Probs) -> dict:
    '''Handle each item in ultralytics.engine.results.Result instance to make 
    it json compatible because numpy.ndarray and torch.Tensor is not uncompatible
    with json.
    
    Args:
        result (Boxes | Masks | Probs): Any one of the attributes in
            ultralytics.engine.results.Result class
    
    Returns:
        res (dict): A dictionary or any data type or data structure that is json
            compatible.
    
    '''
    if isinstance(result, (Boxes, Masks, Probs)):
        attributes = [x for x in result.__class__.__dict__.keys() if not x.startswith("__")]
        res = {}
        for k in attributes:
            value = getattr(result, k)
            res[k] = value.tolist() if isinstance(value, (torch.Tensor, np.ndarray)) else value
    elif isinstance(result, (torch.Tensor, np.ndarray)):
        res = result.tolist()
    elif isinstance(result, tuple):
        res = list(result)
    else:
        res = result
    return res

class BasePreprocessor:
    """A preprocessor callable to transform image and reduce file upload size."""
    def __init__(
        self, 
        imgsz: int | tuple = (640, 640),
        rect: bool = True,
        model_stride: int = 32,
        device: str | torch.device = 'cpu',
        ):
        """
        Args:
            imgsz (int | tuple): model specified input image size during model training.
            rect (bool): If enabled, minimally pads the shorter side of the 
                image until it's divisible by stride to improve inference speed.
                If disabled, pads the image to a square during inference.
            model_stride (int): The stride calculated by comparing the spatially 
                smallest feature map relative to the model input image size.
                Generally image of any size will be preprocessed to a specific
                input size for model training and the value is usually (640, 640)
                [H,W] and the spatiall y smallest feature map is of (20,20)[H,W].
                Hence, stride = 640/20 = 32.
            device (str | torch.device): device to send the tensor to.
        """
        self.imgsz = imgsz
        self.rect = rect
        self.model_stride = model_stride
        self.device = device
    
    def __call__(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.rect,
            # and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model_stride,
        )
        return [letterbox(image=x) for x in im]
