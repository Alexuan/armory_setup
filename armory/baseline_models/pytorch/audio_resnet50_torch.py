"""
Resnet for speech commands classification.
"""

from typing import Optional

from art.estimators.classification import PyTorchClassifier
import torch
from torchvision.models import resnet50
import torchaudio

class ShapeMatch(torch.nn.Module):
   
    """Utility class to match spectrogram shape to expected input shape of ResNet50"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.repeat((1,3,1,1)) # Three channels required, duplicate along channel dimension
        return x

def make_audio_resnet(**kwargs) -> torch.nn.Sequential:

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=256, win_length=255, hop_length=128, power=1)

    resnet = resnet50(weights=None, num_classes=12)

    model = torch.nn.Sequential(
                        spectrogram, 
                        ShapeMatch(),
                        resnet)

    return model


def get_art_model(
    model_kwargs: dict, wrapper_kwargs: dict, weights_path: Optional[str] = None
):

    __import__('ipdb').set_trace()
    if weights_path:
        raise ValueError(
            "This model is implemented for poisoning and does not (yet) load saved weights."
        )

    model = make_audio_resnet(**model_kwargs)

    loss = torch.nn.CrossEntropyLoss()

    opt = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    art_classifier = PyTorchClassifier(
        model=model,
        loss=loss,
        optimizer=opt,
        nb_classes=12,
        input_shape=(16000,),
        **wrapper_kwargs,
    )

    return art_classifier