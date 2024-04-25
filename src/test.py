import torch
import os
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from data.histology import HistologyValidation

def load_model(version_number: Union[int, str], epoch: Union[int, str], step: int, logs_dir: str = os.path.join('logs', DEFAULT_LOGS_DIR)) -> pl.LightningModule:
    model = src.framework.Model.load_from_checkpoint(
        os.path.join(logs_dir, f'version_{version_number}', 'checkpoints', f'epoch={epoch}-step={step+1}.ckpt')
    )
    return model

model = load_model(16, 9, 3850)
model.eval()

validation_dataset = HistologyValidation(directory='/content/drive/MyDrive/HistologyNet/labelled')
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

predictions = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions.append(outputs)
