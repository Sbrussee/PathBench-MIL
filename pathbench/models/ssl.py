import lightly

from lightly.loss import BarlowTwinsLoss, DINOLoss, NTXentLoss
from lightly.models.modules import (BarlowTwinsProjectionHead,
                                    DINOProjectionHead,
                                    SimCLRProjectionHead,
                                    masked_autoencoder)
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.mae_transform import MAETransform
from lightly.utils.scheduler import cosine_schedule
from lightly.models import utils
from lightly.data.dataset import LightlyDataset
from lightly.transforms.mmcr_transform import MMCRTransform
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)

from torch import nn
import torch
from torchvision import models
import copy
import argparse
import os

import pytorch_lightning as pl

argparse.add_argument('--method', choices=["MAE", "BarlowTwins", "SimCLR", "DINO"], type=str, help='SSL method to use')
argparse.add_argument('--backbone', choices=["resnet18", "vit16", "vit32"]. type=str, help='Backbone model to use')
argparse.add_argument('--path_to_train', type=str, help='Path to training images')
argparse.add_argument('--path_to_val', default=None, type=str, help='Path to validation')
argparse.add_argument('--ssl_model_name', type=str, help='Name of the SSL model')

args = argparse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Classifier(pl.LightingModule):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(512, num_classes)

        #Freeze backbone
        deactivate_requires_grad(self.backbone)

        self.criterion = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log("train_loss_fc", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.forward(x)
        y_hat = torch.nn.functional.softmax(logits, dim=1)
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).sum().item()
        self.validation_step_outpus.append((num, correct))
        return num, correct
    

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            total_num = 0
            total_correct = 0
            for num, correct in self.validation_step_outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc_fc", acc)
            self.validation_step_outputs.clear()
    

class BarlowTwins(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead()
        self.criterion = BarlowTwinsLoss()


    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
    
    def training_step(self, batch, batch_id):
        (x0, x1) = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.05)
        return optim
    
class DINO(pl.LightningModule):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
        input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=2048, warmup_teacher_temp_epochs=5)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_id):
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_head, self.teacher_head, momentum)
        update_momentum(self.student_backbone, self.teacher_backbone, momentum)
        views = batch[0]
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        t_out = [self.forward_teacher(view) for view in global_views]
        s_out = [self.forward(view) for view in views]
        loss = self.criterion(t_out, s_out, epoch=self.current_epoch)
        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        return optim
    

class SimCLR(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 2048, 2048)
        self.criterion = NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_id):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    

class MAE(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = 0.75
        self.patch_size = 16
        self.sequence_length = 49
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 512))
        self.decoder = masked_autoencoder.MAEDecoder(
        seq_length=49,
        num_layers=1,
        num_heads=16,
        embed_input_dim=512,
        hidden_dim=512,
        mlp_dim=512 * 4,
        out_dim=16**2 * 3,
        dropout=0,
        attention_dropout=0
        )

        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
        self.mask_token, (batch_size, self.sequence_length)
        )

        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        x_decoded = self.decoder.decode(x_masked)

        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_id):
        views = batch[0]
        images = views[0]
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
        size=(batch_size, self.sequence_length),
        mask_ratio=self.mask_ratio,
        device=images.device
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        patches = utils.patchify(images, self.patch_size)

        target = utils.get_at_index(patches, idx_mask-1)

        loss = self.criterion(x_pred, target)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1.5e-4)
        return optim
    

def train_ssl_model(method, backbone_model, ssl_model_name, path_to_train, path_to_val=None):
    if backbone_model == 'resnet18':
        resnet  = models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        input_dim = 512

    elif backbone_model == 'vit16':
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False)
        input_dim = backbone.embed_dim

    elif backbone_model == 'vit32':
        backbone = models.vit_b_32(pretrained=False)
        input_dim = 512

    if method == 'BarlowTwins':
        transform = SimCLRTransform()
        model = BarlowTwins()

    elif method == 'DINO':
        transform = DINOTransform()
        model = DINO(input_dim=input_dim)

    elif method == "SimCLR":
        transform = SimCLRTransform()
        model = SimCLR()

    elif method == 'MAE':
        transform = MAETransform()
        model = MAE()

    train_dataset = LightlyDataset(path_to_train, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True, num_workers=8)

    val_dataset = None
    val_loader = None
    if path_to_val:
        val_dataset = LightlyDataset(path_to_val, transform=transform)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=8)

    trainer = pl.Trainer(max_epochs=200, gpus=1 if torch.cuda.is_available() else 0, accelerator="ddp")
    
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='min'
    )

    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader, callbacks=[early_stop_callback])

    os.makedirs("train_checkpoints", exist_ok=True)
    #Save the trained model
    torch.save(model.state_dict(), f"train_checkpoints/{ssl_model_name}.pth")

if __name__ == "__main__":
    train_ssl_model(args.method, args.backbone, args.ssl_model_name,
                    args.path_to_train, args.path_to_val)
