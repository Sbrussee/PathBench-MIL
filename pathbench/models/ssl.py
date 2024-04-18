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

from torch import nn
import torch
from torchvision import models
import copy
import argparse

argparse.add_argument('--method', choices=["MAE", "BarlowTwins", "SimCLR", "DINO"], type=str, help='SSL method to use')
argparse.add_argument('--backbone', choices=["resnet18", "vit16", "vit32"]. type=str, help='Backbone model to use')
argparse.add_argument('--path_to_images', type=str, help='Path to images')
argparse.add_argument('--ssl_model_name', type=str, help='Name of the SSL model')

args = argparse.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class DINO(nn.Module):
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

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class MAE(nn.Module):
    def __init__(self, vit):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1,1,decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
        seq_length = vit.seq_length,
        num_layers=1,
        num_heads=16,
        embed_input_dim = vit.hidden_dim,
        hidden_dim=decoder_dim,
        mlp_dim=decoder_dim * 4,
        out_dim = vit.patch_size**2 * 3,
        dropout=0,
        attention_dropout=0
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images, idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        batch_size = x.encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
        self.mask_token, (batch_size, self.sequence_length)
        )
        x_decoded = self.decoder.decode(x_masked)

        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
        size = (batch_size, self.sequence_length),
        mask_ratio = self.mask_ratio,
        device=images.device)

        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask-1)
        return x_pred, target



def train_ssl_model(method, backbone_model, path_to_images, ssl_model_name):
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
        train_barlow_twins(path_to_images, backbone, ssl_model_name)

    elif method == 'DINO':
        train_dino(path_to_images, backbone, input_dim, ssl_model_name)

    elif method == "SimCLR":
        train_simclr(path_to_images, backbone, ssl_model_name)

    elif method == 'MAE':
        train_mae(path_to_images, backbone, ssl_model_name)

def train_barlow_twins(path_to_images, backbone, ssl_model_name):
    #Initialize Barlow Twins
    model = BarlowTwins(backbone)
    model.to(device)

    transform = SimCLRTransform()
    dataloader = torch.utils.data.DataLoader(
    LightlyDataset(path_to_images, transform=transform),
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8
    )

    criterion = BarlowTwinsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    print("Training Barlow Twins model...")
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0), model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}, loss: {avg_loss:.5f}")

    #Save the backbone
    pretrained_backbone = model.backbone
    torch.save(pretrained_backbone.state_dict(), ssl_model_name)


def train_dino(path_to_images, backbone, input_dim, ssl_model_name):
    model = DINO(backbone, input_dim)
    model.to(device)

    transform = DINOTransform()

    dataloader = torch.utils.data.DataLoader(
    LightlyDataset(path_to_images, transform=transform),
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=8
    )

    criterion = DINOLoss(
    output_dim=2048,
    warmup_teacher_temp_epochs=5
    )

    critertion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training DINO model...")
    for epoch in range(50):
        total_loss = 0
        momentum_val = cosine_schedule(epoch, 50, 0.996, 1)
        for batch in dataloader:
            views = batch[0]
            update_momentum(model.student_backbone, model.teacher_backbone)
            update_momentum(model.student_head, model.teacher_head)
            views = [view.to(device) for view in views]
            global_views = views[:2]
            teacher_out = [model.forward_teacher(view) for view in global_views]
            student_out = [model.forward(view) for view in views]
            loss = criterion(teacher_out, student_out, epoch=epoch)
            total_loss += loss.detach()
            loss.backward()
            #Cancel gradients of student head
            model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}, loss: {avg_loss:.5f}")

    #Save the backbone
    pretrained_backbone = model.backbone
    torch.save(pretrained_backbone.state_dict(), ssl_model_name)

def train_simclr(path_to_images, backbone, ssl_model_name):
    model = SimCLR(backbone)
    model.to(device)

    transform = SimCLRTransform(input_size=32)

    dataloader = torch.utils.data.DataLoader(
    LightlyDataset(path_to_images, transform=transform),
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8
    )

    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

    print("Training SimCLR model...")
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            x0, x1 = batch[0]
            x0, x1 = x0.to(device), x1.to(device)
            z0, z1 = model(x0), model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}, loss: {avg_loss:.5f}")

    #Save the backbone
    pretrained_backbone = model.backbone
    torch.save(pretrained_backbone.state_dict(), ssl_model_name)

def train_mae(path_to_images, backbone, ssl_model_name):
    model = MAE(backbone)
    model.to(device)
    transform = MAETransform()

    dataloader = torch.utils.data.DataLoader(
    LightlyDataset(path_to_images, transform=transform),
    batch_size=256,
    shuffle=True,
    drop_last=True,
    num_workers=8
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

    print("Training MAE model...")
    for epoch in range(50):
        total_loss = 0
        for batch in dataloader:
            views = batch[0]
            images = views[0].to(device)
            predictions, targets = model(images)
            loss = criterion(predictions, targets)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch}, loss: {avg_loss:.5f}")

    #Save the backbone
    pretrained_backbone = model.backbone
    torch.save(pretrained_backbone.state_dict(), ssl_model_name)

if __name__ == "__main__":
    train_ssl_model(args.method, args.backbone, args.path_to_images, args.ssl_model_name)
