from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from torch.nn import functional as F
from backbones.swin_transformer import SwinTransformer3D
from einops import rearrange
from utils.tensor_utils import Reduce
from losses.mse import mse_loss
from losses.r2 import r2_loss
from sklearn.metrics import r2_score
from losses.rmse import RMSE


class Exposure(pl.LightningModule):
    def __init__(self,
                 pretrained=None,
                 patch_size=(4,4,4),
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 frozen_stages=-1,
                 use_checkpoint=False, 
                 batch_size=8, 
                 multi_stage_training=False):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.multi_stage_training = multi_stage_training

        #self.train_rmse = RMSE()
        #self.train_mae = torchmetrics.MeanAbsoluteError()
        #self.train_r2 = torchmetrics.R2Score()

        #self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        #self.val_mae = torchmetrics.MeanAbsoluteError()
        #self.val_r2 = torchmetrics.R2Score()

        #self.test_rmse = torchmetrics.MeanSquaredError(squared=False)
        #self.test_mse = torchmetrics.MeanSquaredError()
        #self.test_mae = torchmetrics.MeanAbsoluteError()
        #self.test_r2 = torchmetrics.R2Score()

        self.swin_transformer = SwinTransformer3D(
            pretrained=pretrained,
            pretrained2d=True,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            frozen_stages=frozen_stages,
            use_checkpoint=use_checkpoint
        )
        
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(8*embed_dim),
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(4*embed_dim),
            nn.Linear(in_features=4*embed_dim, out_features=2*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            #nn.LayerNorm(2*embed_dim),
            #nn.Linear(in_features=2*embed_dim, out_features=8, bias=True),

            #Reduce()
        )

        self.neutral_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.happy_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.sad_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.contempt_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.anger_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.disgust_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.surprised_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        self.fear_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))

        #self.dropout = nn.Dropout(p=0.5)
        #self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1)) # output size ==> d' x h' x w'
        #self.ef_regressor = nn.Linear(in_features=8*embed_dim, out_features=1, bias=True)
        #self.reduce = Reduce()

    def forward_features(self, x):
        #print(f'1: {x.shape}')
        # (N, D, C, H, W)
        x = rearrange(x, 'n d c h w -> n c d h w')

        x = self.swin_transformer(x) # n c d h w ==> torch.Size([1, 768, 32, 4, 4])
        #x = x.mean(2) # n c h w
        #x = x.flatten(-2) # n c hxw
        #print(f'x: {x.shape}')
        #x = x.transpose(1, 2) # n hxw c

        return x

    def forward_head(self, x):
        # input ==> n c d h w
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = x.flatten(-2).mean(3) # n d c
        
        x = self.feature_processor(x) # n 8

        return {
            'neutral': self.neutral_classifier(x),
            'happy': self.happy_classifier(x),
            'sad': self.sad_classifier(x),
            'contempt': self.contempt_classifier(x),
            'anger': self.anger_classifier(x),
            'disgust': self.disgust_classifier(x),
            'suprised': self.surprised_classifier(x),
            'fear': self.fear_classifier(x)
        }

    def forward(self, x):
        x = self.forward_features(x) # n c d h w 
        #print(x)
        x = self.forward_head(x) # n 1

        return x

    def training_step(self, batch, batch_idx):
        print(f"batch shape: {batch['video'].shape}")
        prediction_labels = self(batch['video'])
       
        loss = F.cross_entropy(prediction_labels['neutral'].sigmoid(), batch['neutral'])
        loss += F.cross_entropy(prediction_labels['happy'].sigmoid(), batch['happy'])
        loss += F.cross_entropy(prediction_labels['sad'].sigmoid(), batch['sad'])
        loss += F.cross_entropy(prediction_labels['contempt'].sigmoid(), batch['contempt'])
        loss += F.cross_entropy(prediction_labels['anger'].sigmoid(), batch['anger'])
        loss += F.cross_entropy(prediction_labels['disgust'].sigmoid(), batch['disgust'])
        loss += F.cross_entropy(prediction_labels['surprised'].sigmoid(), batch['surprised'])
        loss += F.cross_entropy(prediction_labels['fear'].sigmoid(), batch['fear'])
        
       
        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        print(f"video shape: {batch['video'].shape}")
        prediction_labels = self(batch['video'])

        print(f"label shape: {batch['neutral'].shape}")
        print(f"prediction shape: {prediction_labels['neutral'].shape}")
       
        loss = F.cross_entropy(prediction_labels['neutral'].sigmoid(), batch['neutral'])
        loss += F.cross_entropy(prediction_labels['happy'].sigmoid(), batch['happy'])
        loss += F.cross_entropy(prediction_labels['sad'].sigmoid(), batch['sad'])
        loss += F.cross_entropy(prediction_labels['contempt'].sigmoid(), batch['contempt'])
        loss += F.cross_entropy(prediction_labels['anger'].sigmoid(), batch['anger'])
        loss += F.cross_entropy(prediction_labels['disgust'].sigmoid(), batch['disgust'])
        loss += F.cross_entropy(prediction_labels['surprised'].sigmoid(), batch['surprised'])
        loss += F.cross_entropy(prediction_labels['fear'].sigmoid(), batch['fear'])
        
       
        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        prediction_labels = self(batch['video'])
       
        loss = F.cross_entropy(prediction_labels['neutral'].sigmoid(), batch['neutral'])
        loss += F.cross_entropy(prediction_labels['happy'].sigmoid(), batch['happy'])
        loss += F.cross_entropy(prediction_labels['sad'].sigmoid(), batch['sad'])
        loss += F.cross_entropy(prediction_labels['contempt'].sigmoid(), batch['contempt'])
        loss += F.cross_entropy(prediction_labels['anger'].sigmoid(), batch['anger'])
        loss += F.cross_entropy(prediction_labels['disgust'].sigmoid(), batch['disgust'])
        loss += F.cross_entropy(prediction_labels['suprised'].sigmoid(), batch['suprised'])
        loss += F.cross_entropy(prediction_labels['fear'].sigmoid(), batch['fear'])
        
       
        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)

    def predict_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo) 

        loss = F.mse_loss(ef_pred, ejection)
        return {'filename': filename, 'EF': ef_label * 100., 'Pred': ef_pred * 100., 'loss': loss * 100.}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer], [lr_scheduler]