from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelPrecision
from torchmetrics.classification import MultilabelRecall
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
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.confusion_matrix = torchmetrics.ConfusionMatrix(8, multilabel=True)
        self.prec = MultilabelPrecision(num_labels=8)
        self.recall = MultilabelRecall(num_labels=8)
        self.f1_score = MultilabelF1Score(num_labels=8)
        self.accuracy = MultilabelAccuracy(num_labels=8)

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
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(8*embed_dim),
            nn.Linear(in_features=8*embed_dim, out_features=4*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(4*embed_dim),
            nn.Linear(in_features=4*embed_dim, out_features=2*embed_dim, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(2*embed_dim),
            nn.Linear(in_features=2*embed_dim, out_features=8)

            #Reduce()
        )

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        #self.neutral_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.happy_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.sad_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.contempt_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.anger_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.disgust_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.surprised_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))
        #self.fear_classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=2*embed_dim, out_features=2))

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
        #x = rearrange(x, 'n c d h w -> n d c h w')
        #x = x.flatten(-2).mean(3).mean(1) # n d c
    
        x = self.avg_pool(x) # n c 1 1 1
        x = x.squeeze() # n c
        x = self.classifier(x) # n 8

        #return {
        #    'neutral': self.neutral_classifier(x),
        #    'happy': self.happy_classifier(x),
        #    'sad': self.sad_classifier(x),
        #    'contempt': self.contempt_classifier(x),
        #    'anger': self.anger_classifier(x),
        #    'disgust': self.disgust_classifier(x),
        #    'surprised': self.surprised_classifier(x),
        #    'fear': self.fear_classifier(x)
        #}

        return x

    def forward(self, x):
        x = self.forward_features(x) # n c d h w 
        #print(x)
        x = self.forward_head(x) # n 1

        return x

    def training_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        print(f"pred: {prediction_label}")
        print(f"label: {batch['label']}")
        
        loss = self.loss_fn(prediction_label, batch['label'])
        acc = self.accuracy((prediction_label.sigmoid() > 0.5).long(), batch['label'])

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'train': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'train': acc}, global_step=self.current_epoch) 

        return loss

    def validation_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        loss = self.loss_fn(prediction_label, batch['label'])
        acc = self.accuracy((prediction_label.sigmoid() > 0.5).long(), batch['label'])

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.log("val_acc", acc, on_epoch=True, on_step=True, prog_bar=True, logger=False)

        self.logger.experiment.add_scalars('loss', {'val': loss}, global_step=self.current_epoch) 
        self.logger.experiment.add_scalars('acc', {'val': acc}, global_step=self.current_epoch) 
        
    def test_step(self, batch, batch_idx):
        prediction_label = self(batch['video'])
        pred_label_sigmoid = prediction_label.sigmoid()
        #print(f"predict: {prediction_label}")
        #print(f"label: {batch['label']}")

        loss = self.loss_fn(prediction_label, batch['label'])
        cm = self.confusion_matrix(pred_label_sigmoid, batch['label'].long())
        self.accuracy(pred_label_sigmoid, batch['label'])
        self.f1_score(pred_label_sigmoid, batch['label'])
        self.prec(pred_label_sigmoid, batch['label'])
        self.recall(pred_label_sigmoid, batch['label'])

        cm_mean = cm.float().mean(0)
        
        #true negatives for class i in M(0,0)
        #false positives for class i in M(0,1)
        #false negatives for class i in M(1,0)
        #true positives for class i in M(1,1)

        self.log('test_loss', loss, on_epoch=True)
        self.log('accuracy', self.accuracy, on_epoch=True)

        self.log('TN', cm_mean[0,0], on_epoch=True)
        self.log('FP', cm_mean[0,1], on_epoch=True)
        self.log('FN', cm_mean[1,0], on_epoch=True)
        self.log('TP', cm_mean[1,1], on_epoch=True)

        self.log('precision', self.prec, on_epoch=True)
        self.log('recall', self.recall, on_epoch=True)
        self.log('f1_score', self.f1_score, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        return self.shared_step(batch, 'predict')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)
        #optimizer = torch.optim.AdamW(self.parameters())
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)

        return [optimizer], [lr_scheduler]