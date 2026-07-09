import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import (
    BasicConv3d,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
)


class GLConv(nn.Module):
    """GaitGL-style global-local convolution that preserves feature shape."""

    def __init__(
        self,
        in_channels,
        out_channels,
        halving=3,
        fm_sign=False,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
        bias=False,
    ):
        super().__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias
        )
        self.local_conv3d = BasicConv3d(
            in_channels, out_channels, kernel_size, stride, padding, bias
        )

    def forward(self, x):
        global_feat = self.global_conv3d(x)
        if self.halving == 0:
            local_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = max(int(h // 2**self.halving), 1)
            local_feat = torch.cat(
                [self.local_conv3d(part) for part in x.split(split_size, 3)], 3
            )

        if self.fm_sign:
            return F.leaky_relu(torch.cat([global_feat, local_feat], dim=3))
        return F.leaky_relu(global_feat) + F.leaky_relu(local_feat)


class SMPLTransform(nn.Module):
    """Frame-wise SMPL vector to 16x16 residual transform matrix."""

    def __init__(self, in_dim=85, matrix_size=16):
        super().__init__()
        self.matrix_size = matrix_size
        hidden_dim = matrix_size * matrix_size
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)

    def forward(self, smpls):
        n, s, d = smpls.size()
        sps = smpls.reshape(-1, d)
        sps = F.relu(self.bn1(self.fc1(sps)))
        sps = F.relu(self.bn2(self.dropout2(self.fc2(sps))))
        sps = F.relu(self.bn3(self.dropout3(self.fc3(sps))))
        sps = sps.reshape(n, 1, s, self.matrix_size, self.matrix_size)

        identity = torch.eye(
            self.matrix_size, device=sps.device, dtype=sps.dtype
        ).view(1, 1, 1, self.matrix_size, self.matrix_size)
        return sps + identity


class SMPLGaitGL(BaseModel):
    """SMPLGait with a conservative GaitGL residual refinement branch.

    The main path is the original SMPLGait feature transformation. GaitGL-style
    global-local convolutions refine the transformed feature map with a small
    learnable residual weight, so the model starts close to SMPLGait instead of
    letting the silhouette branch overwrite the SMPL-conditioned representation.
    """

    def build_network(self, model_cfg):
        self.Backbone = SetBlockWrapper(self.get_backbone(model_cfg["backbone_cfg"]))
        self.FCs = SeparateFCs(**model_cfg["SeparateFCs"])
        self.BNNecks = SeparateBNNecks(**model_cfg["SeparateBNNecks"])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg["bin_num"])

        self.smpl_transform = SMPLTransform(
            in_dim=model_cfg.get("smpl_dim", 85),
            matrix_size=model_cfg.get("smpl_matrix_size", 16),
        )

        gl_cfg = model_cfg.get("gaitgl_cfg", {})
        feat_c = model_cfg["SeparateFCs"]["in_channels"]
        temporal_kernel = gl_cfg.get("temporal_kernel_size", 1)
        temporal_padding = temporal_kernel // 2
        halving = gl_cfg.get("halving", 3)

        self.GLRefine = nn.Sequential(
            GLConv(
                feat_c,
                feat_c,
                halving=halving,
                fm_sign=False,
                kernel_size=(temporal_kernel, 3, 3),
                padding=(temporal_padding, 1, 1),
            ),
            nn.BatchNorm3d(feat_c),
            nn.LeakyReLU(inplace=True),
            GLConv(
                feat_c,
                feat_c,
                halving=halving,
                fm_sign=False,
                kernel_size=(temporal_kernel, 3, 3),
                padding=(temporal_padding, 1, 1),
            ),
            nn.BatchNorm3d(feat_c),
        )
        self.fusion_logit = nn.Parameter(
            torch.tensor(float(gl_cfg.get("fusion_logit_init", -4.0)))
        )

    def _square_16(self, outs):
        n, c, s, h, w = outs.size()
        if w < h:
            outs = F.pad(outs, (0, h - w, 0, 0))
        elif w > h:
            outs = F.pad(outs, (0, 0, 0, w - h))

        if outs.size(3) != 16 or outs.size(4) != 16:
            outs = F.interpolate(
                outs, size=(s, 16, 16), mode="trilinear", align_corners=False
            )
        return outs

    def _apply_smpl_transform(self, outs, sps_trans):
        outs = self._square_16(outs)
        n, c, s, h, w = outs.size()

        if sps_trans.size(2) != s:
            sps_trans = F.interpolate(
                sps_trans,
                size=(s, 16, 16),
                mode="trilinear",
                align_corners=False,
            )

        outs = outs.reshape(n * c * s, h, w)
        sps = sps_trans.repeat(1, c, 1, 1, 1).reshape(n * c * s, 16, 16)
        outs = torch.bmm(outs, sps)
        return outs.reshape(n, c, s, 16, 16)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]
        smpls = ipts[1]

        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)

        del ipts

        sps_trans = self.smpl_transform(smpls)
        outs = self.Backbone(sils)
        outs_trans = self._apply_smpl_transform(outs, sps_trans)

        gl_residual = self.GLRefine(outs_trans)
        fusion_weight = torch.sigmoid(self.fusion_logit)
        outs_trans = outs_trans + fusion_weight * gl_residual

        outs_trans = self.TP(outs_trans, seqL, options={"dim": 2})[0]
        feat = self.HPP(outs_trans)
        embed_1 = self.FCs(feat)
        _, logits = self.BNNecks(embed_1)

        n, _, s, h, w = sils.size()
        return {
            "training_feat": {
                "triplet": {"embeddings": embed_1, "labels": labs},
                "softmax": {"logits": logits, "labels": labs},
            },
            "visual_summary": {"image/sils": sils.reshape(n * s, 1, h, w)},
            "inference_feat": {"embeddings": embed_1},
        }
