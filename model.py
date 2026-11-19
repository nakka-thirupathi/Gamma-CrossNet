import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

def _as_nchw(x):
    # timm feature backbones may return NHWC for transformer features.
    if x.ndim == 4 and x.shape[1] < x.shape[-1] and x.shape[2] < x.shape[-1]:
        # likely NHWC when last dim is channel count
        return x.permute(0, 3, 1, 2).contiguous()
    return x

class GammaGuidedModule(nn.Module):
    """
    Operationalization of Supplementary S1.

    Source-defined:
      PDF(x|a,b,mu) = (x-mu)^(a-1) exp(-(x-mu)/b) /
                      (Gamma(a) b^a), x > mu.
    The supplement leaves T (positive transform), E (parameter estimator),
    and G (feature-generation function) unspecified. This implementation
    makes these explicit, replaceable choices:
      T(F)=softplus(F)+eps;
      mu=max(0, min(T)-eps), per sample/channel;
      alpha,beta from method-of-moments on T-mu;
      G(F)=F + Conv1x1(log-PDF(T)).
    This is a reproducible baseline, NOT a claim that these missing choices
    were the authors' original hidden implementation.
    """
    def __init__(self, channels, eps=1e-6, mode="logpdf_residual"):
        super().__init__()
        self.eps, self.mode = eps, mode
        self.proj = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, feat):
        x = F.softplus(feat) + self.eps
        # estimate distribution parameters independently per image/channel
        mu = (x.amin(dim=(-2, -1), keepdim=True) - self.eps).clamp_min(0.0)
        shifted = (x - mu).clamp_min(self.eps)
        mean = shifted.mean(dim=(-2, -1), keepdim=True)
        var = shifted.var(dim=(-2, -1), unbiased=False, keepdim=True).clamp_min(self.eps)
        alpha = (mean.square() / var).clamp(1e-3, 1e3)
        beta = (var / mean.clamp_min(self.eps)).clamp_min(self.eps)
        log_pdf = ((alpha - 1.0) * torch.log(shifted)
                   - shifted / beta
                   - torch.lgamma(alpha)
                   - alpha * torch.log(beta))
        log_pdf = torch.nan_to_num(log_pdf, nan=0.0, posinf=10.0, neginf=-10.0)
        if self.mode == "logpdf":
            gamma_feat = self.proj(log_pdf)
        elif self.mode == "pdf":
            gamma_feat = self.proj(torch.exp(log_pdf.clamp(-20, 10)))
        else:
            gamma_feat = feat + self.proj(log_pdf)
        return gamma_feat

class CrossAttentionFusion(nn.Module):
    """Q from Swin features; K,V from Gamma-guided features, as in S1."""
    def __init__(self, channels, heads=4):
        super().__init__()
        heads = max(1, min(heads, channels))
        while channels % heads:
            heads -= 1
        self.channels, self.heads, self.dk = channels, heads, channels // heads
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, fs, fg):
        b, c, h, w = fs.shape
        qin = fs.flatten(2).transpose(1, 2)
        kin = fg.flatten(2).transpose(1, 2)
        q = self.q(qin).view(b, -1, self.heads, self.dk).transpose(1, 2)
        k = self.k(kin).view(b, -1, self.heads, self.dk).transpose(1, 2)
        v = self.v(kin).view(b, -1, self.heads, self.dk).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dk), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, h*w, c)
        out = self.norm(qin + self.out(out))
        return out.transpose(1, 2).reshape(b, c, h, w)

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.GELU())
    def forward(self, x): return self.net(x)

class UNetDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # channels ordered shallow -> deep
        self.blocks = nn.ModuleList()
        current = channels[-1]
        for skip_c in reversed(channels[:-1]):
            self.blocks.append(ConvBlock(current + skip_c, skip_c))
            current = skip_c
        self.head = nn.Conv2d(current, 1, 1)

    def forward(self, feats):
        x = feats[-1]
        for block, skip in zip(self.blocks, reversed(feats[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)

class GAMMACrossNet(nn.Module):
    def __init__(self, image_size=512, num_classes=2, pretrained=True,
                 swin_name="swin_tiny_patch4_window7_224",
                 pvt_name="pvt_v2_b2", gamma_eps=1e-6):
        super().__init__()
        self.encoder = timm.create_model(
            swin_name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3))
        channels = list(self.encoder.feature_info.channels())
        self.gamma = nn.ModuleList([GammaGuidedModule(c, gamma_eps) for c in channels])
        self.fusions = nn.ModuleList([CrossAttentionFusion(c) for c in channels])
        self.decoder = UNetDecoder(channels)
        self.classifier = timm.create_model(pvt_name, pretrained=pretrained,
                                            num_classes=num_classes)
        self.image_size = image_size

    def forward(self, image, use_gamma=True, use_cross_attention=True):
        raw = self.encoder(image)
        feats = [_as_nchw(f) for f in raw]
        fused = []
        for i, f in enumerate(feats):
            if use_gamma:
                gf = self.gamma[i](f)
            else:
                gf = f
            z = self.fusions[i](f, gf) if use_cross_attention else f
            fused.append(z)
        seg_logits = self.decoder(fused)
        seg_logits = F.interpolate(seg_logits, size=image.shape[-2:],
                                   mode="bilinear", align_corners=False)
        mask = torch.sigmoid(seg_logits)
        # ROI is derived from predicted segmentation; use soft mask for gradients.
        roi = image * mask
        cls_logits = self.classifier(F.interpolate(roi, size=(224, 224),
                                     mode="bilinear", align_corners=False))
        return {"seg_logits": seg_logits, "mask": mask, "cls_logits": cls_logits}
