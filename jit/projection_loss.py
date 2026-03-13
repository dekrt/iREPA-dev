import inspect
import torch
import torch.nn.functional as F
from typing import Callable, Dict
import math

# =========================================
# Registry / Factory with kwarg filtering
# =========================================

_LOSS_REGISTRY: Dict[str, Callable[..., "ProjectionLoss"]] = {}

def register_loss(name: str):
    def deco(cls):
        _LOSS_REGISTRY[name] = cls
        cls.__loss_name__ = name
        return cls
    return deco

def available_losses():
    return sorted(_LOSS_REGISTRY.keys())

def _apply_aliases(cls, kwargs: dict) -> dict:
    # Optional per-class alias map, e.g. {"temperature": "tau", "t": "tau"}
    aliases = getattr(cls, "KWARG_ALIASES", None) or {}
    out = dict(kwargs)
    for a, target in aliases.items():
        if a in out and target not in out:
            out[target] = out.pop(a)
    return out

def make_projection_loss(name: str, strict: bool = False, **kwargs) -> "ProjectionLoss":
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {available_losses()}")
    cls = _LOSS_REGISTRY[name]
    kw = _apply_aliases(cls, kwargs)
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kw.items() if k in sig.parameters}
    unused = {k: v for k, v in kw.items() if k not in sig.parameters}
    if strict and unused:
        raise TypeError(f"Unused kwargs for loss '{name}': {sorted(unused)}")
    return cls(**valid)

# =========================================
# Base
# =========================================

class ProjectionLoss:
    """All projection losses implement __call__(zs, zs_tilde, **kwargs) with tensors shaped [B, T, D]."""
    def _check(self, zs, zs_tilde):
        if zs.ndim != 3 or zs_tilde.ndim != 3:
            raise ValueError(f"zs and zs_tilde must be [B,T,D]; got {zs.shape=} {zs_tilde.shape=}")
        if zs.shape != zs_tilde.shape:
            raise ValueError(f"Shape mismatch: {zs.shape=} vs {zs_tilde.shape=}")

    def __call__(self, zs, zs_tilde, **kwargs):
        raise NotImplementedError

# =========================================
# Cosine
# =========================================

@register_loss("cosine")
class CosineProjectionLoss(ProjectionLoss):
    # accepts only these kwargs; others will be ignored by factory unless strict=True
    def __init__(self, **kwargs):
        pass

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)
        # normalize zs and zs_tilde
        zs = F.normalize(zs, dim=-1) # [B,T,D]
        zs_tilde = F.normalize(zs_tilde, dim=-1) # [B,T,D]
        # compute cosine similarity
        cos_sim = (zs * zs_tilde).sum(dim=-1)    # [B,T]
        loss = -cos_sim
        return loss.mean()

@register_loss("freq_cosine")
class FreqCosineProjectionLoss(ProjectionLoss):
    def __init__(self, radius=4, **kwargs):
        """
        radius: 低通滤波器的截断半径。
        如果特征图是 16x16，radius=4 意味着只保留中心 4x4 的低频核心语义。
        """
        self.radius = radius

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        def low_pass_filter(feats):
            B, T, D = feats.shape
            H = W = int(math.isqrt(T))
            
            # 1. 转换为空间特征 [B, D, H, W]
            x_spatial = feats.transpose(1, 2).reshape(B, D, H, W)
            
            # 为了 FFT 的数值稳定性，强制转换为 float32
            orig_dtype = x_spatial.dtype
            x_spatial = x_spatial.to(torch.float32)

            # 2. 变换到频域 (Real-to-Complex FFT)
            # rfft2 输出形状为 [B, D, H, W/2 + 1]
            x_freq = torch.fft.rfft2(x_spatial, norm='ortho')

            # 3. 构造低频掩码 (Frequency Firewall)
            mask = torch.zeros_like(x_freq, dtype=torch.bool)
            r = self.radius
            
            # 在 rfft2 中，低频位于图像的左上角和左下角
            mask[:, :, :r, :r] = True      # 左上角低频
            mask[:, :, -r:, :r] = True     # 左下角低频

            # 4. 硬截断：高频直接乘 0，物理阻断梯度！
            x_freq_low = x_freq * mask

            # 5. 逆变换回空域
            x_low_spatial = torch.fft.irfft2(x_freq_low, s=(H, W), norm='ortho')
            
            # 还原为原始数据类型并展平回 [B, T, D]
            x_low_spatial = x_low_spatial.to(orig_dtype)
            return x_low_spatial.flatten(2).transpose(1, 2)

        # 分别提取 Teacher 和 Student 的低频语义
        zs_low = low_pass_filter(zs)
        zs_tilde_low = low_pass_filter(zs_tilde)

        # 仅在低频分量上进行归一化并计算 Cosine Similarity
        zs_low = F.normalize(zs_low, dim=-1)
        zs_tilde_low = F.normalize(zs_tilde_low, dim=-1)
        
        cos_sim = (zs_low * zs_tilde_low).sum(dim=-1)
        loss = -cos_sim
        return loss.mean()


@register_loss("freq_l2")
class FreqL2ProjectionLoss(ProjectionLoss):
    def __init__(self, **kwargs):
        """
        全频段对齐，不再需要截断半径参数。
        """
        pass

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        B, T, D = zs.shape
        H = W = int(math.isqrt(T))

        # 1. 展平并转换为空间特征 [B, D, H, W]
        # 为了 FFT 的数值稳定性，强制转换为 float32
        zs_spatial = zs.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)
        zs_tilde_spatial = zs_tilde.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)

        # 2. 变换到频域 (Real-to-Complex FFT)
        # 使用 norm='ortho' 保证能量守恒，输出形状为 [B, D, H, W/2 + 1] 的复数张量
        zs_freq = torch.fft.rfft2(zs_spatial, norm='ortho')
        zs_tilde_freq = torch.fft.rfft2(zs_tilde_spatial, norm='ortho')

        # 3. 处理复数并计算 L2 Loss (MSE)
        # 将复数拆分为实部和虚部，形状变为 [B, D, H, W/2 + 1, 2]
        zs_freq_real = torch.view_as_real(zs_freq)
        zs_tilde_freq_real = torch.view_as_real(zs_tilde_freq)

        # 在全频段、实部和虚部上直接计算均方误差
        loss = F.mse_loss(zs_freq_real, zs_tilde_freq_real)

        return loss


@register_loss("freq_asym_mse")
class FreqAsymMSEProjectionLoss(ProjectionLoss):
    def __init__(self, radius=4, **kwargs):
        """
        radius: 低通滤波器的截断半径。
        直接在频域进行不对称对齐：仅对 Student 滤波，Teacher 保持全频谱，直接算 L2。
        """
        self.radius = radius

    def __call__(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        B, T_seq, D = zs.shape
        H = W = int(math.isqrt(T_seq))

        # 1. 转换为空间特征 [B, D, H, W]，并转为 float32 保障 FFT 精度
        zs_spatial = zs.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)
        zs_tilde_spatial = zs_tilde.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)

        # 2. 变换到频域 (Real-to-Complex FFT)
        # 注意必须使用 norm='ortho' 保证帕塞瓦尔定理成立（能量守恒）
        zs_freq = torch.fft.rfft2(zs_spatial, norm='ortho')
        zs_tilde_freq = torch.fft.rfft2(zs_tilde_spatial, norm='ortho')

        # 3. 构造低频掩码
        mask = torch.zeros_like(zs_tilde_freq, dtype=torch.bool)
        r = self.radius
        mask[:, :, :r, :r] = True  # 左上角低频
        mask[:, :, -r:, :r] = True  # 左下角低频

        # 4. 【核心不对称截断】：仅对 Student 频域特征进行硬截断，高频置零
        # Teacher (zs_freq) 保持原样，没有任何滤波操作
        zs_tilde_freq_low = zs_tilde_freq * mask

        # 5. 【直接频域计算 Loss】：分离实部虚部，并计算 L2 (MSE)
        # torch.view_as_real 会把复数张量在最后一维展开为大小为 2 的 [实部, 虚部]
        pred_real = torch.view_as_real(zs_tilde_freq_low)
        target_real = torch.view_as_real(zs_freq)

        # 计算 MSE 损失
        loss = F.mse_loss(pred_real, target_real)
        return loss