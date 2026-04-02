import inspect
import torch
import torch.nn as nn
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

class ProjectionLoss(nn.Module):
    """All projection losses implement forward(zs, zs_tilde, **kwargs) with tensors shaped [B, T, D]."""
    def __init__(self):
        super().__init__()

    def _check(self, zs, zs_tilde):
        if zs.ndim != 3 or zs_tilde.ndim != 3:
            raise ValueError(f"zs and zs_tilde must be [B,T,D]; got {zs.shape=} {zs_tilde.shape=}")
        if zs.shape != zs_tilde.shape:
            raise ValueError(f"Shape mismatch: {zs.shape=} vs {zs_tilde.shape=}")

    def forward(self, zs, zs_tilde, **kwargs):
        raise NotImplementedError

    def __call__(self, zs, zs_tilde, **kwargs):
        return self.forward(zs, zs_tilde, **kwargs)

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


@register_loss("freq_time_gaussian_cosine")
class FreqTimeGaussianCosineProjectionLoss(ProjectionLoss):
    def __init__(self, **kwargs):
        super().__init__()

        # 定义一个轻量级 MLP，将时间步 t 映射为高斯滤波器的带宽参数 log(sigma)
        self.t_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

        # 精心初始化：让 bias 初始为 -1.5，使得初始 sigma ≈ 0.22。
        # 这是一个适中的低通滤波初始状态
        nn.init.constant_(self.t_mlp[-1].weight, 0)
        nn.init.constant_(self.t_mlp[-1].bias, -1.5)

    def forward(self, zs, zs_tilde, zs_tilde_original=None, t=None, **kwargs):
        self._check(zs, zs_tilde)
        if t is None:
            raise ValueError("Timestep 't' is required for FreqTimeGaussianCosineProjectionLoss!")

        B, T_seq, D = zs.shape
        H = W = int(math.isqrt(T_seq))

        # 1. 转换为空间特征并转为 float32
        # 注意：为了后续能够顺利算梯度的空域 Cosine，这里保留原始 dtype 用于还原
        orig_dtype = zs.dtype
        zs_spatial = zs.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)
        zs_tilde_spatial = zs_tilde.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)

        # 2. 正交傅里叶变换到频域
        zs_freq = torch.fft.rfft2(zs_spatial, norm='ortho')
        zs_tilde_freq = torch.fft.rfft2(zs_tilde_spatial, norm='ortho')

        # 3. 计算频域坐标网格 (物理上的频率距离的平方 D^2)
        freq_y = torch.fft.fftfreq(H, device=zs.device)
        freq_x = torch.fft.rfftfreq(W, device=zs.device)
        grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        D_sq = (grid_y ** 2 + grid_x ** 2).view(1, 1, H, -1)  # 形状: [1, 1, H, W/2+1]

        # 4. 通过 MLP 动态计算当前时间步 t 的带宽 sigma
        if t.ndim == 1:
            t_input = t.unsqueeze(1).to(torch.float32)
        else:
            t_input = t.flatten(1).mean(dim=1, keepdim=True).to(torch.float32)

        log_sigma = self.t_mlp(t_input)  # 形状: [B, 1]
        sigma_sq = torch.exp(log_sigma).view(B, 1, 1, 1) + 1e-4

        # 5. 构造可学习的高斯柔性掩码 (Gaussian Soft Mask)
        soft_mask = torch.exp(- D_sq / (2 * sigma_sq))  # 形状: [B, 1, H, W/2+1]

        # 6. 【核心对称滤波】：对 Teacher 和 Student 施加相同的平滑频域滤波
        zs_freq_filtered = zs_freq * soft_mask
        zs_tilde_freq_filtered = zs_tilde_freq * soft_mask

        # 7. 逆变换回空域 (因为是高斯掩码，绝对不会产生振铃效应！)
        zs_spatial_filtered = torch.fft.irfft2(zs_freq_filtered, s=(H, W), norm='ortho')
        zs_tilde_spatial_filtered = torch.fft.irfft2(zs_tilde_freq_filtered, s=(H, W), norm='ortho')

        # 8. 还原为原始维度和数据类型 [B, T, D]
        zs_low = zs_spatial_filtered.to(orig_dtype).flatten(2).transpose(1, 2)
        zs_tilde_low = zs_tilde_spatial_filtered.to(orig_dtype).flatten(2).transpose(1, 2)

        # 9. 纯正的空域 Cosine Similarity (流形完美匹配)
        zs_low = F.normalize(zs_low, dim=-1)
        zs_tilde_low = F.normalize(zs_tilde_low, dim=-1)

        cos_sim = (zs_low * zs_tilde_low).sum(dim=-1)
        loss = -cos_sim.mean()

        return loss


@register_loss("patch_nce")
class PatchNCEProjectionLoss(ProjectionLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__()
        # 典型的对比学习温度系数，0.07 是经典默认值
        self.temperature = temperature

    def forward(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        # 1. L2 归一化 (匹配 DINO 原始流形)
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)

        # 2. 计算点积相似度矩阵: [B, T, T]
        # sim[b, i, j] 代表 Student 的第 i 个 Patch 和 Teacher 的第 j 个 Patch 的相似度
        sim = torch.bmm(zs_tilde, zs.transpose(1, 2)) / self.temperature

        B, T_seq, _ = zs.shape

        # 3. 构造标签：正样本是对角线 (i == j)
        labels = torch.arange(T_seq, device=zs.device).unsqueeze(0).expand(B, -1)  # [B, T]

        # 4. 展平并计算交叉熵
        # CrossEntropy 会自动拉近 i==j 的特征，同时强烈排斥 i!=j 的特征
        sim = sim.view(B * T_seq, T_seq)
        labels = labels.flatten()

        loss = F.cross_entropy(sim, labels)

        return loss


@register_loss("semantic_nce")
class SemanticAwareNCE(ProjectionLoss):
    def __init__(self, temperature=0.07, pos_threshold=0.7, **kwargs):
        super().__init__()
        self.temperature = temperature
        # 语义阈值：如果 Teacher 认为两个 Patch 相似度大于 0.7，就不把它们当负样本推开
        self.pos_threshold = pos_threshold

    def forward(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        # 1. L2 归一化
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)

        B, T_seq, _ = zs.shape

        # =======================================================
        # 2. Teacher 裁判阶段 (不产生梯度)
        # =======================================================
        with torch.no_grad():
            # 计算 Teacher 自己的 Patch 两两相似度 [B, T, T]
            t_sim = torch.bmm(zs, zs.transpose(1, 2))

            # 找出 "假负样本"：Teacher 认为相似度很高的地方 (>0.7)
            false_neg_mask = t_sim > self.pos_threshold

            # 对角线 (i==j) 是真正的正样本，不能被屏蔽
            diag_mask = torch.eye(T_seq, device=zs.device, dtype=torch.bool).unsqueeze(0)

            # 我们需要屏蔽的，是那些 "既不在对角线上，又高度相似" 的坑人样本
            mask_to_ignore = false_neg_mask & (~diag_mask)

        # =======================================================
        # 3. Student 对齐阶段
        # =======================================================
        # 计算 Student 和 Teacher 的相似度矩阵
        sim = torch.bmm(zs_tilde, zs.transpose(1, 2)) / self.temperature

        # 核心：把假负样本的 Logit 设为极小值（-1e4），这样 CrossEntropy 就会无视它们，不产生排斥梯度！
        sim.masked_fill_(mask_to_ignore, -1e4)

        # 4. 计算对比损失
        labels = torch.arange(T_seq, device=zs.device).unsqueeze(0).expand(B, -1)
        loss = F.cross_entropy(sim.view(B * T_seq, T_seq), labels.flatten())

        return loss

@register_loss("vanilla_nce")
class VanillaNCE(ProjectionLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        # 1. 纯净的 L2 归一化 (匹配 DINO 流形)
        zs = F.normalize(zs, dim=-1)
        zs_tilde = F.normalize(zs_tilde, dim=-1)

        B, T_seq, _ = zs.shape

        # 2. 计算相似度矩阵 (没有任何 Teacher 裁判的干预！)
        sim = torch.bmm(zs_tilde, zs.transpose(1, 2)) / self.temperature

        # 3. 构造对角线标签 (只认为 i==i 是正样本，其他全是死对头)
        labels = torch.arange(T_seq, device=zs.device).unsqueeze(0).expand(B, -1)

        # 4. 交叉熵计算推斥力
        loss = F.cross_entropy(sim.view(B * T_seq, T_seq), labels.flatten())

        return loss


@register_loss("smooth_freq_cosine")
class SmoothFreqCosineProjectionLoss(ProjectionLoss):
    def __init__(self, sigma=4.0, **kwargs):
        """
        sigma: 高斯低通滤波器的标准差（带宽）。
        值越大，保留的高频越多；值越小，图像越平滑。
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        def gaussian_low_pass_filter(feats):
            B, T_seq, D = feats.shape
            H = W = int(math.isqrt(T_seq))

            # 1. 转为空间特征
            orig_dtype = feats.dtype
            x_spatial = feats.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)

            # 2. 正交 FFT
            x_freq = torch.fft.rfft2(x_spatial, norm='ortho')

            # 3. 构造高斯柔性掩码 (Gaussian Soft Mask)
            freq_y = torch.fft.fftfreq(H, device=feats.device)
            freq_x = torch.fft.rfftfreq(W, device=feats.device)
            grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')

            # 计算频率距离平方
            D_sq = (grid_y ** 2 + grid_x ** 2).view(1, 1, H, -1)

            # 高斯衰减函数 (平滑压制高频，而非暴力截断)
            soft_mask = torch.exp(- D_sq / (2 * (self.sigma / H) ** 2))

            # 4. 频域滤波
            x_freq_smooth = x_freq * soft_mask

            # 5. 逆变换回空域 (无振铃效应)
            x_spatial_smooth = torch.fft.irfft2(x_freq_smooth, s=(H, W), norm='ortho')

            return x_spatial_smooth.to(orig_dtype).flatten(2).transpose(1, 2)

        # 提取平滑后的低频语义
        zs_smooth = gaussian_low_pass_filter(zs)
        zs_tilde_smooth = gaussian_low_pass_filter(zs_tilde)

        # 空域 Cosine 相似度 (流形匹配)
        zs_smooth = F.normalize(zs_smooth, dim=-1)
        zs_tilde_smooth = F.normalize(zs_tilde_smooth, dim=-1)

        cos_sim = (zs_smooth * zs_tilde_smooth).sum(dim=-1)
        loss = -cos_sim.mean()

        return loss


@register_loss("freq_direct_cosine")
class FreqDirectCosineProjectionLoss(ProjectionLoss):
    def __init__(self, **kwargs):
        """
        直接在频域展开复数，算频率向量的角度对齐 (Cosine)。
        """
        super().__init__()

    def forward(self, zs, zs_tilde, zs_tilde_original=None, **kwargs):
        self._check(zs, zs_tilde)

        B, T_seq, D = zs.shape
        H = W = int(math.isqrt(T_seq))

        # 1. 转换为空间特征
        zs_spatial = zs.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)
        zs_tilde_spatial = zs_tilde.transpose(1, 2).reshape(B, D, H, W).to(torch.float32)

        # 2. 正交 FFT 到频域
        zs_freq = torch.fft.rfft2(zs_spatial, norm='ortho')
        zs_tilde_freq = torch.fft.rfft2(zs_tilde_spatial, norm='ortho')

        # 3. 将复数拆分为 [实部, 虚部] 的实数张量
        # shape: [B, D, H, W/2+1, 2]
        zs_freq_real = torch.view_as_real(zs_freq)
        zs_tilde_freq_real = torch.view_as_real(zs_tilde_freq)

        # 4. 展平为频谱向量
        # 我们按通道(D)进行展开，比较每个 Patch 的局部频谱分布
        zs_freq_flat = zs_freq_real.flatten(2)  # [B, D, H * (W/2+1) * 2]
        zs_tilde_freq_flat = zs_tilde_freq_real.flatten(2)  # [B, D, H * (W/2+1) * 2]

        # 5. 频域 L2 归一化 (让高频和低频的能量分布具有一致的角度)
        zs_freq_norm = F.normalize(zs_freq_flat, dim=-1)
        zs_tilde_freq_norm = F.normalize(zs_tilde_freq_flat, dim=-1)

        # 6. 计算频谱角度相似度
        cos_sim = (zs_freq_norm * zs_tilde_freq_norm).sum(dim=-1)
        loss = -cos_sim.mean()

        return loss