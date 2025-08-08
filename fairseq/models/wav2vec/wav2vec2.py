# Copyright (c) Facebook, Inc. and its affiliates.
# Facebook公司及其附属机构版权所有
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 此源代码基于MIT许可证授权，许可证文件位于项目根目录

# ============================================================================
# Wav2Vec 2.0 模型实现
# 基于掩码预测的自监督语音表示学习模型
# 论文: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (2020)
# 
# 主要改进：
# 1. 使用Transformer替代卷积聚合器
# 2. 采用BERT式的掩码预测而非未来预测
# 3. 改进的向量量化策略
# 4. 更强的表示学习能力
# ============================================================================

import math  # 数学函数库
from dataclasses import dataclass, field  # 数据类装饰器，用于定义配置类
from typing import List, Tuple  # 类型注解支持

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口

from fairseq import utils  # Fairseq工具函数
from fairseq.data.data_utils import compute_mask_indices  # 掩码索引计算工具
from fairseq.dataclass import ChoiceEnum, FairseqDataclass  # Fairseq数据类和选择枚举
from fairseq.distributed import fsdp_wrap  # 完全分片数据并行包装器
from fairseq.models import BaseFairseqModel, register_model  # Fairseq基础模型类和注册装饰器
from fairseq.distributed.fully_sharded_data_parallel import FullyShardedDataParallel  # 完全分片数据并行
from fairseq.modules import (  # Fairseq预定义模块
    Fp32GroupNorm,          # 32位浮点组归一化
    Fp32LayerNorm,          # 32位浮点层归一化
    GradMultiply,           # 梯度乘法器(用于特征提取器梯度缩放)
    GumbelVectorQuantizer,  # Gumbel向量量化器
    LayerNorm,              # 层归一化
    MultiheadAttention,     # 多头注意力机制
    RelPositionalEncoding,  # 相对位置编码
    SamePad,                # 保持输入输出尺寸的填充
    TransposeLast,          # 转置最后维度的工具模块
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper  # 激活检查点包装器(节省内存)
from fairseq.modules.conformer_layer import ConformerWav2Vec2EncoderLayer  # Conformer编码器层
from fairseq.modules.transformer_sentence_encoder import init_bert_params  # BERT参数初始化
from fairseq.utils import buffered_arange, index_put, is_xla_tensor  # 工具函数

from .utils import pad_to_multiple  # 填充到指定倍数的工具函数

# ============================================================================
# 枚举类型定义
# ============================================================================
EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])  # 特征提取器模式：默认(组归一化)或层归一化
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])  # 掩码长度分布：静态、均匀、正态、泊松
LAYER_TYPE_CHOICES = ChoiceEnum(["transformer", "conformer", "trf_adp"])  # 编码器层类型：Transformer、Conformer、带适配器的Transformer


@dataclass
class Wav2Vec2Config(FairseqDataclass):
    """
    Wav2Vec 2.0模型配置类
    包含模型架构、训练策略、掩码策略、向量量化等所有可配置参数
    相比Wav2Vec 1.0，增加了Transformer架构和掩码预测相关配置
    """
    
    # ============================================================================
    # 特征提取器配置
    # ============================================================================
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )  # 特征提取器模式：default(第一层组归一化)或layer_norm(每层层归一化)
    
    # ============================================================================
    # Transformer编码器架构配置
    # ============================================================================
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )  # Transformer编码器层数：默认12层(Base模型)，Large模型为24层
    
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )  # 编码器嵌入维度：Base模型768，Large模型1024
    
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )  # 前馈网络隐藏层维度：通常为embed_dim的4倍
    
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )  # 多头注意力的头数：Base模型12头，Large模型16头
    
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )  # 激活函数：GELU在Transformer中表现更好
    
    layer_type: LAYER_TYPE_CHOICES = field(
        default="transformer", metadata={"help": "layer type in encoder"}
    )  # 编码器层类型：transformer、conformer或trf_adp(带适配器)
    
    # ============================================================================
    # Dropout正则化配置
    # ============================================================================
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )  # Transformer整体dropout概率
    
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )  # 注意力权重dropout概率
    
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )  # 前馈网络激活函数后的dropout概率
    
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )  # LayerDrop：随机跳过整个Transformer层的概率
    
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )  # 特征提取后、输入编码器前的dropout概率
    
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )  # 特征层的dropout概率(用于未掩码特征)

    # ============================================================================
    # 投影和归一化配置
    # ============================================================================
    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )  # 最终输出维度：0表示使用encoder_embed_dim，>0则投影到指定维度
    
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )  # 是否在Transformer中先应用LayerNorm(Pre-LN)：True为Pre-LN，False为Post-LN
    
    # ============================================================================
    # 卷积特征提取器配置
    # ============================================================================
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )  # 卷积特征提取层配置：相比Wav2Vec 1.0，使用更小的卷积核和步长
    
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )  # 卷积层是否使用偏置项：通常配合归一化时设为False
    
    # ============================================================================
    # 对比学习和损失函数配置
    # ============================================================================
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )  # 对比学习的温度参数：控制相似度分布的锐利程度
    
    # ============================================================================
    # 向量量化配置
    # ============================================================================
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )  # 是否对目标特征进行向量量化：离散化目标表示
    
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )  # 是否对输入特征进行向量量化：离散化输入表示
    
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )  # 输入和目标是否使用相同的量化器：共享量化参数
    
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )  # 目标特征是否使用GLU(门控线性单元)：增强非线性表达能力
    
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )  # 特征提取器梯度乘数：控制特征提取器的学习速度
    
    quantizer_depth: int = field(
        default=1,
        metadata={"help": "number of quantizer layers"},
    )  # 量化器层数：深度量化器可能有更好的表示能力
    
    quantizer_factor: int = field(
        default=3,
        metadata={
            "help": "dimensionality increase for inner quantizer layers (if depth > 1)"
        },
    )  # 量化器内部层维度扩展因子：depth>1时内部层的维度倍数
    
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )  # 每组码本中的潜在变量数量V：码本大小，影响离散化粒度
    
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )  # 码本分组数量G：乘积量化，总码本大小为V^G
    
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )  # 潜在变量维度：0表示使用final_dim/latent_groups，>0则使用指定维度

    # ============================================================================
    # 时序掩码策略配置 (核心创新：BERT式掩码预测)
    # ============================================================================
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    # 掩码长度：连续掩码的时间步数，通常为10个时间步(约100ms)
    
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )  # 掩码概率：65%的时间步会被掩码，比BERT的15%更高(语音信号冗余度更大)
    
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )  # 掩码长度选择策略：static(固定)、uniform(均匀)、normal(正态)、poisson(泊松)
    
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )  # 掩码分布的辅助参数：用于复杂分布(如泊松分布的lambda参数)
    
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )  # 是否允许掩码重叠：False允许重叠，True禁止重叠
    
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )  # 掩码间最小间隔：禁止重叠时，掩码段之间的最小距离
    
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )  # 是否要求批次内样本的掩码数量相同：保持批处理一致性
    
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )  # 掩码dropout：随机取消一部分掩码，增加训练多样性
    
    # ============================================================================
    # 通道(特征维度)掩码策略配置
    # ============================================================================
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )  # 通道掩码长度：连续掩码的特征维度数
    
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )  # 通道掩码概率：特征维度被掩码的概率(默认不使用)
    
    mask_channel_before: bool = False
    # 是否在特征提取后立即应用通道掩码：True为提取后，False为编码器前
    
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )  # 通道掩码长度选择策略：与时序掩码类似
    
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )  # 通道掩码分布的辅助参数
    
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )  # 是否允许通道掩码重叠
    
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )  # 通道掩码间最小间隔

    # ============================================================================
    # 负样本采样策略配置
    # ============================================================================
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )  # 同样本负样本数：从当前样本的其他位置采样的负样本数量
    
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )  # 是否从所有位置采样负样本：False仅从掩码位置，True从所有位置
    
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )  # 跨样本负样本数：从批次中其他样本采样的负样本数量
    
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )  # 码本负样本数：直接从量化器码本采样的负样本数量
    
    # ============================================================================
    # 位置编码配置 (相对位置编码)
    # ============================================================================
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )  # 卷积位置编码的卷积核数量：替代绝对位置编码
    
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )  # 卷积位置编码的分组数：分组卷积提高效率
    
    pos_conv_depth: int = field(
        default=1,
        metadata={"help": "depth of positional encoder network"},
    )  # 位置编码网络深度：多层卷积位置编码
    
    # ============================================================================
    # 量化器温度和训练配置
    # ============================================================================
    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )  # Gumbel Softmax温度参数：(起始温度, 结束温度, 衰减率)
    
    max_positions: int = field(default=100000, metadata={"help": "Max positions"})
    # 模型支持的最大位置数：理论上的序列长度限制
    
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )  # 激活检查点：重计算激活以节省内存(增加计算开销)
    
    # ============================================================================
    # FP16和序列长度优化配置
    # ============================================================================
    required_seq_len_multiple: int = field(
        default=2,
        metadata={
            "help": "pad the input to encoder such that the sequence length is divisible by multiple"
        },
    )  # 编码器输入序列长度必须是此数的倍数：优化FP16性能
    
    crop_seq_to_multiple: int = field(
        default=1,
        metadata={
            "help": "crop convolutional feature extractor output such that the sequence length is divisible by multiple"
        },
    )  # 特征提取器输出裁剪到此数的倍数：对齐要求
    
    # ============================================================================
    # Conformer架构专用配置
    # ============================================================================
    depthwise_conv_kernel_size: int = field(
        default=31,
        metadata={
            "help": "depthwise-conv-kernel-size for convolution in conformer layer"
        },
    )  # Conformer中深度可分离卷积的核大小：通常为奇数
    
    attn_type: str = field(
        default="",
        metadata={"help": "if espnet use ESPNET MHA"},
    )  # 注意力类型：兼容ESPnet的多头注意力实现
    
    pos_enc_type: str = field(
        default="abs",
        metadata={"help": "Positional encoding type to use in conformer"},
    )  # 位置编码类型：abs(绝对)、rel_pos(相对)、rope(旋转位置编码)
    
    fp16: bool = field(default=False, metadata={"help": "If fp16 is being used"})
    # 是否使用FP16：混合精度训练标志
    
    # ============================================================================
    # 适配器(Adapter)模块配置
    # ============================================================================
    adp_num: int = field(
        default=-1
    )  # 适配器数量：-1表示不使用适配器
    
    adp_dim: int = field(
        default=64
    )  # 适配器隐藏层维度：瓶颈架构的中间维度
    
    adp_act_fn: str = field(
        default="relu"
    )  # 适配器激活函数：ReLU、GELU等
    
    adp_trf_idx: str = field(
        default="all",
    )  # 适配器应用的Transformer层索引：all表示所有层，或指定层范围


@register_model("wav2vec2", dataclass=Wav2Vec2Config)  # 注册模型到Fairseq框架
class Wav2Vec2Model(BaseFairseqModel):
    """
    Wav2Vec 2.0 主模型类
    
    相比Wav2Vec 1.0的主要改进：
    1. 使用Transformer编码器替代卷积聚合器
    2. 采用掩码预测任务而非对比预测编码
    3. 更灵活的向量量化策略
    4. 支持Conformer等新架构
    
    模型架构：
    1. 卷积特征提取器 (ConvFeatureExtractionModel)
    2. 可选的特征投影层
    3. 掩码应用 (时序掩码 + 通道掩码)
    4. Transformer/Conformer编码器
    5. 最终投影层和对比学习头
    """
    
    def __init__(self, cfg: Wav2Vec2Config):
        """
        初始化Wav2Vec 2.0模型
        Args:
            cfg: 模型配置参数
        """
        super().__init__()  # 调用父类初始化
        self.cfg = cfg  # 保存配置引用

        # ============================================================================
        # 1. 卷积特征提取器初始化
        # ============================================================================
        feature_enc_layers = eval(cfg.conv_feature_layers)  # 解析卷积层配置字符串
        self.embed = feature_enc_layers[-1][0]  # 获取最后一层的输出维度作为嵌入维度

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,  # 卷积层配置列表
            dropout=0.0,  # 特征提取器内部不使用dropout
            mode=cfg.extractor_mode,  # 归一化模式：default或layer_norm
            conv_bias=cfg.conv_bias,  # 是否使用卷积偏置
        )

        # ============================================================================
        # 2. 特征投影层 (可选)
        # ============================================================================
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
            else None
        )  # 当特征维度与编码器维度不匹配且不使用输入量化时，添加线性投影层

        # ============================================================================
        # 3. 序列长度处理配置
        # ============================================================================
        self.crop_seq_to_multiple = cfg.crop_seq_to_multiple  # 裁剪序列长度到指定倍数

        # ============================================================================
        # 4. 时序掩码参数配置
        # ============================================================================
        self.mask_prob = cfg.mask_prob              # 掩码概率
        self.mask_selection = cfg.mask_selection    # 掩码长度选择策略
        self.mask_other = cfg.mask_other            # 掩码分布辅助参数
        self.mask_length = cfg.mask_length          # 掩码长度
        self.no_mask_overlap = cfg.no_mask_overlap  # 是否禁止掩码重叠
        self.mask_min_space = cfg.mask_min_space    # 掩码间最小间隔

        # ============================================================================
        # 5. 通道掩码参数配置
        # ============================================================================
        self.mask_channel_prob = cfg.mask_channel_prob                  # 通道掩码概率
        self.mask_channel_before = cfg.mask_channel_before              # 是否在特征提取后立即掩码
        self.mask_channel_selection = cfg.mask_channel_selection        # 通道掩码长度选择策略
        self.mask_channel_other = cfg.mask_channel_other                # 通道掩码分布辅助参数
        self.mask_channel_length = cfg.mask_channel_length              # 通道掩码长度
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap      # 是否禁止通道掩码重叠
        self.mask_channel_min_space = cfg.mask_channel_min_space        # 通道掩码间最小间隔

        # ============================================================================
        # 6. Dropout层初始化
        # ============================================================================
        self.dropout_input = nn.Dropout(cfg.dropout_input)        # 输入dropout
        self.dropout_features = nn.Dropout(cfg.dropout_features)  # 特征dropout

        # ============================================================================
        # 7. 特征提取器梯度控制
        # ============================================================================
        self.feature_grad_mult = cfg.feature_grad_mult  # 特征提取器梯度乘数

        # ============================================================================
        # 8. 量化器初始化 (延后初始化)
        # ============================================================================
        self.quantizer = None        # 目标量化器
        self.input_quantizer = None  # 输入量化器

        # ============================================================================
        # 9. 负样本采样配置
        # ============================================================================
        self.n_negatives = cfg.num_negatives                            # 同样本负样本数
        self.cross_sample_negatives = cfg.cross_sample_negatives        # 跨样本负样本数
        self.codebook_negatives = cfg.codebook_negatives                # 码本负样本数
        self.negatives_from_everywhere = cfg.negatives_from_everywhere  # 是否从所有位置采样负样本

        # ============================================================================
        # 10. 对比学习温度参数
        # ============================================================================
        self.logit_temp = cfg.logit_temp  # 对比学习的温度参数

        # ============================================================================
        # 11. 最终输出维度计算
        # ============================================================================
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim  # 最终投影维度

        # ============================================================================
        # 12. 目标量化器初始化 (可选)
        # ============================================================================
        if cfg.quantize_targets:
            # 计算量化向量维度
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,                         # 输入特征维度
                num_vars=cfg.latent_vars,               # 每组码本大小
                temp=cfg.latent_temp,                   # 温度调度参数
                groups=cfg.latent_groups,               # 量化分组数
                combine_groups=False,                   # 不合并分组
                vq_dim=vq_dim,                          # 量化向量维度
                time_first=True,                        # 时间维度在前(与Wav2Vec 1.0不同)
                weight_proj_depth=cfg.quantizer_depth, # 权重投影层深度
                weight_proj_factor=cfg.quantizer_factor, # 权重投影扩展因子
            )
            self.project_q = nn.Linear(vq_dim, final_dim)  # 量化特征投影到最终维度
        else:
            # 不使用量化时直接投影原始特征
            self.project_q = nn.Linear(self.embed, final_dim)

        # ============================================================================
        # 13. 输入量化器初始化 (可选)
        # ============================================================================
        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                # 输入和目标共享同一个量化器
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                # 为输入创建独立的量化器
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,                         # 输入特征维度
                    num_vars=cfg.latent_vars,               # 每组码本大小
                    temp=cfg.latent_temp,                   # 温度调度参数
                    groups=cfg.latent_groups,               # 量化分组数
                    combine_groups=False,                   # 不合并分组
                    vq_dim=vq_dim,                          # 量化向量维度
                    time_first=True,                        # 时间维度在前
                    weight_proj_depth=cfg.quantizer_depth, # 权重投影层深度
                    weight_proj_factor=cfg.quantizer_factor, # 权重投影扩展因子
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)  # 量化输入投影到编码器维度

        # ============================================================================
        # 14. 掩码嵌入向量
        # ============================================================================
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )  # 掩码令牌的可学习嵌入向量，用均匀分布初始化

        # ============================================================================
        # 15. 编码器选择和初始化
        # ============================================================================
        encoder_cls = TransformerEncoder  # 默认使用Transformer编码器
        if cfg.layer_type == "conformer" and cfg.pos_enc_type in ["rel_pos", "rope"]:
            encoder_cls = ConformerEncoder  # 使用Conformer编码器(需要相对位置编码)

        self.encoder = encoder_cls(cfg)  # 初始化编码器
        self.layer_norm = LayerNorm(self.embed)  # 特征归一化层

        # ============================================================================
        # 16. 目标GLU层 (可选)
        # ============================================================================
        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )  # 门控线性单元：增强目标特征的非线性表达能力

        # ============================================================================
        # 17. 最终投影层
        # ============================================================================
        self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)  # 编码器输出投影到最终维度

    def upgrade_state_dict_named(self, state_dict, name):
        """
        升级状态字典以兼容新版本的Fairseq
        Args:
            state_dict: 模型状态字典
            name: 模型名称
        Returns:
            更新后的状态字典
        """
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2Config, task=None):
        """
        构建新的模型实例 (工厂方法)
        Args:
            cfg: 模型配置
            task: 任务配置(可选)
        Returns:
            Wav2Vec2Model实例
        """
        return cls(cfg)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        """
        应用掩码到输入特征 (Wav2Vec 2.0的核心创新)
        
        掩码策略：
        1. 通道掩码(可选)：在特征维度上掩码，类似SpecAugment
        2. 时序掩码：在时间维度上掩码，类似BERT的[MASK]
        
        Args:
            x: 输入特征 [B, T, C]
            padding_mask: 填充掩码 [B, T]
            mask_indices: 预计算的时序掩码索引(可选)
            mask_channel_indices: 预计算的通道掩码索引(可选)
        
        Returns:
            masked_x: 掩码后的特征 [B, T, C]
            mask_indices: 时序掩码索引 [B, T] (用于损失计算)
        """
        B, T, C = x.shape  # 批次大小、时间步数、特征维度

        # ============================================================================
        # 1. 早期通道掩码 (特征提取后立即掩码)
        # ============================================================================
        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),                              # 掩码形状：批次×特征维度
                None,                                # 通道掩码不考虑padding
                self.mask_channel_prob,              # 通道掩码概率
                self.mask_channel_length,            # 连续掩码的特征维度数
                self.mask_channel_selection,         # 掩码长度分布策略
                self.mask_channel_other,             # 分布辅助参数
                no_overlap=self.no_mask_channel_overlap,     # 是否禁止重叠
                min_space=self.mask_channel_min_space,       # 最小间隔
            )
            # 转换为张量并扩展到时间维度：[B, C] -> [B, T, C]
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)           # [B, C] -> [B, 1, C]
                .expand(-1, T, -1)      # [B, 1, C] -> [B, T, C]
            )
            x[mask_channel_indices] = 0  # 将掩码位置置零

        # ============================================================================
        # 2. 时序掩码 (核心：BERT式掩码预测)
        # ============================================================================
        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),                          # 掩码形状：批次×时间步
                    padding_mask,                    # 考虑序列的有效长度
                    self.mask_prob,                  # 掩码概率(65%)
                    self.mask_length,                # 连续掩码长度(10步)
                    self.mask_selection,             # 掩码长度分布策略
                    self.mask_other,                 # 分布辅助参数
                    min_masks=2,                     # 最少掩码数量
                    no_overlap=self.no_mask_overlap, # 是否禁止重叠
                    min_space=self.mask_min_space,   # 最小间隔
                    require_same_masks=self.cfg.require_same_masks,  # 批次内掩码数量一致
                    mask_dropout=self.cfg.mask_dropout,              # 掩码dropout
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            # 用可学习的掩码嵌入向量替换掩码位置的特征
            x = index_put(x, mask_indices, self.mask_emb)
        else:
            mask_indices = None

        # ============================================================================
        # 3. 晚期通道掩码 (编码器前掩码)
        # ============================================================================
        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),                              # 掩码形状：批次×特征维度
                    None,                                # 通道掩码不考虑padding
                    self.mask_channel_prob,              # 通道掩码概率
                    self.mask_channel_length,            # 连续掩码的特征维度数
                    self.mask_channel_selection,         # 掩码长度分布策略
                    self.mask_channel_other,             # 分布辅助参数
                    no_overlap=self.no_mask_channel_overlap,     # 是否禁止重叠
                    min_space=self.mask_channel_min_space,       # 最小间隔
                )
                # 转换为张量并扩展到时间维度
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)           # [B, C] -> [B, 1, C]
                    .expand(-1, T, -1)      # [B, 1, C] -> [B, T, C]
                )
            x = index_put(x, mask_channel_indices, 0)  # 将掩码位置置零

        return x, mask_indices

    def sample_negatives(self, y, num, padding_count=None):
        """
        采样负样本用于对比学习 (InfoNCE损失)
        
        负样本策略：
        1. 同样本负样本：从当前样本的其他时间步采样
        2. 跨样本负样本：从批次中其他样本采样
        
        Args:
            y: 目标特征 [B, T, C]
            num: 每个位置需要的负样本总数
            padding_count: 填充的时间步数(可选)
        
        Returns:
            negs: 负样本特征 [N, B, T, C] (N=负样本数)
            neg_idxs: 负样本索引
        """
        
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)  # 不使用负样本时返回空张量

        bsz, tsz, fsz = y.shape  # 批次大小、时间步数、特征维度
        y = y.view(-1, fsz)  # 重塑为 [B*T, C] 便于索引

        # ============================================================================
        # 计算采样范围
        # ============================================================================
        cross_high = tsz * bsz          # 跨样本采样的总范围
        high = tsz - (padding_count or 0)  # 有效时间步数(排除填充)
        
        with torch.no_grad():
            assert high > 1, f"有效时间步数必须>1, 当前: {bsz,tsz,fsz}"

            # ========================================================================
            # 1. 同样本负样本采样
            # ========================================================================
            if self.n_negatives > 0:
                # 为每个掩码位置生成时间步索引
                tszs = (
                    buffered_arange(num)                    # [0, 1, ..., num-1]
                    .unsqueeze(-1)                          # [num, 1]
                    .expand(-1, self.n_negatives)          # [num, n_negatives]
                    .flatten()                              # [num * n_negatives]
                )

                # 随机采样负样本索引(避免采样到正样本位置)
                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                # 当索引>=当前位置时，向后偏移1以跳过正样本
                neg_idxs[neg_idxs >= tszs] += 1

            # ========================================================================
            # 2. 跨样本负样本采样
            # ========================================================================
            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)                           # [0, 1, ..., num-1]
                    .unsqueeze(-1)                                 # [num, 1]
                    .expand(-1, self.cross_sample_negatives)       # [num, cross_negatives]
                    .flatten()                                     # [num * cross_negatives]
                )

                # 从整个批次范围内随机采样
                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        # ============================================================================
        # 3. 合并负样本索引
        # ============================================================================
        if self.n_negatives > 0:
            # 为同样本负样本添加批次偏移
            neg_idxs = neg_idxs + (torch.arange(bsz).unsqueeze(1) * high)
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            # 合并同样本和跨样本负样本
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        # ============================================================================
        # 4. 根据索引提取负样本特征
        # ============================================================================
        negs = y[neg_idxs.view(-1)]  # 提取负样本特征
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # 重塑为 [N, B, T, C] 格式
        
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):
        """
        计算对比学习的预测logits (InfoNCE损失核心)
        
        计算上下文表示与正/负样本的相似度，用于对比学习
        
        Args:
            x: 上下文表示(编码器输出) [B, T, C]
            y: 正样本目标特征 [B, T, C]  
            negatives: 负样本特征 [N, B, T, C]
        
        Returns:
            logits: 相似度分数 [N+1, B, T] (第0维是正样本，其余是负样本)
        """
        
        # ============================================================================
        # 1. 检测负样本中是否有与正样本相同的 (避免虚假负样本)
        # ============================================================================
        neg_is_pos = (y == negatives).all(-1)  # 检查负样本是否与正样本相同
        
        # ============================================================================
        # 2. 合并正样本和负样本
        # ============================================================================
        y = y.unsqueeze(0)  # [B, T, C] -> [1, B, T, C]
        targets = torch.cat([y, negatives], dim=0)  # [N+1, B, T, C]
        
        # ============================================================================
        # 3. 计算余弦相似度
        # ============================================================================
        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1)
        # 余弦相似度：cos(θ) = (x·y)/(||x||·||y||)，范围[-1, 1]
        
        # ============================================================================
        # 4. 温度缩放
        # ============================================================================
        logits = logits / self.logit_temp  # 温度缩放：τ越小分布越锐利
        logits = logits.type_as(x)  # 保持原始数据类型
        
        # ============================================================================
        # 5. 处理虚假负样本 (将相同的负样本设为负无穷)
        # ============================================================================
        if is_xla_tensor(logits) or neg_is_pos.any():
            if not hasattr(self, "_inftensor"):
                fillval = -float(2**30)  # 接近负无穷的值
                self._inftensor = (
                    torch.tensor(fillval).to(x.device)
                    if is_xla_tensor(logits)
                    else float("-inf")
                )
            # 将虚假负样本的logits设为负无穷(在softmax中概率趋于0)
            logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
        corpus_key=None,
    ):
        """
        Wav2Vec 2.0主模型前向传播
        
        完整的自监督学习流程：
        1. 特征提取：原始音频 -> 卷积特征
        2. 掩码应用：BERT式掩码预测
        3. 上下文编码：Transformer编码器
        4. 对比学习：InfoNCE损失
        
        Args:
            source: 原始音频波形 [B, T]
            padding_mask: 填充掩码 [B, T]
            mask: 是否应用掩码（训练时True，推理时False）
            features_only: 是否只返回特征（用于下游任务）
            layer: 目标编码器层
            mask_indices: 预计算的掩码索引
            mask_channel_indices: 预计算的通道掩码索引
            padding_count: 填充计数
            corpus_key: 语料库键（适配器选择）
        
        Returns:
            result: 包含logits、掩码、困惑度等的字典
        """

        # ============================================================================
        # 1. 卷积特征提取 (原始音频 -> 高层特征)
        # ============================================================================
        if self.feature_grad_mult > 0:
            # 允许梯度回传到特征提取器
            features = self.feature_extractor(source)  # [B, T] -> [B, C, T]
            if self.feature_grad_mult != 1.0:
                # 梯度缩放：控制特征提取器的学习速度
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            # 冻结特征提取器：不更新预训练的卷积层
            with torch.no_grad():
                features = self.feature_extractor(source)

        # ============================================================================
        # 2. 特征惩罚项 (正则化)
        # ============================================================================
        features_pen = features.float().pow(2).mean()  # L2惩罚项：防止特征过大

        # ============================================================================
        # 3. 特征格式转换和归一化
        # ============================================================================
        features = features.transpose(1, 2)  # [B, C, T] -> [B, T, C] (Transformer格式)
        features = self.layer_norm(features)  # 层归一化：稳定训练
        unmasked_features = features.clone()  # 保存未掩码的特征用于目标计算

        # ============================================================================
        # 4. 填充掩码处理 (处理变长序列)
        # ============================================================================
        if padding_mask is not None and padding_mask.any():
            # 计算每个样本的有效长度
            input_lengths = (1 - padding_mask.long()).sum(-1)  # [B]
            # 根据卷积下采样计算输出长度
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            # 重新构建padding_mask以匹配特征序列长度
            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # 标记有效序列的最后一个位置
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,  # 最后一个有效位置
                )
            ] = 1
            # 使用cumsum创建掩码：有效位置为False，填充位置为True
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        # ============================================================================
        # 5. 序列长度裁剪 (满足模型架构要求)
        # ============================================================================
        time_steps_to_drop = features.size(1) % self.crop_seq_to_multiple
        if time_steps_to_drop != 0:
            # 裁剪到指定倍数，确保后续处理的兼容性
            features = features[:, :-time_steps_to_drop]
            unmasked_features = unmasked_features[:, :-time_steps_to_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-time_steps_to_drop]

        # ============================================================================
        # 6. 特征投影 (维度对齐)
        # ============================================================================
        if self.post_extract_proj is not None:
            # 将卷积特征投影到编码器维度
            features = self.post_extract_proj(features)

        # ============================================================================
        # 7. 输入dropout (防止过拟合)
        # ============================================================================
        features = self.dropout_input(features)          # 对输入特征应用dropout
        unmasked_features = self.dropout_features(unmasked_features)  # 对未掩码特征应用dropout

        # ============================================================================
        # 8. 量化器状态变量初始化
        # ============================================================================
        num_vars = None      # 码本变量数量
        code_ppl = None      # 码本困惑度
        prob_ppl = None      # 概率困惑度
        curr_temp = None     # 当前温度参数

        # ============================================================================
        # 9. 输入量化 (可选)
        # ============================================================================
        if self.input_quantizer:
            # 对输入特征进行向量量化
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]                    # 量化后的特征
            num_vars = q["num_vars"]             # 码本大小
            code_ppl = q["code_perplexity"]      # 码本使用的均匀程度
            prob_ppl = q["prob_perplexity"]      # 概率分布的均匀程度
            curr_temp = q["temp"]                # Gumbel softmax温度
            features = self.project_inp(features)  # 投影到编码器维度

        # ============================================================================
        # 10. 掩码应用 (核心：BERT式掩码语言建模)
        # ============================================================================
        if mask:
            # 应用时序掩码和通道掩码
            x, mask_indices = self.apply_mask(
                features,                            # 输入特征
                padding_mask,                        # 填充掩码
                mask_indices=mask_indices,           # 预计算的掩码索引
                mask_channel_indices=mask_channel_indices,  # 预计算的通道掩码索引
            )
            
            if not is_xla_tensor(x) and mask_indices is not None:
                # 提取被掩码位置的原始特征作为预测目标
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )  # [B, masked_timesteps, C]
            else:
                # XLA模式下或未指定掩码时使用全部特征
                y = unmasked_features
        else:
            # 推理模式：不应用掩码
            x = features               # 编码器输入
            y = unmasked_features      # 目标特征
            mask_indices = None        # 无掩码

        # ============================================================================
        # 11. Transformer编码器 (上下文建模)
        # ============================================================================
        x, layer_results = self.encoder(
            x,                          # 掩码后的特征 [B, T, C]
            padding_mask=padding_mask,  # 填充掩码 [B, T]
            layer=layer,                # 目标层
            corpus_key=corpus_key       # 语料库键(适配器)
        )  # 输出：x [B, T, C], layer_results [(x, z, lr), ...]

        # ============================================================================
        # 12. 特征提取模式 (下游任务使用)
        # ============================================================================
        if features_only:
            # 只返回编码器特征，不进行对比学习
            return {
                "x": x,                      # 编码器输出 [B, T, C]
                "padding_mask": padding_mask,  # 填充掩码 [B, T]
                "features": unmasked_features,  # 未掩码的原始特征
                "layer_results": layer_results,  # 各层的输出结果
            }

        # ============================================================================
        # 13. 目标量化和负样本采样 (对比学习核心)
        # ============================================================================
        if self.quantizer:
            # ------------------------------------------------------------------------
            # 13.1 目标特征量化
            # ------------------------------------------------------------------------
            if self.negatives_from_everywhere:
                # 从完整的未掩码特征中量化和采样
                q = self.quantizer(unmasked_features, produce_targets=False)
                y = q["x"]                       # 量化后的目标特征
                num_vars = q["num_vars"]         # 码本大小
                code_ppl = q["code_perplexity"]  # 码本困惑度
                prob_ppl = q["prob_perplexity"]  # 概率困惑度
                curr_temp = q["temp"]            # 当前温度
                y = self.project_q(y)            # 投影到最终维度

                # 从量化特征中采样负样本
                negs, _ = self.sample_negatives(
                    y,
                    mask_indices[0].sum(),       # 掩码位置总数
                    padding_count=padding_count,
                )
                # 提取掩码位置的目标特征
                y = y[mask_indices].view(y.size(0), -1, y.size(-1))

            else:
                # 仅对掩码位置的特征进行量化
                q = self.quantizer(y, produce_targets=False)
                y = q["x"]                       # 量化后的目标特征
                num_vars = q["num_vars"]         # 码本大小
                code_ppl = q["code_perplexity"]  # 码本困惑度
                prob_ppl = q["prob_perplexity"]  # 概率困惑度
                curr_temp = q["temp"]            # 当前温度

                y = self.project_q(y)            # 投影到最终维度

                # 从目标特征中采样负样本
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),                   # 目标序列长度
                    padding_count=padding_count,
                )

            # ------------------------------------------------------------------------
            # 13.2 码本负样本 (增加对比学习难度)
            # ------------------------------------------------------------------------
            if self.codebook_negatives > 0:
                # 直接从码本中采样负样本
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # [codebook_negs, B, T, C]
                cb_negs = self.project_q(cb_negs)  # 投影到最终维度
                negs = torch.cat([negs, cb_negs], dim=0)  # 合并负样本
        else:
            # ------------------------------------------------------------------------
            # 13.3 无量化器模式：直接使用连续特征
            # ------------------------------------------------------------------------
            y = self.project_q(y)  # 投影目标特征

            if self.negatives_from_everywhere:
                # 从完整特征中采样负样本
                negs, _ = self.sample_negatives(
                    unmasked_features,
                    y.size(1),
                    padding_count=padding_count,
                )
                negs = self.project_q(negs)  # 投影负样本
            else:
                # 从目标特征中采样负样本
                negs, _ = self.sample_negatives(
                    y,
                    y.size(1),
                    padding_count=padding_count,
                )

        # ============================================================================
        # 14. 预测头计算 (对比学习损失)
        # ============================================================================
        if not is_xla_tensor(x):
            # 提取掩码位置的上下文表示
            x = x[mask_indices].view(x.size(0), -1, x.size(-1))  # [B, masked_T, C]

        # 可选的目标门控单元 (GLU)
        if self.target_glu:
            y = self.target_glu(y)      # 对目标特征应用GLU
            negs = self.target_glu(negs)  # 对负样本应用GLU

        # 最终投影和对比预测
        x = self.final_proj(x)          # 投影上下文表示到最终维度
        x = self.compute_preds(x, y, negs)  # 计算对比学习logits

        # ============================================================================
        # 15. 构建返回结果
        # ============================================================================
        result = {
            "x": x,                     # 对比学习logits [N+1, B, T]
            "padding_mask": padding_mask,  # 填充掩码
            "features_pen": features_pen,  # 特征惩罚项
        }

        # 添加量化器相关的统计信息
        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl  # 概率困惑度
            result["code_perplexity"] = code_ppl  # 码本困惑度
            result["num_vars"] = num_vars         # 码本大小
            result["temp"] = curr_temp            # 当前温度

        return result

    def quantize(self, x):
        """
        对输入音频进行量化 (返回离散码本索引)
        
        Args:
            x: 原始音频波形 [B, T]
        
        Returns:
            量化索引 [B, T']
        """
        assert self.quantizer is not None, "模型必须配置量化器"
        x = self.feature_extractor(x)     # 特征提取 [B, T] -> [B, C, T]
        x = x.transpose(1, 2)             # [B, C, T] -> [B, T, C]
        x = self.layer_norm(x)            # 层归一化
        return self.quantizer.forward_idx(x)  # 返回量化索引

    def extract_features(
        self, source, padding_mask, mask=False, layer=None, corpus_key=None
    ):
        """
        提取特征的便利接口 (用于下游任务)
        
        Args:
            source: 原始音频波形 [B, T]
            padding_mask: 填充掩码 [B, T]
            mask: 是否应用掩码
            layer: 目标编码器层
            corpus_key: 语料库键
        
        Returns:
            特征字典：包含编码器输出、填充掩码、层结果等
        """
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,          # 只返回特征，不进行对比学习
            layer=layer,
            corpus_key=corpus_key,
        )
        return res

    def get_logits(self, net_output):
        """
        获取对比学习的logits (用于损失计算)
        
        Args:
            net_output: 模型输出字典
        
        Returns:
            展平的logits [B*T, N+1] (N+1: 1个正样本+N个负样本)
        """
        logits = net_output["x"]          # [N+1, B, T] 对比学习logits
        logits = logits.transpose(0, 2)   # [N+1, B, T] -> [T, B, N+1]
        logits = logits.reshape(-1, logits.size(-1))  # [T*B, N+1]
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        """
        获取对比学习的目标标签 (正样本索引始终为0)
        
        Args:
            sample: 数据样本
            net_output: 模型输出
            expand_steps: 是否扩展到所有时间步
        
        Returns:
            目标标签 [B*T] (全为0，表示第0个位置是正样本)
        """
        x = net_output["x"]  # [N+1, B, T]
        # 对比学习中，正样本总是第0个位置，所以目标全为0
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        """
        计算额外的正则化损失
        
        包括：
        1. 量化器损失：鼓励码本的均匀使用
        2. 特征惩罚项：防止特征幅值过大
        
        Args:
            net_output: 模型输出字典
        
        Returns:
            正则化损失列表
        """
        pen = []

        if "prob_perplexity" in net_output:
            # 量化器多样性损失：鼓励使用更多的码本条目
            # 损失 = (总码本数 - 实际困惑度) / 总码本数
            # 当困惑度接近码本数时，损失接近0（理想状态）
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            # 特征L2惩罚项：防止特征幅值过大
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self, last_layer=None):
        """
        移除预训练模块 (用于下游任务微调)
        
        移除自监督学习相关的组件：
        - 量化器
        - 投影层  
        - GLU门控
        - 最终投影
        
        Args:
            last_layer: 保留的最后一层编码器索引
        """
        self.quantizer = None      # 移除量化器
        self.project_q = None      # 移除目标投影层
        self.target_glu = None     # 移除目标GLU
        self.final_proj = None     # 移除最终投影层

        if last_layer is not None:
            # 只保留前N层编码器（用于轻量化或特定任务）
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class ConvFeatureExtractionModel(nn.Module):
    """
    卷积特征提取器 (Wav2Vec 2.0的音频特征提取模块)
    
    将原始音频波形转换为高层语义特征，相比Wav2Vec 1.0：
    1. 使用更小的卷积核和步长（降低下采样倍数）
    2. 支持不同的归一化策略（组归一化/层归一化）
    3. 使用GELU激活函数替代ReLU
    
    默认配置：[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2
    总下采样倍数：5 × 2^4 × 2^2 = 320 (vs Wav2Vec1.0的160)
    """
    
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        """
        初始化卷积特征提取器
        
        Args:
            conv_layers: 卷积层配置列表 [(输出维度, 核大小, 步长), ...]
            dropout: dropout概率
            mode: 归一化模式 ("default"=组归一化, "layer_norm"=层归一化)
            conv_bias: 是否使用卷积偏置
        """
        super().__init__()

        assert mode in {"default", "layer_norm"}, f"不支持的模式: {mode}"

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            """
            构建单个卷积块
            
            架构: Conv1d -> Dropout -> Normalization -> GELU
            """
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)  # He初始化，适合ReLU/GELU
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "层归一化和组归一化互斥"

            if is_layer_norm:
                # 层归一化模式：每层都使用LayerNorm
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),                              # [B, C, T] -> [B, T, C]
                        Fp32LayerNorm(dim, elementwise_affine=True),  # 层归一化
                        TransposeLast(),                              # [B, T, C] -> [B, C, T]
                    ),
                    nn.GELU(),  # GELU激活函数(相比ReLU更平滑)
                )
            elif is_group_norm:
                # 组归一化模式：仅第一层使用GroupNorm
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),  # 组归一化(每个通道一组)
                    nn.GELU(),
                )
            else:
                # 无归一化模式
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        # ============================================================================
        # 构建卷积层序列
        # ============================================================================
        in_d = 1  # 输入维度(原始音频)
        self.conv_layers = nn.ModuleList()
        
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, f"卷积层配置错误: {cl}"
            (dim, k, stride) = cl  # 输出维度、核大小、步长

            self.conv_layers.append(
                block(
                    in_d,       # 输入维度
                    dim,        # 输出维度
                    k,          # 卷积核大小
                    stride,     # 步长
                    is_layer_norm=mode == "layer_norm",           # 是否使用层归一化
                    is_group_norm=mode == "default" and i == 0,   # 仅第一层使用组归一化
                    conv_bias=conv_bias,                          # 是否使用偏置
                )
            )
            in_d = dim  # 更新下一层的输入维度

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 原始音频波形 [B, T] (批次大小, 时间步)
        
        Returns:
            features: 卷积特征 [B, C, T'] (批次大小, 特征维度, 下采样后时间步)
        """
        
        # ============================================================================
        # 1. 添加通道维度: [B, T] -> [B, 1, T]
        # ============================================================================
        x = x.unsqueeze(1)  # 为1D卷积添加通道维度

        # ============================================================================
        # 2. 逐层卷积特征提取
        # ============================================================================
        for conv in self.conv_layers:
            x = conv(x)  # 应用卷积块：Conv1d -> Dropout -> Norm -> GELU

        return x  # [B, C, T'] 其中T' = T / 下采样倍数


def make_conv_pos(e, k, g, is_batch_norm=False):
    """
    创建卷积位置编码层 (Wav2Vec 2.0的位置编码创新)
    
    使用组卷积实现相对位置编码，相比绝对位置编码：
    1. 能够处理任意长度的序列
    2. 具有平移不变性
    3. 参数量更少
    
    Args:
        e: 嵌入维度
        k: 卷积核大小(控制位置感受野)
        g: 组数(减少参数，增加归纳偏置)
        is_batch_norm: 是否使用批归一化
    
    Returns:
        位置编码模块: Conv1d -> (BatchNorm/WeightNorm) -> Padding -> GELU
    """
    # ============================================================================
    # 1. 构建组卷积层
    # ============================================================================
    pos_conv = nn.Conv1d(
        e,              # 输入通道数
        e,              # 输出通道数(保持维度不变)
        kernel_size=k,  # 卷积核大小
        padding=k // 2, # 保持序列长度不变的填充
        groups=g,       # 组卷积(减少参数：e*k -> e*k/g)
    )
    
    # ============================================================================
    # 2. 权重初始化 (基于方差缩放)
    # ============================================================================
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))  # 方差缩放初始化
    nn.init.normal_(pos_conv.weight, mean=0, std=std)  # 正态分布初始化
    nn.init.constant_(pos_conv.bias, 0)               # 偏置置零

    # ============================================================================
    # 3. 归一化策略选择和模块构建
    # ============================================================================
    if not is_batch_norm:
        # 权重归一化：稳定训练，适合序列模型
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    else:
        # 批归一化：可能在变长序列上不稳定
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(k), nn.GELU())

    return pos_conv


class TransformerEncoder(nn.Module):
    """
    Transformer编码器 (Wav2Vec 2.0的上下文建模核心)
    
    支持三种编码器架构：
    1. Transformer: 标准自注意力机制
    2. Conformer: 卷积增强的Transformer（结合CNN和自注意力）
    3. TRF_ADP: 带适配器的Transformer（用于参数高效迁移学习）
    """
    
    def build_encoder_layer(self, args: Wav2Vec2Config, **kwargs):
        """
        构建编码器层
        
        Args:
            args: 模型配置
            **kwargs: 额外参数(如layer_idx用于适配器索引)
        
        Returns:
            编码器层实例
        """
        if args.layer_type == "transformer":
            # ========================================================================
            # 标准Transformer编码器层
            # ========================================================================
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,                     # 嵌入维度
                ffn_embedding_dim=args.encoder_ffn_embed_dim,         # 前馈网络维度
                num_attention_heads=args.encoder_attention_heads,     # 注意力头数
                dropout=self.dropout,                                 # 主dropout
                attention_dropout=args.attention_dropout,             # 注意力dropout
                activation_dropout=args.activation_dropout,           # 激活函数dropout
                activation_fn=args.activation_fn,                     # 激活函数类型
                layer_norm_first=args.layer_norm_first,               # Pre-LN vs Post-LN
            )
        elif args.layer_type == "conformer":
            # ========================================================================
            # Conformer编码器层 (卷积+自注意力)
            # ========================================================================
            layer = ConformerWav2Vec2EncoderLayer(
                embed_dim=self.embedding_dim,                         # 嵌入维度
                ffn_embed_dim=args.encoder_ffn_embed_dim,             # 前馈网络维度
                attention_heads=args.encoder_attention_heads,         # 注意力头数
                dropout=args.dropout,                                 # dropout概率
                depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,  # 深度卷积核大小
                activation_fn="swish",                                # Swish激活函数
                attn_type=args.attn_type,                             # 注意力类型
                use_fp16=args.fp16,                                   # 是否使用半精度
                pos_enc_type="abs",                                   # 绝对位置编码
            )
        elif args.layer_type == "trf_adp":
            # ========================================================================
            # 适配器增强的Transformer (参数高效迁移学习)
            # ========================================================================
            use_adp = False
            if args.adp_trf_idx == "all":
                # 所有层都使用适配器
                use_adp = True
            else:
                # 仅指定层使用适配器 (格式: "start:end")
                adp_trf_idx = list(range(*[int(g) for g in args.adp_trf_idx.split(":")]))
                if kwargs.get("layer_idx", None) in adp_trf_idx:
                    use_adp = True
                    
            if use_adp:
                # 带适配器的Transformer层
                layer = TransformerSentenceEncoderWithAdapterLayer(
                    embedding_dim=self.embedding_dim,                 # 嵌入维度
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,     # 前馈网络维度
                    num_attention_heads=args.encoder_attention_heads, # 注意力头数
                    dropout=self.dropout,                             # 主dropout
                    attention_dropout=args.attention_dropout,         # 注意力dropout
                    activation_dropout=args.activation_dropout,       # 激活函数dropout
                    activation_fn=args.activation_fn,                 # 激活函数类型
                    layer_norm_first=args.layer_norm_first,           # Pre-LN vs Post-LN
                    adapter_num=args.adp_num,                         # 适配器数量
                    adapter_dim=args.adp_dim,                         # 适配器维度
                    adapter_act_fn=args.adp_act_fn,                   # 适配器激活函数
                )
            else:
                # 标准Transformer层(不使用适配器)
                layer = TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,                 # 嵌入维度
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,     # 前馈网络维度
                    num_attention_heads=args.encoder_attention_heads, # 注意力头数
                    dropout=self.dropout,                             # 主dropout
                    attention_dropout=args.attention_dropout,         # 注意力dropout
                    activation_dropout=args.activation_dropout,       # 激活函数dropout
                    activation_fn=args.activation_fn,                 # 激活函数类型
                    layer_norm_first=args.layer_norm_first,           # Pre-LN vs Post-LN
                )

        # ============================================================================
        # 优化配置：分布式训练和内存优化
        # ============================================================================
        layer = fsdp_wrap(layer)  # FSDP包装：用于大模型分布式训练
        if args.checkpoint_activations:
            # 激活检查点：用时间换显存
            layer = checkpoint_wrapper(layer)
        return layer

    def __init__(self, args: Wav2Vec2Config, skip_pos_conv: bool = False, override_encoder_layer: int = None):
        """
        初始化Transformer编码器
        
        Args:
            args: 模型配置
            skip_pos_conv: 是否跳过位置卷积
            override_encoder_layer: 覆盖编码器层数
        """
        super().__init__()

        # ============================================================================
        # 基础配置
        # ============================================================================
        self.dropout = args.dropout                                 # dropout概率
        self.embedding_dim = args.encoder_embed_dim                 # 嵌入维度
        self.required_seq_len_multiple = args.required_seq_len_multiple  # 序列长度倍数要求

        # ============================================================================
        # 位置编码配置 (三种模式)
        # ============================================================================
        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            # 深度位置卷积：多层卷积块
            num_layers = args.pos_conv_depth                        # 位置卷积层数
            k = max(3, args.conv_pos // num_layers)                 # 每层卷积核大小

            def make_conv_block(e, k, g, l):
                """构建多层位置卷积块"""
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,              # 输入/输出维度
                                e,              # 保持维度不变
                                kernel_size=k,  # 卷积核大小
                                padding=k // 2, # 保持序列长度
                                groups=g,       # 组卷积
                            ),
                            SamePad(k),                                    # 相同填充
                            TransposeLast(),                               # [B,C,T] -> [B,T,C]
                            LayerNorm(e, elementwise_affine=False),        # 层归一化
                            TransposeLast(),                               # [B,T,C] -> [B,C,T]
                            nn.GELU(),                                     # GELU激活
                        )
                        for _ in range(l)  # 重复l层
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )
        elif skip_pos_conv:
            # 无位置编码：适用于预训练特征已包含位置信息的情况
            self.pos_conv = None
        else:
            # 标准位置卷积：单层组卷积
            self.pos_conv = make_conv_pos(
                self.embedding_dim,                                   # 嵌入维度
                args.conv_pos,                                        # 卷积核大小
                args.conv_pos_groups,                                 # 组数
                is_batch_norm=args.conv_pos_batch_norm               # 是否使用批归一化
                if hasattr(args, "conv_pos_batch_norm")
                else False,
            )

        # ============================================================================
        # 编码器层构建
        # ============================================================================
        if override_encoder_layer is None:
            encoder_layers = args.encoder_layers                     # 使用配置中的层数
        else:
            encoder_layers = override_encoder_layer                  # 使用覆盖的层数

        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args, layer_idx=ii) for ii in range(encoder_layers)]
        )
        
        # ============================================================================
        # 归一化和正则化配置
        # ============================================================================
        self.layer_norm_first = args.layer_norm_first               # Pre-LN vs Post-LN
        self.layer_norm = LayerNorm(self.embedding_dim)             # 最终层归一化
        self.layerdrop = args.encoder_layerdrop                     # 层dropout概率

        # ============================================================================
        # 参数初始化
        # ============================================================================
        self.apply(init_bert_params)  # 使用BERT风格的参数初始化

    def forward(self, x, padding_mask=None, layer=None, corpus_key=None):
        """
        Transformer编码器前向传播
        
        Args:
            x: 输入特征 [B, T, C]
            padding_mask: 填充掩码 [B, T]
            layer: 目标层索引(如果只需要前N层的输出)
            corpus_key: 语料库键(用于适配器选择)
        
        Returns:
            x: 编码后的特征 [B, T, C]
            layer_results: 各层的输出结果列表
        """
        # ============================================================================
        # 1. 特征提取：通过所有Transformer层
        # ============================================================================
        x, layer_results = self.extract_features(
            x, padding_mask, layer, corpus_key=corpus_key
        )

        # ============================================================================
        # 2. 最终归一化 (Pre-LN模式下需要)
        # ============================================================================
        if self.layer_norm_first and layer is None:
            # Pre-LN: 在最后应用层归一化
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        corpus_key=None,
    ):
        """
        特征提取：通过Transformer层序列处理输入特征
        
        Args:
            x: 输入特征 [B, T, C]
            padding_mask: 填充掩码 [B, T]
            tgt_layer: 目标层索引(如果指定，只计算到该层)
            min_layer: 最小层索引(从该层开始记录结果)
            corpus_key: 语料库键(用于适配器选择)
        
        Returns:
            x: 最终特征 [B, T, C]
            layer_results: 各层结果 [(x, z, lr), ...]
        """

        # ============================================================================
        # 1. 填充位置置零 (避免填充位置参与计算)
        # ============================================================================
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # ============================================================================
        # 2. 位置编码 (卷积位置编码)
        # ============================================================================
        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))  # [B, T, C] -> [B, C, T] -> 卷积 -> [B, C, T]
            x_conv = x_conv.transpose(1, 2)            # [B, C, T] -> [B, T, C]
            x = x + x_conv                             # 残差连接：原特征 + 位置编码

        # ============================================================================
        # 3. 输入层归一化 (Post-LN模式)
        # ============================================================================
        if not self.layer_norm_first:
            # Post-LN: 在输入端应用层归一化
            x = self.layer_norm(x)

        # ============================================================================
        # 4. 序列长度填充 (满足模型要求的倍数关系)
        # ============================================================================
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            # 为新增的填充位置创建掩码
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            # 同步填充padding_mask
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        
        # ============================================================================
        # 5. 应用dropout
        # ============================================================================
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ============================================================================
        # 6. 转换为Transformer格式 [B, T, C] -> [T, B, C]
        # ============================================================================
        x = x.transpose(0, 1)

        # ============================================================================
        # 7. 逐层处理
        # ============================================================================
        layer_results = []  # 存储各层的输出结果
        r = None           # 目标层的输出

        for i, layer in enumerate(self.layers):
            # ------------------------------------------------------------------------
            # 7.1 层dropout (随机跳过某些层以提高泛化性)
            # ------------------------------------------------------------------------
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                # ------------------------------------------------------------------------
                # 7.2 处理FSDP包装的层
                # ------------------------------------------------------------------------
                layer_check = layer
                if isinstance(layer, FullyShardedDataParallel):
                    layer_check = layer.unwrapped_module
                
                # ------------------------------------------------------------------------
                # 7.3 根据层类型选择调用方式
                # ------------------------------------------------------------------------
                if (corpus_key is None) or (
                    not isinstance(layer_check, (
                        TransformerSentenceEncoderWithAdapterLayer,
                        )
                    )
                ):
                    # 标准Transformer层或Conformer层
                    x, (z, lr) = layer(
                        x, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                else:
                    # 带适配器的层(需要corpus_key参数)
                    x, (z, lr) = layer(
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        corpus_key=corpus_key,
                    )
                
                # ------------------------------------------------------------------------
                # 7.4 记录层结果
                # ------------------------------------------------------------------------
                if i >= min_layer:
                    layer_results.append((x, z, lr))
            
            # ------------------------------------------------------------------------
            # 7.5 检查是否达到目标层
            # ------------------------------------------------------------------------
            if i == tgt_layer:
                r = x  # 保存目标层的输出
                break

        # ============================================================================
        # 8. 使用目标层输出(如果指定)
        # ============================================================================
        if r is not None:
            x = r

        # ============================================================================
        # 9. 转换回标准格式 [T, B, C] -> [B, T, C]
        # ============================================================================
        x = x.transpose(0, 1)

        # ============================================================================
        # 10. 移除填充 (恢复原始序列长度)
        # ============================================================================
        if pad_length > 0:
            x = x[:, :-pad_length]  # 移除序列末尾的填充

            def undo_pad(a, b, c):
                """移除layer_results中的填充"""
                return (
                    a[:-pad_length],                    # 移除x的填充
                    b[:-pad_length] if b is not None else b,  # 移除z的填充
                    c[:-pad_length],                    # 移除lr的填充
                )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

    def max_positions(self):
        """
        获取编码器支持的最大序列长度
        
        Returns:
            最大位置数(通常用于位置编码的上限)
        """
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """
        升级状态字典以兼容新版本的Fairseq
        
        用于模型版本兼容性，处理参数名称变化等
        
        Args:
            state_dict: 旧版本的状态字典
            name: 模块名称
        
        Returns:
            升级后的状态字典
        """
        return state_dict


class ConformerEncoder(TransformerEncoder):
    """
    Conformer编码器 (卷积增强的Transformer)
    
    Conformer = Transformer + CNN，结合了两种架构的优势：
    1. Transformer: 全局长距离依赖建模
    2. CNN: 局部特征提取和位置不变性
    
    架构创新：
    - 多头注意力 + 卷积模块的串联设计
    - 残差连接和Layer Norm的优化布局
    - Swish激活函数和相对位置编码
    """
    
    def build_encoder_layer(self, args):
        """
        构建Conformer编码器层
        
        Args:
            args: 模型配置
        
        Returns:
            ConformerWav2Vec2EncoderLayer实例
        """
        layer = ConformerWav2Vec2EncoderLayer(
            embed_dim=self.embedding_dim,                         # 嵌入维度
            ffn_embed_dim=args.encoder_ffn_embed_dim,             # 前馈网络维度
            attention_heads=args.encoder_attention_heads,         # 注意力头数
            dropout=args.dropout,                                 # dropout概率
            depthwise_conv_kernel_size=args.depthwise_conv_kernel_size,  # 深度卷积核大小
            activation_fn="swish",                                # Swish激活函数(对语音更有效)
            attn_type=args.attn_type,                             # 注意力类型
            pos_enc_type=args.pos_enc_type,                       # 位置编码类型
            use_fp16=args.fp16,                                   # 半精度训练(仅用于RoPE)
        )
        layer = fsdp_wrap(layer)                                  # FSDP包装
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer)                     # 激活检查点
        return layer

    def __init__(self, args):
        """
        初始化Conformer编码器
        
        Args:
            args: 模型配置
        """
        super().__init__(args)  # 调用父类TransformerEncoder的初始化
        
        # ============================================================================
        # 基础配置
        # ============================================================================
        self.args = args                                          # 保存配置引用
        self.dropout = args.dropout                               # dropout概率
        self.embedding_dim = args.encoder_embed_dim               # 嵌入维度
        self.pos_enc_type = args.pos_enc_type                     # 位置编码类型
        max_source_positions = self.max_positions()               # 最大序列长度

        # ============================================================================
        # 位置编码配置 (支持多种类型)
        # ============================================================================
        if self.pos_enc_type == "rel_pos":
            # 相对位置编码：T5/Transformer-XL风格
            self.embed_positions = RelPositionalEncoding(
                max_source_positions, self.embedding_dim
            )
        elif self.pos_enc_type == "rope":
            # 旋转位置编码：RoFormer风格，无需显式位置嵌入
            self.embed_positions = None
        else:
            # 抛出异常：不支持的位置编码类型
            raise Exception("Unsupported positional encoding type")

        # ============================================================================
        # 编码器层构建
        # ============================================================================
        self.layers = nn.ModuleList(
            [self.build_encoder_layer(args) for _ in range(args.encoder_layers)]
        )
        
        # ============================================================================
        # 归一化和正则化配置
        # ============================================================================
        self.layer_norm_first = args.layer_norm_first               # Pre-LN vs Post-LN
        self.layer_norm = LayerNorm(self.embedding_dim)             # 最终层归一化
        self.layerdrop = args.encoder_layerdrop                     # 层dropout概率

        # ============================================================================
        # 参数初始化
        # ============================================================================
        self.apply(init_bert_params)  # 使用BERT风格的参数初始化

    def extract_features(self, x, padding_mask=None, tgt_layer=None):
        """
        Conformer特征提取 (与标准Transformer的差异)
        
        Args:
            x: 输入特征 [B, T, C]
            padding_mask: 填充掩码 [B, T]
            tgt_layer: 目标层索引
        
        Returns:
            x: 编码特征 [B, T, C]
            layer_results: 各层结果列表
        """
        # ============================================================================
        # 1. 填充位置置零
        # ============================================================================
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        # ============================================================================
        # 2. 转换维度格式 [B, T, C] -> [T, B, C]
        # ============================================================================
        x = x.transpose(0, 1)

        # ============================================================================
        # 3. 位置编码计算 (相对位置编码)
        # ============================================================================
        position_emb = None
        if self.pos_enc_type == "rel_pos":
            # 相对位置编码：基于序列内位置关系
            position_emb = self.embed_positions(x)

        # ============================================================================
        # 4. 输入层归一化 (Post-LN模式)
        # ============================================================================
        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # ============================================================================
        # 5. 输入dropout
        # ============================================================================
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ============================================================================
        # 6. 逐层Conformer处理
        # ============================================================================
        layer_results = []
        r = None
        for i, layer in enumerate(self.layers):
            # 层级dropout：随机跳过某些层
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                # Conformer层前向传播 (注意力+卷积+前馈)
                x, z = layer(
                    x,
                    self_attn_padding_mask=padding_mask,
                    need_weights=False,
                    position_emb=position_emb,  # 传递位置编码
                )
                if tgt_layer is not None:
                    layer_results.append((x, z))
            if i == tgt_layer:
                r = x  # 保存目标层输出
                break

        # ============================================================================
        # 7. 使用目标层输出(如果指定)
        # ============================================================================
        if r is not None:
            x = r

        # ============================================================================
        # 8. 转换回标准格式 [T, B, C] -> [B, T, C]
        # ============================================================================
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    标准Transformer编码器层 (BERT/XLM风格)
    
    架构: Multi-Head Self-Attention + Position-wise FFN
    支持Pre-LN和Post-LN两种归一化模式
    
    结构:
    - Pre-LN:  LN -> Attn -> Residual -> LN -> FFN -> Residual
    - Post-LN: Attn -> Residual -> LN -> FFN -> Residual -> LN
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:
        """
        初始化Transformer编码器层
        
        Args:
            embedding_dim: 嵌入维度 (默认768)
            ffn_embedding_dim: 前馈网络隐层维度 (默认3072)
            num_attention_heads: 注意力头数 (默认8)
            dropout: 主dropout概率
            attention_dropout: 注意力dropout概率
            activation_dropout: 激活函数dropout概率
            activation_fn: 激活函数类型 (relu/gelu等)
            layer_norm_first: 是否使用Pre-LN (默认False为Post-LN)
        """
        super().__init__()
        
        # ============================================================================
        # 基础参数配置
        # ============================================================================
        self.embedding_dim = embedding_dim              # 嵌入维度
        self.dropout = dropout                          # 主dropout概率
        self.activation_dropout = activation_dropout    # 激活函数dropout概率

        # ============================================================================
        # 子模块初始化
        # ============================================================================
        self.activation_fn = utils.get_activation_fn(activation_fn)  # 激活函数
        
        # 多头自注意力机制
        self.self_attn = MultiheadAttention(
            self.embedding_dim,           # Query/Key/Value维度
            num_attention_heads,          # 注意力头数
            dropout=attention_dropout,    # 注意力dropout
            self_attention=True,          # 自注意力模式
        )

        # 三个dropout层 (注意力后、激活函数后、前馈网络后)
        self.dropout1 = nn.Dropout(dropout)              # 注意力输出dropout
        self.dropout2 = nn.Dropout(self.activation_dropout)  # 激活函数dropout
        self.dropout3 = nn.Dropout(dropout)              # 前馈网络输出dropout

        # ============================================================================
        # 归一化策略配置
        # ============================================================================
        self.layer_norm_first = layer_norm_first          # Pre-LN vs Post-LN

        # ============================================================================
        # 注意力子层组件
        # ============================================================================
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)  # 注意力层归一化

        # ============================================================================
        # 前馈网络子层组件
        # ============================================================================
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)  # FFN第一层
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)  # FFN第二层

        # 前馈网络层归一化
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        Transformer编码器层前向传播
        
        支持两种归一化策略：
        - Pre-LN (layer_norm_first=True): 层归一化在子层之前
        - Post-LN (layer_norm_first=False): 层归一化在残差连接之后
        
        Args:
            x: 输入特征 [T, B, C] 或 [B, T, C]
            self_attn_mask: 自注意力掩码
            self_attn_padding_mask: 填充掩码
            need_weights: 是否返回注意力权重
            att_args: 注意力额外参数
        
        Returns:
            x: 输出特征 [T, B, C] 或 [B, T, C]
            (attn, layer_result): 注意力权重和层结果
        """
        # ============================================================================
        # 保存输入用于残差连接
        # ============================================================================
        residual = x

        if self.layer_norm_first:
            # ========================================================================
            # Pre-LN模式: LN -> Attn -> Residual -> LN -> FFN -> Residual
            # ========================================================================
            
            # ------------------------------------------------------------------------
            # 1. 自注意力子层 (Pre-LN)
            # ------------------------------------------------------------------------
            x = self.self_attn_layer_norm(x)  # 注意力前的层归一化
            x, attn = self.self_attn(
                query=x,                      # Query向量
                key=x,                        # Key向量 (自注意力中与Query相同)
                value=x,                      # Value向量
                key_padding_mask=self_attn_padding_mask,  # 填充掩码
                attn_mask=self_attn_mask,     # 注意力掩码
                need_weights=False,           # 不返回注意力权重
            )
            x = self.dropout1(x)              # 注意力输出dropout
            x = residual + x                  # 残差连接

            # ------------------------------------------------------------------------
            # 2. 前馈网络子层 (Pre-LN)
            # ------------------------------------------------------------------------
            residual = x                      # 更新残差连接的基准
            x = self.final_layer_norm(x)      # 前馈网络前的层归一化
            x = self.activation_fn(self.fc1(x))  # 第一层线性变换 + 激活函数
            x = self.dropout2(x)              # 激活函数后dropout
            x = self.fc2(x)                   # 第二层线性变换

            layer_result = x                  # 保存层输出结果

            x = self.dropout3(x)              # 前馈网络输出dropout
            x = residual + x                  # 残差连接
            
        else:
            # ========================================================================
            # Post-LN模式: Attn -> Residual -> LN -> FFN -> Residual -> LN
            # ========================================================================
            
            # ------------------------------------------------------------------------
            # 1. 自注意力子层 (Post-LN)
            # ------------------------------------------------------------------------
            x, attn = self.self_attn(
                query=x,                      # Query向量
                key=x,                        # Key向量
                value=x,                      # Value向量
                key_padding_mask=self_attn_padding_mask,  # 填充掩码
                need_weights=False,           # 不返回注意力权重
            )

            x = self.dropout1(x)              # 注意力输出dropout
            x = residual + x                  # 残差连接
            x = self.self_attn_layer_norm(x)  # 残差连接后的层归一化

            # ------------------------------------------------------------------------
            # 2. 前馈网络子层 (Post-LN)
            # ------------------------------------------------------------------------
            residual = x                      # 更新残差连接的基准
            x = self.activation_fn(self.fc1(x))  # 第一层线性变换 + 激活函数
            x = self.dropout2(x)              # 激活函数后dropout
            x = self.fc2(x)                   # 第二层线性变换

            layer_result = x                  # 保存层输出结果

            x = self.dropout3(x)              # 前馈网络输出dropout
            x = residual + x                  # 残差连接
            x = self.final_layer_norm(x)      # 残差连接后的层归一化

        return x, (attn, layer_result)


class AdapterFast(nn.Module):
    """
    快速适配器模块 (参数高效的迁移学习)
    
    适配器的核心思想：
    1. 在预训练模型中插入少量可训练参数
    2. 冻结主模型，只训练适配器参数
    3. 实现高效的任务特定化
    
    架构: LayerNorm -> Linear -> Activation -> Linear
    相比标准适配器的优化：
    - 使用3D张量存储多个适配器参数，避免ModuleList开销
    - 优化了训练吞吐量
    """
    
    def __init__(self, adapter_num, input_dim, hidden_dim, act_fn):
        """
        初始化快速适配器
        
        Args:
            adapter_num: 适配器数量(支持多任务/多语料)
            input_dim: 输入维度
            hidden_dim: 隐层维度(通常比input_dim小，实现降维)
            act_fn: 激活函数类型
        """
        super().__init__()

        # ============================================================================
        # 基础配置
        # ============================================================================
        self.adapter_num = adapter_num      # 适配器数量
        self.input_dim = input_dim          # 输入维度
        self.hidden_dim = hidden_dim        # 隐层维度
        
        # ============================================================================
        # 适配器权重参数 (3D张量，批量存储)
        # ============================================================================
        # 下投影：input_dim -> hidden_dim (降维)
        self.W_a = nn.Parameter(torch.empty(adapter_num, hidden_dim, input_dim))
        # 上投影：hidden_dim -> input_dim (升维)
        self.W_b = nn.Parameter(torch.empty(adapter_num, input_dim, hidden_dim))
        # 偏置项
        self.b_a = nn.Parameter(torch.empty(adapter_num, hidden_dim))
        self.b_b = nn.Parameter(torch.empty(adapter_num, input_dim))

        # ============================================================================
        # 层归一化参数
        # ============================================================================
        self.ln_W = nn.Parameter(torch.empty(adapter_num, input_dim))  # 缩放参数
        self.ln_b = nn.Parameter(torch.empty(adapter_num, input_dim))  # 偏移参数
        
        # ============================================================================
        # 激活函数配置
        # ============================================================================
        self.act_fn = nn.Identity()  # 默认恒等映射
        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "gelu":
            self.act_fn = nn.GELU()
        elif act_fn == "selu":
            self.act_fn = nn.SELU()
        else:
            raise ValueError(f"不支持的激活函数: {act_fn}")

        self.input_dim = input_dim
        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        """
        参数初始化 (Kaiming初始化 + 均匀分布偏置)
        """
        for ii in range(self.adapter_num):
            # ========================================================================
            # 权重矩阵初始化 (Kaiming均匀分布)
            # ========================================================================
            nn.init.kaiming_uniform_(self.W_a[ii], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_b[ii], a=math.sqrt(5))
            
            # ========================================================================
            # 偏置初始化 (基于fan_in的均匀分布)
            # ========================================================================
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_a[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_a[ii], -bound, bound)
            
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W_b[ii])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b_b[ii], -bound, bound)

        # ========================================================================
        # 层归一化参数初始化
        # ========================================================================
        nn.init.ones_(self.ln_W)   # 缩放参数初始化为1
        nn.init.zeros_(self.ln_b)  # 偏移参数初始化为0

    def forward(self, x, adapter_id):
        """
        适配器前向传播
        
        Args:
            x: 输入特征 [B, T, C]
            adapter_id: 适配器索引 (选择使用哪个适配器)
        
        Returns:
            适配器输出 [B, T, C]
        """
        ii = adapter_id  # 适配器索引
        h = x           # 输入特征
        
        # ============================================================================
        # 适配器计算流程: LN -> Linear -> Act -> Linear
        # ============================================================================
        h = F.layer_norm(h, (self.input_dim, ), self.ln_W[ii], self.ln_b[ii])  # 层归一化
        h = F.linear(h, self.W_a[ii], self.b_a[ii])                            # 下投影(降维)
        h = self.act_fn(h)                                                      # 激活函数
        h = F.linear(h, self.W_b[ii], self.b_b[ii])                            # 上投影(升维)
        
        outputs = h
        return outputs

    def extra_repr(self):
        """返回模块的字符串表示"""
        return ('adapter={}, input_dim={}, hidden_dim={}'.format(
            self.adapter_num, self.input_dim, self.hidden_dim))



class TransformerSentenceEncoderWithAdapterLayer(TransformerSentenceEncoderLayer):
    """
    带适配器的Transformer编码器层 (参数高效迁移学习)
    
    在标准Transformer层的基础上添加适配器模块：
    1. 继承标准TransformerSentenceEncoderLayer的所有功能
    2. 在层输出后插入适配器模块
    3. 支持多语料/多任务的参数高效微调
    
    架构: Transformer Layer + Adapter Layer + Residual Connection
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        adapter_num=201,
        adapter_dim=64,
        adapter_act_fn="relu",
    ) -> None:
        """
        初始化带适配器的Transformer编码器层
        
        Args:
            embedding_dim: 嵌入维度
            ffn_embedding_dim: 前馈网络隐层维度
            num_attention_heads: 注意力头数
            dropout: 主dropout概率
            attention_dropout: 注意力dropout概率
            activation_dropout: 激活函数dropout概率
            activation_fn: 激活函数类型
            layer_norm_first: 是否使用Pre-LN
            adapter_num: 适配器数量(支持多语料)
            adapter_dim: 适配器隐层维度
            adapter_act_fn: 适配器激活函数
        """
        # ============================================================================
        # 调用父类初始化 (标准Transformer层)
        # ============================================================================
        super().__init__(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
        )

        # ============================================================================
        # 适配器模块配置
        # ============================================================================
        self.adapter_num = adapter_num      # 适配器数量
        self.adapter_dim = adapter_dim      # 适配器隐层维度
        
        # 创建快速适配器模块
        self.adapter_layer = AdapterFast(
            adapter_num,            # 适配器数量
            self.embedding_dim,     # 输入维度(与Transformer层输出维度相同)
            self.adapter_dim,       # 隐层维度(通常更小)
            adapter_act_fn         # 激活函数
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
        corpus_key=None,
    ):
        """
        带适配器的Transformer层前向传播
        
        计算流程:
        1. 标准Transformer层前向传播
        2. 根据corpus_key选择适配器
        3. 适配器输出与Transformer输出相加(残差连接)
        
        Args:
            x: 输入特征 [T, B, C] 或 [B, T, C]
            self_attn_mask: 自注意力掩码
            self_attn_padding_mask: 填充掩码
            need_weights: 是否返回注意力权重
            att_args: 注意力额外参数
            corpus_key: 语料库键列表(用于选择适配器)
        
        Returns:
            x: 输出特征 [T, B, C] 或 [B, T, C]
            (attn, layer_result): 注意力权重和层结果
        """
        # ============================================================================
        # 1. 标准Transformer层前向传播
        # ============================================================================
        x, (attn, layer_result) = super().forward(
            x=x,
            self_attn_mask=self_attn_mask,
            self_attn_padding_mask=self_attn_padding_mask,
            need_weights=need_weights,
            att_args=att_args,
        )
        
        # ============================================================================
        # 2. 适配器模块处理
        # ============================================================================
        assert corpus_key is not None, "适配器层需要corpus_key参数"
        assert len(set(corpus_key)) == 1, f"批次内corpus_key必须相同: {corpus_key}"
        
        # 通过适配器处理
        y = self.adapter_layer(x, corpus_key[0])  # 使用第一个corpus_key作为适配器索引
        
        # ============================================================================
        # 3. 残差连接 (适配器 + Transformer输出)
        # ============================================================================
        x = x + y  # 残差连接：原始特征 + 适配器特征
        
        return x, (attn, layer_result)
