# Copyright (c) Facebook, Inc. and its affiliates.
# Facebook公司及其附属机构版权所有
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 此源代码基于MIT许可证授权，许可证文件位于项目根目录

# ============================================================================
# Wav2Vec 1.0 模型实现
# 基于对比预测编码(CPC)的自监督语音表示学习模型
# 论文: wav2vec: Unsupervised Pre-training for Speech Recognition (2019)
# ============================================================================

from dataclasses import dataclass, field  # 数据类装饰器，用于定义配置类
import logging  # 日志记录模块
import math  # 数学函数库
from typing import Optional, Tuple  # 类型注解支持
from omegaconf import II  # OmegaConf配置管理库的插值功能
import sys  # 系统相关功能

import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.nn.functional as F  # PyTorch函数式接口
from fairseq.dataclass import ChoiceEnum, FairseqDataclass  # Fairseq数据类和选择枚举
from fairseq.models import BaseFairseqModel, register_model  # Fairseq基础模型类和注册装饰器
from fairseq.modules import (  # Fairseq预定义模块
    Fp32GroupNorm,          # 32位浮点组归一化
    Fp32LayerNorm,          # 32位浮点层归一化
    GumbelVectorQuantizer,  # Gumbel向量量化器
    KmeansVectorQuantizer,  # K-means向量量化器
    TransposeLast,          # 转置最后维度的工具模块
)
from fairseq.tasks import FairseqTask  # Fairseq任务基类
from fairseq.utils import buffered_arange  # 缓冲区范围生成工具


logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# 定义模型配置的枚举选择项
AGGREGATOR_CHOICES = ChoiceEnum(["cnn", "gru"])  # 聚合器类型：卷积神经网络或门控循环单元
PROJECT_FEATURES_CHOICES = ChoiceEnum(["none", "same", "new"])  # 特征投影方式：无投影、复用聚合器、新建聚合器
ACTIVATION_CHOICES = ChoiceEnum(["relu", "gelu"])  # 激活函数选择：ReLU或GELU
VQ_TYPE_CHOICES = ChoiceEnum(["none", "gumbel", "kmeans"])  # 向量量化类型：无量化、Gumbel或K-means


@dataclass
class Wav2VecConfig(FairseqDataclass):
    """
    Wav2Vec模型配置类
    包含模型架构、训练策略、向量量化等所有可配置参数
    """
    
    # ============================================================================
    # 对比预测编码(CPC)相关参数
    # ============================================================================
    prediction_steps: int = field(
        default=12, metadata={"help": "number of steps ahead to predict"}
    )  # 预测步数：模型需要预测未来多少个时间步的表示
    
    sample_distance: Optional[int] = field(
        default=None,
        metadata={
            "help": "sample distance from target. does not work properly with cross-sampling"
        },
    )  # 采样距离：从目标位置采样负样本的距离限制
    
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "num of cross sampled negatives"}
    )  # 跨样本负样本数：从其他样本中采样的负样本数量
    
    num_negatives: int = field(
        default=10, metadata={"help": "num of sampled negatives"}
    )  # 负样本数量：每个正样本对应的负样本数量
    
    # ============================================================================
    # 卷积网络架构参数
    # ============================================================================
    conv_feature_layers: str = field(
        default="[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1), (512, 1, 1)]",
        metadata={
            "help": "convolutional feature extraction layers [(dim, kernel_size, stride), ...]"
        },
    )  # 特征提取卷积层配置：每层的(输出维度, 卷积核大小, 步长)
    
    conv_aggregator_layers: str = field(
        default="[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]",
        metadata={
            "help": "convolutional aggregator layers [(dim, kernel_size, stride), ...]"
        },
    )  # 聚合器卷积层配置：用于时序信息聚合的卷积层参数
    
    # ============================================================================
    # 正则化参数
    # ============================================================================
    dropout: float = field(
        default=0.0, metadata={"help": "dropout to apply within the model"}
    )  # 模型内部的dropout概率
    
    dropout_features: float = field(
        default=0.0, metadata={"help": "dropout to apply to the features"}
    )  # 特征层的dropout概率
    
    dropout_agg: float = field(
        default=0.0, metadata={"help": "dropout to apply after aggregation step"}
    )  # 聚合步骤后的dropout概率
    
    # ============================================================================
    # 聚合器配置
    # ============================================================================
    aggregator: AGGREGATOR_CHOICES = field(
        default="cnn", metadata={"help": "type of aggregator to use"}
    )  # 聚合器类型：使用CNN或GRU进行时序信息聚合
    
    gru_dim: int = field(default=512, metadata={"help": "GRU dimensionality"})
    # GRU聚合器的隐藏维度
    
    # ============================================================================
    # 卷积层配置
    # ============================================================================
    no_conv_bias: bool = field(
        default=False, metadata={"help": "if set, does not learn bias for conv layers"}
    )  # 是否禁用卷积层的偏置项
    
    agg_zero_pad: bool = field(
        default=False,
        metadata={"help": "if set, zero pads in aggregator instead of repl pad"},
    )  # 聚合器是否使用零填充而非复制填充
    
    # ============================================================================
    # 跳跃连接配置
    # ============================================================================
    skip_connections_feat: bool = field(
        default=False,
        metadata={"help": "if set, adds skip connections to the feature extractor"},
    )  # 特征提取器是否使用跳跃连接
    
    skip_connections_agg: bool = field(
        default=True,
        metadata={"help": "if set, adds skip connections to the aggregator"},
    )  # 聚合器是否使用跳跃连接
    
    residual_scale: float = field(
        default=0.5, metadata={"help": "scales residual by sqrt(value)"}
    )  # 残差连接的缩放因子：residual = sqrt(residual_scale) * residual
    
    # ============================================================================
    # 特征处理配置
    # ============================================================================
    log_compression: bool = field(
        default=True,
        metadata={"help": "if set, adds a log compression to feature extractor"},
    )  # 是否对特征进行对数压缩：log(|x| + 1)
    
    balanced_classes: bool = field(
        default=False,
        metadata={"help": "if set, loss is scaled to balance for number of negatives"},
    )  # 是否平衡正负样本的损失权重
    
    project_features: PROJECT_FEATURES_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, features are projected using the (same or new) aggregator"
        },
    )  # 特征投影方式：无投影/复用聚合器/新建投影器
    
    non_affine_group_norm: bool = field(
        default=False, metadata={"help": "if set, group norm is not affine"}
    )  # 组归一化是否使用仿射变换(可学习的scale和shift参数)
    
    # ============================================================================
    # 时序对齐配置
    # ============================================================================
    offset: str = field(
        default="auto",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )  # 时序偏移量：auto表示根据感受野自动计算，否则使用指定值
    
    activation: ACTIVATION_CHOICES = field(
        default="relu",
        metadata={
            "help": "if set to 'auto', it is computed automatically from the receptive field, else set to int value"
        },
    )  # 激活函数类型：ReLU或GELU
    
    # ============================================================================
    # 向量量化(VQ)配置
    # ============================================================================
    vq_type: VQ_TYPE_CHOICES = field(
        default="none", metadata={"help": "which type of quantizer to use"}
    )  # 向量量化器类型：无量化/Gumbel Softmax/K-means
    
    vq_vars: int = field(
        default=320,
        metadata={"help": "project to this many vector quantized variables per group"},
    )  # 每组向量量化变量的数量(码本大小)
    
    vq_groups: int = field(
        default=2, metadata={"help": "number of groups of latent variables"}
    )  # 潜在变量的分组数量(乘积量化的组数)
    
    vq_dim: int = field(
        default=0,
        metadata={
            "help": "uses this dimensionality for quantized vectors. 0 to use model dim // groups"
        },
    )  # 量化向量的维度，0表示使用model_dim // groups
    
    vq_depth: int = field(
        default=1, metadata={"help": "number of layers for vq weight projection"}
    )  # 向量量化权重投影的层数
    
    combine_groups: bool = field(
        default=False, metadata={"help": "if set, variables are shared among groups"}
    )  # 是否在组间共享变量
    
    vq_temp: Tuple[float, float, float] = field(
        default=(2.0, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling with gumbel softmax. should be a tuple of 3 values (start, end, decay)"
        },
    )  # Gumbel Softmax温度参数：(起始温度, 结束温度, 衰减率)
    
    vq_gamma: float = field(
        default=0.25,
        metadata={"help": "gamma parameter for kmeans style vector quantization"},
    )  # K-means向量量化的gamma参数(聚类损失权重)
    
    infonce: bool = II("criterion.infonce")  # 是否使用InfoNCE损失(从损失函数配置中继承)


@register_model("wav2vec", dataclass=Wav2VecConfig)  # 注册模型到Fairseq框架
class Wav2VecModel(BaseFairseqModel):
    """
    Wav2Vec 1.0 主模型类
    
    模型架构：
    1. 卷积特征提取器 (ConvFeatureExtractionModel)
    2. 向量量化器 (可选: GumbelVectorQuantizer/KmeansVectorQuantizer)  
    3. 特征聚合器 (ConvAggegator/GRU)
    4. 对比预测模块 (Wav2VecPredictionsModel)
    
    训练目标：通过对比学习预测未来时间步的特征表示
    """
    
    @classmethod
    def build_model(cls, cfg: Wav2VecConfig, task: FairseqTask):
        """
        模型构建工厂方法
        Args:
            cfg: 模型配置
            task: Fairseq任务实例
        Returns:
            构建好的Wav2VecModel实例
        """
        model = Wav2VecModel(cfg)  # 创建模型实例
        logger.info(model)  # 记录模型结构信息
        return model

    def __init__(self, cfg: Wav2VecConfig):
        """
        初始化Wav2Vec模型
        Args:
            cfg: 模型配置参数
        """
        super().__init__()  # 调用父类初始化

        # 保存预测步数配置
        self.prediction_steps = cfg.prediction_steps
        offset = cfg.offset  # 时序偏移量

        # ============================================================================
        # 1. 激活函数配置
        # ============================================================================
        if cfg.activation == "relu":
            activation = nn.ReLU()  # 使用ReLU激活函数
        elif cfg.activation == "gelu":
            activation = nn.GELU()  # 使用GELU激活函数
        else:
            raise Exception("unknown activation " + cfg.activation)  # 未知激活函数

        # ============================================================================
        # 2. 卷积特征提取器
        # ============================================================================
        feature_enc_layers = eval(cfg.conv_feature_layers)  # 解析卷积层配置字符串
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,  # 卷积层配置列表
            dropout=0.0,  # 特征提取器内部不使用dropout
            log_compression=cfg.log_compression,  # 是否使用对数压缩
            skip_connections=cfg.skip_connections_feat,  # 是否使用跳跃连接
            residual_scale=cfg.residual_scale,  # 残差缩放因子
            non_affine_group_norm=cfg.non_affine_group_norm,  # 组归一化配置
            activation=activation,  # 激活函数
        )
        embed = feature_enc_layers[-1][0]  # 获取最后一层的输出维度作为嵌入维度

        # ============================================================================
        # 3. 向量量化器 (可选)
        # ============================================================================
        self.vector_quantizer = None  # 默认不使用量化器
        if cfg.vq_type == "gumbel":
            # 使用Gumbel Softmax向量量化器
            self.vector_quantizer = GumbelVectorQuantizer(
                dim=embed,  # 输入特征维度
                num_vars=cfg.vq_vars,  # 每组量化变量数量
                temp=cfg.vq_temp,  # 温度参数(起始,结束,衰减)
                groups=cfg.vq_groups,  # 量化分组数
                combine_groups=cfg.combine_groups,  # 是否组合分组
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,  # 量化维度
                time_first=False,  # 时间维度不在第一维
                activation=activation,  # 激活函数
                weight_proj_depth=cfg.vq_depth,  # 权重投影层深度
                weight_proj_factor=2,  # 权重投影因子
            )
        elif cfg.vq_type == "kmeans":
            # 使用K-means向量量化器
            self.vector_quantizer = KmeansVectorQuantizer(
                dim=embed,  # 输入特征维度
                num_vars=cfg.vq_vars,  # 每组量化变量数量
                groups=cfg.vq_groups,  # 量化分组数
                combine_groups=cfg.combine_groups,  # 是否组合分组
                vq_dim=cfg.vq_dim if cfg.vq_dim > 0 else embed,  # 量化维度
                time_first=False,  # 时间维度不在第一维
                gamma=cfg.vq_gamma,  # 聚类损失权重
            )
        else:
            # 验证量化器类型配置
            assert (
                cfg.vq_type == "none" or cfg.vq_type is None
            ), "Unknown quantizer type"

        # ============================================================================
        # 4. 自动计算时序偏移量
        # ============================================================================
        if cfg.offset == "auto":
            # 根据卷积网络的感受野自动计算偏移量
            jin = 0  # 累积步长
            rin = 0  # 感受野大小
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k  # 初始感受野等于第一层卷积核大小
                rin = rin + (k - 1) * jin  # 更新感受野大小
                if jin == 0:
                    jin = stride  # 初始步长
                else:
                    jin *= stride  # 累积步长
            offset = math.ceil(rin / jin)  # 计算最终偏移量

        offset = int(offset)  # 确保偏移量为整数

        # ============================================================================
        # 5. 聚合器构建函数
        # ============================================================================
        def make_aggregator():
            """
            创建特征聚合器
            Returns:
                tuple: (聚合器模块, 输出维度)
            """
            if cfg.aggregator == "cnn":
                # 使用卷积聚合器
                agg_layers = eval(cfg.conv_aggregator_layers)  # 解析聚合器层配置
                agg_dim = agg_layers[-1][0]  # 获取最后一层输出维度
                feature_aggregator = ConvAggegator(
                    conv_layers=agg_layers,  # 卷积层配置
                    embed=embed,  # 输入嵌入维度
                    dropout=cfg.dropout,  # dropout概率
                    skip_connections=cfg.skip_connections_agg,  # 跳跃连接
                    residual_scale=cfg.residual_scale,  # 残差缩放
                    non_affine_group_norm=cfg.non_affine_group_norm,  # 组归一化
                    conv_bias=not cfg.no_conv_bias,  # 卷积偏置
                    zero_pad=cfg.agg_zero_pad,  # 零填充
                    activation=activation,  # 激活函数
                )
            elif cfg.aggregator == "gru":
                # 使用GRU聚合器
                agg_dim = cfg.gru_dim  # GRU隐藏维度
                feature_aggregator = nn.Sequential(
                    TransposeLast(),  # 转置：BxTxC -> BxCxT，为GRU准备
                    nn.GRU(
                        input_size=embed,  # 输入维度
                        hidden_size=agg_dim,  # 隐藏层维度
                        num_layers=1,  # 单层GRU
                        dropout=cfg.dropout,  # dropout
                    ),
                    TransposeLast(deconstruct_idx=0),  # 转置回来，只取output
                )
            else:
                raise Exception("unknown aggregator type " + cfg.aggregator)

            return feature_aggregator, agg_dim

        # 创建聚合器实例
        self.feature_aggregator, agg_dim = make_aggregator()

        # ============================================================================
        # 6. 对比预测模块
        # ============================================================================
        self.wav2vec_predictions = Wav2VecPredictionsModel(
            in_dim=agg_dim,  # 输入维度(聚合器输出维度)
            out_dim=embed,  # 输出维度(特征维度)
            prediction_steps=cfg.prediction_steps,  # 预测步数
            n_negatives=cfg.num_negatives,  # 负样本数量
            cross_sample_negatives=cfg.cross_sample_negatives,  # 跨样本负样本数
            sample_distance=cfg.sample_distance,  # 采样距离
            dropout=cfg.dropout,  # dropout概率
            offset=offset,  # 时序偏移量
            balanced_classes=cfg.balanced_classes,  # 是否平衡类别
            infonce=cfg.infonce,  # 是否使用InfoNCE损失
        )

        # ============================================================================
        # 7. Dropout层
        # ============================================================================
        self.dropout_feats = nn.Dropout(p=cfg.dropout_features)  # 特征dropout
        self.dropout_agg = nn.Dropout(p=cfg.dropout_agg)  # 聚合后dropout

        # ============================================================================
        # 8. 特征投影器 (可选)
        # ============================================================================
        if cfg.project_features == "none":
            self.project_features = None  # 不使用特征投影
        elif cfg.project_features == "same":
            self.project_features = self.feature_aggregator  # 复用聚合器
        elif cfg.project_features == "new":
            self.project_features, _ = make_aggregator()  # 创建新的聚合器作为投影器

    def forward(self, source):
        """
        Wav2Vec模型前向传播
        
        数据流：
        原始音频 -> 特征提取 -> [向量量化] -> dropout -> 聚合 -> dropout -> 对比预测
        
        Args:
            source (Tensor): 原始音频波形 [batch_size, seq_len]
            
        Returns:
            dict: 包含对比预测logits和targets的字典
                - cpc_logits: 对比预测的logits
                - cpc_targets: 对比预测的目标
                - 其他量化器相关的输出(如果使用)
        """
        result = {}  # 存储所有输出结果

        # ============================================================================
        # 1. 特征提取：原始音频 -> 卷积特征 [B, T] -> [B, C, T]
        # ============================================================================
        features = self.feature_extractor(source)
        
        # ============================================================================
        # 2. 向量量化 (可选)：连续特征 -> 离散码本
        # ============================================================================
        if self.vector_quantizer:
            q_res = self.vector_quantizer(features)  # 执行向量量化
            features = q_res["x"]  # 获取量化后的特征
            # 保存量化器的其他输出(如困惑度、损失等)
            for k in q_res.keys():
                if k != "x":
                    result[k] = q_res[k]

        # ============================================================================
        # 3. 特征处理：dropout -> 聚合 -> dropout
        # ============================================================================
        x = self.dropout_feats(features)  # 对特征应用dropout
        x = self.feature_aggregator(x)    # 时序信息聚合(CNN或GRU)
        x = self.dropout_agg(x)           # 对聚合结果应用dropout

        # ============================================================================
        # 4. 特征投影 (可选)：为对比学习准备目标特征
        # ============================================================================
        if self.project_features is not None:
            features = self.project_features(features)  # 投影原始特征作为目标
            
        # ============================================================================
        # 5. 对比预测：预测未来时间步的特征表示
        # ============================================================================
        x, targets = self.wav2vec_predictions(x, features)
        result["cpc_logits"] = x      # 对比预测的logits
        result["cpc_targets"] = targets  # 对比预测的目标

        return result

    def upgrade_state_dict_named(self, state_dict, name):
        """
        升级模型状态字典，用于向后兼容
        Args:
            state_dict: 模型状态字典
            name: 模型名称
        """
        super().upgrade_state_dict_named(state_dict, name)

    def max_positions(self):
        """
        模型支持的最大序列长度
        Returns:
            int: 最大位置数(理论上无限制)
        """
        return sys.maxsize

    def get_logits(self, net_output):
        """
        从网络输出中提取logits
        Args:
            net_output (dict): 网络前向传播输出
        Returns:
            Tensor: CPC logits张量
        """
        logits = net_output["cpc_logits"]
        return logits

    def get_targets(self, sample, net_output):
        """
        从网络输出中提取目标
        Args:
            sample: 输入样本(未使用)
            net_output (dict): 网络前向传播输出
        Returns:
            Tensor: CPC目标张量
        """
        t = net_output["cpc_targets"]
        if isinstance(t, tuple):  # 如果目标是元组，取第一个元素
            t = t[0]
        return t.contiguous()  # 确保内存连续

    def get_target_weights(self, targets, net_output):
        """
        获取目标权重(用于加权损失计算)
        Args:
            targets: 目标张量(未使用)
            net_output (dict): 网络前向传播输出
        Returns:
            Tensor or None: 目标权重张量
        """
        targets = net_output["cpc_targets"]
        if isinstance(targets, tuple) and targets[-1] is not None:
            return targets[-1]  # 返回权重(元组的最后一个元素)
        return None

    def get_extra_losses(self, net_output):
        """
        获取额外的损失项(主要来自向量量化器)
        Args:
            net_output (dict): 网络前向传播输出
        Returns:
            Tensor or None: 额外的损失项
        """
        loss = None
        if "prob_perplexity" in net_output:
            # Gumbel VQ的困惑度损失：鼓励使用更多的码本条目
            loss = net_output["num_vars"] - net_output["prob_perplexity"]
        elif "kmeans_loss" in net_output:
            # K-means VQ的聚类损失
            loss = net_output["kmeans_loss"]

        return loss


def norm_block(is_layer_norm, dim, affine=True):
    """
    创建归一化模块
    Args:
        is_layer_norm (bool): 是否使用层归一化(否则使用组归一化)
        dim (int): 特征维度
        affine (bool): 是否使用仿射变换(可学习的scale和shift)
    Returns:
        nn.Module: 归一化模块
    """
    if is_layer_norm:
        # 层归一化：需要转置以适应1D卷积的输出格式
        mod = nn.Sequential(
            TransposeLast(),  # [B, C, T] -> [B, T, C]
            Fp32LayerNorm(dim, elementwise_affine=affine),  # 在最后一维进行归一化
            TransposeLast(),  # [B, T, C] -> [B, C, T]
        )
    else:
        # 组归一化：直接在通道维度进行，num_groups=1等价于实例归一化
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class ConvFeatureExtractionModel(nn.Module):
    """
    卷积特征提取器
    
    功能：
    1. 将原始音频波形转换为高维特征表示
    2. 通过多层1D卷积逐步提取层次化特征
    3. 可选的跳跃连接和对数压缩
    
    架构：
    - 多层1D卷积 + 组归一化 + 激活函数
    - 可选的残差连接
    - 最终的对数压缩
    """
    
    def __init__(
        self,
        conv_layers,           # 卷积层配置列表
        dropout,               # dropout概率
        log_compression,       # 是否使用对数压缩
        skip_connections,      # 是否使用跳跃连接
        residual_scale,        # 残差缩放因子
        non_affine_group_norm, # 组归一化是否禁用仿射变换
        activation,            # 激活函数
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            """
            创建单个卷积块
            Args:
                n_in (int): 输入通道数
                n_out (int): 输出通道数  
                k (int): 卷积核大小
                stride (int): 步长
            Returns:
                nn.Sequential: 卷积块(Conv1d + Dropout + GroupNorm + Activation)
            """
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),  # 1D卷积，无偏置
                nn.Dropout(p=dropout),  # dropout正则化
                norm_block(
                    is_layer_norm=False,  # 使用组归一化
                    dim=n_out,  # 归一化维度
                    affine=not non_affine_group_norm  # 是否使用仿射变换
                ),
                activation,  # 激活函数(ReLU/GELU)
            )

        # ============================================================================
        # 构建卷积层序列
        # ============================================================================
        in_d = 1  # 初始输入通道数(原始音频为1通道)
        self.conv_layers = nn.ModuleList()
        
        # 根据配置创建每一层卷积
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim  # 更新下一层的输入通道数

        # 保存配置参数
        self.log_compression = log_compression          # 是否使用对数压缩
        self.skip_connections = skip_connections        # 是否使用跳跃连接
        self.residual_scale = math.sqrt(residual_scale) # 残差缩放因子(开平方)

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 原始音频波形 [batch_size, seq_len]
        Returns:
            Tensor: 提取的特征 [batch_size, feature_dim, seq_len']
        """
        # ============================================================================
        # 1. 维度扩展：BxT -> Bx1xT (添加通道维度)
        # ============================================================================
        x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        # ============================================================================
        # 2. 逐层卷积特征提取
        # ============================================================================
        for conv in self.conv_layers:
            residual = x  # 保存残差连接的输入
            x = conv(x)   # 卷积变换
            
            # 残差连接 (如果启用且通道数匹配)
            if self.skip_connections and x.size(1) == residual.size(1):
                # 处理时序长度不匹配的情况
                tsz = x.size(2)      # 当前输出的时序长度
                r_tsz = residual.size(2)  # 残差的时序长度
                
                # 下采样残差以匹配当前输出的时序长度
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                
                # 残差连接并缩放
                x = (x + residual) * self.residual_scale

        # ============================================================================
        # 3. 对数压缩 (可选)：log(|x| + 1)
        # ============================================================================
        if self.log_compression:
            x = x.abs()   # 取绝对值，避免负数
            x = x + 1     # 加1防止log(0)
            x = x.log()   # 对数变换，压缩动态范围

        return x


class ZeroPad1d(nn.Module):
    """
    1D零填充模块
    对1D张量进行左右零填充
    """
    def __init__(self, pad_left, pad_right):
        """
        Args:
            pad_left (int): 左侧填充数量
            pad_right (int): 右侧填充数量
        """
        super().__init__()
        self.pad_left = pad_left    # 左侧填充数量
        self.pad_right = pad_right  # 右侧填充数量

    def forward(self, x):
        """
        Args:
            x (Tensor): 输入张量 [B, C, T]
        Returns:
            Tensor: 填充后的张量 [B, C, T + pad_left + pad_right]
        """
        return F.pad(x, (self.pad_left, self.pad_right))  # 在最后一维进行填充


class ConvAggegator(nn.Module):
    """
    卷积聚合器
    
    功能：
    1. 对特征提取器的输出进行时序信息聚合
    2. 通过因果卷积保持时序关系
    3. 使用跳跃连接增强信息流
    
    架构：
    - 多层因果1D卷积
    - 残差连接 + 维度投影
    - 层归一化 + dropout
    """
    
    def __init__(
        self,
        conv_layers,           # 卷积层配置列表
        embed,                 # 输入嵌入维度
        dropout,               # dropout概率
        skip_connections,      # 是否使用跳跃连接
        residual_scale,        # 残差缩放因子
        non_affine_group_norm, # 组归一化是否禁用仿射变换
        conv_bias,             # 是否使用卷积偏置
        zero_pad,              # 是否使用零填充(否则使用复制填充)
        activation,            # 激活函数
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            """
            创建单个聚合卷积块
            Args:
                n_in (int): 输入通道数
                n_out (int): 输出通道数
                k (int): 卷积核大小
                stride (int): 步长
            Returns:
                nn.Sequential: 聚合卷积块
            """
            # ================================================================
            # 因果填充：确保未来信息不泄露
            # ================================================================
            ka = k // 2     # 左侧填充
            kb = ka - 1 if k % 2 == 0 else ka  # 右侧填充(偶数核需要-1)

            # 选择填充方式：零填充 vs 复制填充
            pad = (
                ZeroPad1d(ka + kb, 0) if zero_pad 
                else nn.ReplicationPad1d((ka + kb, 0))
            )

            return nn.Sequential(
                pad,  # 填充
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias),  # 1D卷积
                nn.Dropout(p=dropout),  # dropout正则化
                norm_block(False, n_out, affine=not non_affine_group_norm),  # 组归一化
                activation,  # 激活函数
            )

        # ============================================================================
        # 构建聚合器层
        # ============================================================================
        in_d = embed  # 初始输入维度
        self.conv_layers = nn.ModuleList()      # 卷积层列表
        self.residual_proj = nn.ModuleList()    # 残差投影层列表
        
        for dim, k, stride in conv_layers:
            # 残差投影：处理维度不匹配的情况
            if in_d != dim and skip_connections:
                # 需要1x1卷积进行维度变换
                self.residual_proj.append(nn.Conv1d(in_d, dim, 1, bias=False))
            else:
                # 维度匹配，不需要投影
                self.residual_proj.append(None)

            # 添加聚合卷积块
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim  # 更新下一层的输入维度
            
        # 注意：这里没有将conv_layers转为Sequential，保持ModuleList以配合residual_proj使用
        self.skip_connections = skip_connections              # 是否使用跳跃连接
        self.residual_scale = math.sqrt(residual_scale)       # 残差缩放因子

    def forward(self, x):
        """
        前向传播
        Args:
            x (Tensor): 输入特征 [batch_size, embed_dim, seq_len]
        Returns:
            Tensor: 聚合后的特征 [batch_size, output_dim, seq_len]
        """
        # ============================================================================
        # 逐层聚合处理
        # ============================================================================
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x  # 保存残差连接的输入
            x = conv(x)   # 卷积聚合
            
            # 跳跃连接处理
            if self.skip_connections:
                if rproj is not None:
                    # 维度不匹配，需要投影
                    residual = rproj(residual)
                # 残差连接并缩放
                x = (x + residual) * self.residual_scale
        return x


class Wav2VecPredictionsModel(nn.Module):
    """
    Wav2Vec对比预测模块
    
    功能：
    1. 实现对比预测编码(CPC)的核心逻辑
    2. 预测未来时间步的特征表示
    3. 通过负采样进行对比学习
    
    工作流程：
    1. 将聚合器输出投影到多个预测步骤
    2. 采样负样本进行对比学习
    3. 计算预测logits和目标targets
    """
    
    def __init__(
        self,
        in_dim,                  # 输入维度(聚合器输出维度)
        out_dim,                 # 输出维度(目标特征维度)
        prediction_steps,        # 预测步数
        n_negatives,            # 负样本数量
        cross_sample_negatives,  # 跨样本负样本数量
        sample_distance,         # 采样距离限制
        dropout,                 # dropout概率
        offset,                  # 时序偏移量
        balanced_classes,        # 是否平衡类别权重
        infonce,                 # 是否使用InfoNCE损失
    ):
        super().__init__()

        # 保存配置参数
        self.n_negatives = n_negatives                          # 负样本数量
        self.cross_sample_negatives = cross_sample_negatives    # 跨样本负样本数量
        self.sample_distance = sample_distance                  # 采样距离限制
        
        # ============================================================================
        # 预测投影层：将聚合特征投影到多个预测步骤
        # ============================================================================
        self.project_to_steps = nn.ConvTranspose2d(
            in_dim, out_dim, (1, prediction_steps)
        )  # 2D转置卷积：[B,in_dim,T,1] -> [B,out_dim,T,prediction_steps]
        
        self.dropout = nn.Dropout(p=dropout)    # dropout正则化
        self.offset = offset                     # 时序偏移量
        self.balanced_classes = balanced_classes # 是否平衡类别权重
        self.infonce = infonce                   # 是否使用InfoNCE损失

    def sample_negatives(self, y):
        """
        负样本采样函数
        
        目标：为对比学习采样负样本
        策略：
        1. 同批次内采样 (避免采样到真实目标)
        2. 跨批次采样 (增加负样本多样性)
        
        Args:
            y (Tensor): 目标特征 [batch_size, feature_dim, seq_len]
        Returns:
            Tensor: 负样本 [n_negatives, batch_size, feature_dim, seq_len]
        """
        bsz, fsz, tsz = y.shape  # 批次大小、特征维度、序列长度

        # ============================================================================
        # 1. 数据重组：方便负样本采样
        # ============================================================================
        y = y.transpose(0, 1)  # [B,C,T] -> [C,B,T]
        y = y.contiguous().view(fsz, -1)  # [C,B,T] -> [C,B*T] 展平为特征池

        # ============================================================================
        # 2. 计算采样范围
        # ============================================================================
        cross_high = tsz * bsz  # 跨样本采样的总范围
        # 同样本内采样范围：受sample_distance限制
        high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)
        assert high > 1, "采样范围必须大于1"

        # 预分配负样本索引
        neg_idxs = torch.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))

        with torch.no_grad():  # 不需要梯度计算
            # ====================================================================
            # 3. 同批次内负样本采样
            # ====================================================================
            if self.n_negatives > 0:
                # 创建时间步索引：[0,1,2,...,T-1] 重复 n_negatives 次
                tszs = (
                    buffered_arange(tsz)           # [0,1,2,...,T-1]
                    .unsqueeze(-1)                 # [T,1]
                    .expand(-1, self.n_negatives)  # [T,N]
                    .flatten()                     # [T*N]
                )

                # 随机采样负样本索引，避免采样到真实目标
                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * tsz)
                )
                # 如果采样到的索引 >= 当前时间步，则+1跳过真实目标
                neg_idxs[neg_idxs >= tszs] += 1

            # ====================================================================
            # 4. 跨批次负样本采样
            # ====================================================================
            if self.cross_sample_negatives > 0:
                # 跨样本时间步索引
                tszs = (
                    buffered_arange(tsz)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                # 从整个批次池中采样
                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * tsz),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        # ============================================================================
        # 5. 调整索引以匹配批次结构
        # ============================================================================
        if self.n_negatives > 0:
            # 为每个批次调整索引偏移
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            # 只使用跨样本负样本
            neg_idxs = cross_neg_idxs

        # 合并同批次和跨批次负样本
        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        # ============================================================================
        # 6. 提取负样本特征
        # ============================================================================
        negs = y[..., neg_idxs.view(-1)]  # 根据索引提取负样本
        negs = negs.view(
            fsz, bsz, self.n_negatives + self.cross_sample_negatives, tsz
        ).permute(
            2, 1, 0, 3
        )  # 重组为 [N_negatives, Batch, Features, Time]

        return negs

    def forward(self, x, y):
        """
        对比预测前向传播
        
        核心逻辑：
        1. 将聚合特征投影到多个预测步骤
        2. 采样负样本构建对比学习目标
        3. 计算预测与目标的相似度
        4. 返回logits和labels用于损失计算
        
        Args:
            x (Tensor): 聚合后的上下文特征 [B, agg_dim, T]
            y (Tensor): 目标特征 [B, feature_dim, T]
        Returns:
            tuple: (predictions, labels)
                - predictions: 预测logits
                - labels: 对比学习的标签
        """

        # ============================================================================
        # 1. 多步预测投影：上下文 -> 预测
        # ============================================================================
        x = x.unsqueeze(-1)              # [B,C,T] -> [B,C,T,1]，添加预测步维度
        x = self.project_to_steps(x)     # [B,C,T,1] -> [B,out_dim,T,steps] 投影到多步预测
        x = self.dropout(x)              # dropout正则化

        # ============================================================================
        # 2. 构建对比学习目标：正样本 + 负样本
        # ============================================================================
        negatives = self.sample_negatives(y)  # 采样负样本 [N_neg, B, C, T]
        y = y.unsqueeze(0)                     # 正样本 [1, B, C, T]
        targets = torch.cat([y, negatives], dim=0)  # 合并目标 [1+N_neg, B, C, T]

        # ============================================================================
        # 3. 计算预测相似度
        # ============================================================================
        copies = targets.size(0)              # 目标总数 = 1 + n_negatives
        bsz, dim, tsz, steps = x.shape       # 预测张量的形状
        steps = min(steps, tsz - self.offset) # 有效预测步数（考虑偏移）

        # 预分配预测结果张量
        predictions = x.new(
            bsz * copies * (tsz - self.offset + 1) * steps
            - ((steps + 1) * steps // 2) * copies * bsz
        )  # 考虑因果约束后的实际预测数量
        
        # ============================================================================
        # 4. 准备标签和权重
        # ============================================================================
        if self.infonce:
            # InfoNCE：正样本标签为0，其余为负样本
            labels = predictions.new_full(
                (predictions.shape[0] // copies,), 0, dtype=torch.long
            )
        else:
            # 二分类：正样本为1，负样本为0
            labels = torch.zeros_like(predictions)
            
        # 类别平衡权重
        weights = (
            torch.full_like(labels, 1 / self.n_negatives)
            if self.balanced_classes and not self.infonce
            else None
        )

        # ============================================================================
        # 5. 逐步预测计算
        # ============================================================================
        start = end = 0
        for i in range(steps):
            offset = i + self.offset  # 当前预测的时间偏移
            end = start + (tsz - offset) * bsz * copies  # 当前步的结束位置
            
            if self.infonce:
                # InfoNCE：计算预测和所有目标的点积
                predictions[start:end] = torch.einsum(
                    "bct,nbct->tbn",           # 爱因斯坦求和：批次×特征×时间 与 目标×批次×特征×时间
                    x[..., :-offset, i],       # 预测特征（去掉未来部分）
                    targets[..., offset:]      # 目标特征（从offset开始）
                ).flatten()
            else:
                # 二分类：计算每个目标的预测分数
                pos_num = (end - start) // copies  # 正样本数量
                predictions[start:end] = torch.einsum(
                    "bct,nbct->nbt", x[..., :-offset, i], targets[..., offset:]
                ).flatten()
                
                # 设置正样本标签为1.0
                labels[start : start + pos_num] = 1.0
                if weights is not None:
                    weights[start : start + pos_num] = 1.0
            start = end
            
        # 验证预测数量正确性
        assert end == predictions.numel(), f"预测数量不匹配: {end} != {predictions.numel()}"

        # ============================================================================
        # 6. 格式化输出
        # ============================================================================
        if self.infonce:
            # InfoNCE：重组为 [样本数, 目标数] 用于交叉熵损失
            predictions = predictions.view(-1, copies)
        else:
            # 二分类：添加权重信息
            if weights is not None:
                labels = (labels, weights)

        return predictions, labels
