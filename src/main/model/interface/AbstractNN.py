"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file AbstractNN.py
 * @version 1.0
 * @description 神经网络模型父类
 * @since 2024/1/26 10:16
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""

from abc import abstractmethod

from keras.models import Sequential

from src.main.model.interface.AbstractModel import AbstractModel


class AbstractNN(AbstractModel):
    @abstractmethod
    def build_model(
            self,
            hidden_layers: list,
            activation: list,
            loss: str = None,
            optimizer: str = "adam",
            metrics: list = None
    ) -> Sequential:
        """
        构建模型

        Parameters
        ----------
        hidden_layers 隐藏层列表，每个元素代表每一层的神经元个数
        activation 激活函数列表，每个元素代表每一层的激活函数
        loss 损失函数
        optimizer 优化器，默认adam
        metrics 评估指标

        Returns
        -------
        model: 模型
        """
