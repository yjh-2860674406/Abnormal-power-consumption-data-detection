"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file AbstractCNN.py
 * @version 1.0
 * @description 
 * @since 2024/1/26 16:13
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
from abc import abstractmethod
from keras import Sequential
from src.main.model.interface.AbstractModel import AbstractModel


class AbstractCNN(AbstractModel):
    @abstractmethod
    def build_model(
            self,
            hidden_layers: list,
            activation: list,
            loss: str = None,
            optimizer: str = "adam",
            metrics: list = None,
            filters_num: int = 32,
            filter_size: int = 3,
            pool_size: int = 2,
    ) -> Sequential:
        """
        构建卷积神经网络模型
        :param hidden_layers: 隐藏层神经元个数
        :param activation: 激活函数
        :param loss: 损失函数
        :param optimizer: 优化器
        :param metrics: 评估指标
        :param filters_num: 卷积核个数
        :param filter_size: 卷积核大小
        :param pool_size: 池化大小
        :return: 卷积神经网络模型
        """

