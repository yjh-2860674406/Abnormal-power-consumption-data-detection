"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file CNN.py
 * @version 1.0
 * @description 卷积神经网络模型父类
 * @since 2024/1/26 15:19
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
from keras import Sequential

from src.main.model.interface.AbstractCNN import AbstractCNN
from keras.layers import Conv2D, MaxPooling2D, Dense


class CNN(AbstractCNN):
    def __init__(self):
        super().__init__()

    def build_model(
            self,
            hidden_layers: list,
            activation: list,
            loss: str = None,
            optimizer: str = "adam",
            metrics: list = None,
            filters_num: int = 32,
            filter_size: int = 3,
            pool_size: int = 2
    ) -> Sequential:
        # 创建模型
        self.model = Sequential()
        # 添加卷积层
        self.model.add(
            Conv2D(
                filters=filters_num,
                kernel_size=filter_size,
                activation=activation[0],
                input_shape=self.x_train.shape[1:]
            )
        )
        # 添加池化层
        self.model.add(MaxPooling2D(pool_size=pool_size))
        # 添加全连接层
        for i in range(len(hidden_layers)):
            self.model.add(Dense(hidden_layers[i], activation=activation[i+1]))
        # 编译模型
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        return self.model
