"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file NN.py
 * @version 1.0
 * @description 神经网络模型实现类
 * @since 2024/1/26 16:02
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
from keras.layers import Dense
from keras import Sequential
from src.main.model.interface.AbstractNN import AbstractNN


class NN(AbstractNN):
    def __init__(self):
        super().__init__()

    def build_model(
            self,
            hidden_layers: list,
            activation: list,
            loss: str = None,
            optimizer: str = "adam",
            metrics: list = None
    ) -> Sequential:
        # 创建模型
        self.model = Sequential()
        # 添加输入层以及第一层隐藏层
        self.model.add(Dense(hidden_layers[0], input_shape=(self.x_train.shape[1],), activation=activation[0]))
        # 添加其他隐藏层
        for i in range(1, len(hidden_layers) - 1):
            self.__model.add(Dense(hidden_layers[i], activation=activation[i]))
        # 添加输出层
        self.model.add(Dense(self.y_train.shape[1], activation=activation[-1]))
        # 编译模型
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        # 返回模型
        return self.model
