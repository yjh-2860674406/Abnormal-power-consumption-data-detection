"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file test.py
 * @version 1.0
 * @description 
 * @since 2024/1/26 14:22
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
from src.main.model.NN import NN
from keras import datasets
from keras import utils

# 从自带数据集载入数据
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

y_train = utils.to_categorical(y_train, num_classes=10)
y_test = utils.to_categorical(y_test, num_classes=10)

# 创建神经网络模型
nn = NN()
nn.x_train = x_train.astype("float32") / 255
nn.y_train = y_train
nn.x_test = x_test.astype("float32") / 255
nn.y_test = y_test
nn.build_model(
    hidden_layers=[512, 256, 128, 64, 32],
    activation=["relu", "relu", "relu", "relu", "sigmoid"],
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
nn.train(epochs=40, batch_size=32)

nn.model.summary()

nn.evaluate(batch_size=32)
