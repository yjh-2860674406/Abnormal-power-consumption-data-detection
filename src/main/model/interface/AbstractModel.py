"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file AbstractModel.py
 * @version 1.0
 * @description 
 * @since 2024/1/26 16:13
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
import numpy as np
from keras import Sequential


def format_X_data(train_data: np.ndarray):
    """
    格式化训练数据

    Parameters
    ----------
    train_data: 训练数据

    Returns
    -------

    """
    # 将数据降维
    # 降维前维度
    old_shape = train_data.shape
    # 降维后维度
    new_shape = 1
    for i in (1, len(old_shape) - 1):
        new_shape *= old_shape[i]
    # 降至一维
    train_data = train_data.reshape((-1, new_shape))
    return train_data


class AbstractModel(object):
    def __init__(self):
        self.__x_train: np.ndarray = np.zeros((0, 0))
        self.__y_train: np.ndarray = np.zeros((0, 0))
        self.__x_test: np.ndarray = np.zeros((0, 0))
        self.__y_test: np.ndarray = np.zeros((0, 0))
        self.__model: Sequential = Sequential()

    def train(
            self,
            epochs: int = 1,
            batch_size: int = None,
            verbose: str = "auto"
    ):
        """
        训练数据

        Parameters
        ----------
        epochs: 训练次数
        batch_size: 批次大小
        verbose: 日志显示方式，默认auto

        Returns
        -------

        """
        self.__model.fit(
            self.__x_train,
            self.__y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def evaluate(
            self,
            batch_size: int = None,
            verbose: str = "auto"
    ):
        """
        评估数据

        Parameters
        ----------
        batch_size: 批次大小
        verbose: 日志显示方式，默认auto

        Returns
        -------

        """
        return self.__model.evaluate(
            self.__x_test,
            self.__y_test,
            batch_size=batch_size,
            verbose=verbose
        )

    def predict(
            self,
            x_predict: np.ndarray,
            batch_size: int = None,
            verbose: str = "auto"
    ):
        """
        预测数据

        Parameters
        ----------
        x_predict: 预测数据
        batch_size: 批次大小
        verbose: 日志显示方式，默认auto

        Returns
        -------

        """
        return self.__model.predict(
            x_predict,
            batch_size=batch_size,
            verbose=verbose
        )

    @property
    def x_train(self):
        return self.__x_train

    @property
    def y_train(self):
        return self.__y_train

    @property
    def x_test(self):
        return self.__x_test

    @property
    def y_test(self):
        return self.__y_test

    @property
    def model(self):
        return self.__model

    @x_train.setter
    def x_train(self, x_train: np.ndarray):
        self.__x_train = format_X_data(x_train)

    @y_train.setter
    def y_train(self, y_train: np.ndarray):
        self.__y_train = y_train

    @x_test.setter
    def x_test(self, x_test: np.ndarray):
        self.__x_test = format_X_data(x_test)

    @y_test.setter
    def y_test(self, y_test: np.ndarray):
        self.__y_test = y_test

    @model.setter
    def model(self, mode: Sequential):
        self.__model = mode
