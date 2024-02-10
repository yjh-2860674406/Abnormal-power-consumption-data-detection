"""
 * @author yejinhua  Email:2860674406@qq.com
 * @github https://github.com/yjh-2860674406/Customer-power-consumption-abnormal-data-detection.git
 * @file DataProcessor.py
 * @version 1.0
 * @description 
 * @since 2024/1/25 10:55
 * Copyright (C) 2023-2024 CASEEDER, All Rights Reserved.
 * 注意：本内容仅限于内部传阅，禁止外泄以及用于其他的商业目的
"""
import sqlite3
import pyarrow
import pandas as pd
import numpy as np


class DataProcessor(object):
    def __init__(self):
        self.__path: str = "../../resource/data/"
        self.__data: pd.DataFrame = pd.DataFrame()

    def read_csv(self, csv_name: str):
        self.__path += csv_name
        self.__data = pd.read_csv(self.__path, encoding='gbk')
        return self.__data

    def read_sqlite(self, sqlite_name: str, table_name: str):
        self.__path += sqlite_name
        with sqlite3.connect(self.__path) as con:
            self.__data = pd.read_sql("SELECT * FROM " + table_name, con=con)
        return self.__data

    @property
    def data(self) -> pd.DataFrame:
        return self.__data

    @data.setter
    def data(self, data: pd.DataFrame):
        self.__data = data
