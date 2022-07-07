# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(Z):
    """
    实现sigmoid激活函数
    :param Z: 任意形状的numpy数组
    :return: A, cache
    """
    A = 1 / (1 + np.exp(-Z))
    assert (A.shape == Z.shape)
    cache = Z

    return A, cache


def relu(Z):
    """
    实现relu激活函数
    :param Z: 任意形状的numpy数组
    :return: A, cache
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    """
    对单一的sigmoid函数单元实施反向传播
    :param Z: 任意形状的numpy数组
    :param cache: 用于有效计算反向传播
    :return: dZ
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)

    return dZ


def relu_backward(dA, cache):
    """
    对单一的relu函数单元实施反向传播
    :param dZ: 任意形状的numpy数组
    :param cache: 用于有效计算反向传播
    :return: dZ
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ
