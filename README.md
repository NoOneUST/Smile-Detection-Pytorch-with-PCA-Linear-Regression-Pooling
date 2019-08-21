# Smile Detection (Pytorch)

#### This is an original and useful `smile detection` program implemented with Pytorch.
---
#### `AdvancedLinearRegression.py`

`AdvancedLinearRegression.py` is the main file implemented with Pytorch which you can directly run to know whether the man on a given picture is smiling or not. 

This algorithm is based on PCA & Linear Regression & Pooling. It can at least acheive ```76.35735869407654%``` accuracy on our dataset which is contructed on CelebA. 

Of course I think it is absolutely possible for you to obtain further accuracy improvement by simply adjusting the PCA coefficient or pooling size or LSR(least square regression) method.

#### `LinearRegressionOnly.py`

`LinearRegressionOnly.py` is the simplified version of `AdvancedLinearRegression.py` which is only based on linear regression. 

It will no longer be maintained thus please directly choose the above one instead of this.

#### `Result`
```
===> Loading Data...
processed: 1000 / 20260
processed: 2000 / 20260
processed: 3000 / 20260
processed: 4000 / 20260
processed: 5000 / 20260
processed: 6000 / 20260
processed: 7000 / 20260
processed: 8000 / 20260
processed: 9000 / 20260
processed: 10000 / 20260
processed: 11000 / 20260
processed: 12000 / 20260
processed: 13000 / 20260
processed: 14000 / 20260
processed: 15000 / 20260
processed: 16000 / 20260
processed: 17000 / 20260
processed: 18000 / 20260
processed: 19000 / 20260
processed: 20000 / 20260

===> Doing PCA...

===> Saving models...
```

![1](https://github.com/NoOneUST/Smile-Detection-Pytorch-with-PCA-Linear-Regression-Pooling/blob/master/image/1.png)

```
No, I guess he is not smiling.
You guess correctly.
```

![2](https://github.com/NoOneUST/Smile-Detection-Pytorch-with-PCA-Linear-Regression-Pooling/blob/master/image/2.png)

```
Yes, I guess he is smiling.
You guess correctly.
```

![3](https://github.com/NoOneUST/Smile-Detection-Pytorch-with-PCA-Linear-Regression-Pooling/blob/master/image/3.png)

```
No, I guess he is not smiling.
You did not guess correctly.

Accuracy on backtesting: 76.35735869407654 %
Loss on backtesting: 0.7765066623687744
Loss on Trainingset: 0.3217602074146271
```

