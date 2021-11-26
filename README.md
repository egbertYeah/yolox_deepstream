# yolox_deepstream

#### 介绍
将YoloX部署到deepstream中，进行推理加速。

#### 测试环境
- tensorrt=7.2.1
- deepstream=5.1


#### 使用方法

1.  基于自己的训练集训练得到pytorch权重
2.  将pytorch权重转成onnx格式
3.  onnx转tensorrt的engine文件
4.  编写后处理插件，进行deepstream推理

#### 具体使用方法
针对yoloX在deepstream上部署的细节可以参考博客[【Deepstream之YoloX部署】](https://blog.csdn.net/hello_dear_you/article/details/121558358)

