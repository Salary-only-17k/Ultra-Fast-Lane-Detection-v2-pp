# 改进

UFLD-v2是一个非常优秀的车道线检测模型。也有一些问题。

1 参数量太大，一个模型600+M。

2 不能区分车道线。

针对两个问题作了改进

# 1 数据标注

我用的culane数据做的测试。

首先，需要把culane的数据改成labelme的格式，修改标出来的4条车道线的lable。白实线，白虚线，黄实线，黄虚线。

然后在转回到culane数据的格式。

根据原始train_gt.txt以及culane的生成json文件，生成新的train_gt.txt。比如原始的为

```
/driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 1 0 1
```

生成新得为：

```
/driver_23_30frame/05151649_0422.MP4/00000.jpg /laneseg_label_w16/driver_23_30frame/05151649_0422.MP4/00000.png 1 3 0 2
```

# 2 降低参数量

对模型网络逐层参数量，可以发现，86%（很久之前分析的）的参数集中在最后的两个fc层上。

fc_a+fc_b 拆解成(fc0_a + fc1_a)+(fc0_b + fc1_b).

举例：

```
fc1 = liner(100,200)(...)
fc2 = liner(200,100)(fc1)
```

分解为

```
fc1_a= liner(100,120)(...)
fc1_b= liner(100,80)(...)
fc2_a = liner(120,80)(fc1_a)
fc2_b = liner(80,20)(fc1_b)
```

参数计算

(100*200+200*100)-(100*120+100*80+120*80+80*20)=40000-31200=8800.

## 3 区分车道线类别

判别车道线类别：白实线，白虚线，黄实线，黄虚线，双线。

这里参考的实例分割。有一种实例分割采用的是在语意分割的基础上，添加一个分类头去实现。

如果直接添加在最后一层，将会大大增加模型参数量，经测试，模型到了1.3g大小。

经过测试，backbone为resnet34，在第3个block尾部加入分类器效果最好。

# 4 关于训练

保持和作者的提供的方法一致。注意里面参数，添加了训练方式选择(train_method)，需要根据自己情况修改。

## 4.1 训练技巧

分头训练策略。首先训练原始的网络内容，然后凝固参数，再训练分类器参数。

1 重新训练

2 加载作者的模型

2.1 微调检测车道线头

2.2 只训练检测分类车道线头

## 4.2 快速应用

训练这个模型需要很大的现存，我用的p40和V100训练过，收敛并不快。为了快速应用起来，可以在直接在官方提供的网络上修改，只添加车道线分类部分，加载官方模型，然后冻结参数，只训练车道线分类头部分。这里要求，与训练模型数据相同。