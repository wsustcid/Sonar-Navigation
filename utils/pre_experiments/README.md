<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-04-27 11:44:54
 * @LastEditTime: 2020-04-28 11:31:41
 -->

# Overview
There are some pre-experiments which I have done to find proper strategies of the data collection, data representation, sequence length selection, model design, model evaluation, etc.

## Track-I
### Track-I-MLP
In this file, we build a mlp model to predict steering angles using the data collected in Track-I. The following key modules are included in this file.
- data instance construction
- data generator
- sequence length, hidden units number and evaluation metrics selection
- training
- training precess plot
- evaluation
- predict

See `./track-I/Track-I_summary.md` for more details.

### Track-I-CNN
In this file, we build three CNN models (PilotNet, PilotNetv1, PilotNetv2) to predict steering angles using the data collected in Track-I. 
The regression differences caused by three sequence lengths (16,32,48) are also investigated.

**PointNet 结论**：
1. 随着时序的增加，拟合效果越来越好；
2. 48要稍微好于32，在0.01附近，最优值应该是0.008

**PointNetv1 结论：**
全部采用3x3卷积
1. 这里16时序是最好的，说明其实最佳时长也和模型有关系
2. 3种时长的拟合效果都不怎么样，都要差于PilotNet
3. 这里模型对卷积核做了改变，也许是大的卷积核偏向于大的时长，小卷积核偏向于小的时长； 因此输入尺寸也很关键！要注意控制变量，我们基于某个模型作为baseline,要确保和其输入尺寸保持一致！

**PointNetv2结论：**
v1基础上去掉一层卷积一层全连接
1. 效果也没有好到哪里去
2. 说明瞎改没有任何出路，必须首先基于标准模型，建立Baseline, 确定多个分支(1-D, 2-D, 3-D)，在每个分支上进行模型设计；

See `./track-I/Track-I_summary.md` for more details.

### Track-I-ResNet8
In this file, we build a 8 layer ResNet to predict steering angles using the data collected in Track-I. 
The regression differences caused by three sequence lengths (16,32,48) are also investigated.

**ResNet8结论：**
1. 和之前实验基本类似，基本都是48最好，32最差，16和48相差不大，所以可能之间的关系是一个钟型曲线；
2. 但其实从实际值来看，都是可以接受的值；(因此关注重点不应过度放在时序长度选择上)
3. 但如何确定最佳时序是一个值得探讨的问题！
4. 后期利用MLP画出所有时序的拟合精度曲线，看这几个实验是否符合；也可作为辅助证明，证明时序和模型无关
5. 验证集结果在0.015~0.02 之间，

See `./track-I/Track-I_summary.md` for more details.


**总之， 还是要基于 一个准确、标准 的数据集， 进行更加严谨，合理的实验。**


## Track-I-II
### Track-I-II_MLP
In this file, we build a mlp model to predict steering angles using the data collected in Track-I-II. 

**Core lines in data construction:**
```python
with open(csv_path, 'r') as f:
        f_reader = csv.reader(f)
        for line in f_reader:
            # a list which saves time-dependent distance data
            dist = [float(line[j+i*18]) for i in range(length) for j in range(16)]
            # use the newest command as label
            agl  = float(line[17+(length-1)*18])
            #[float(line[17+i*18]) for i in range(length)]
            Data.append(UltrasonicData(dist, agl))

for i in range(len(data_batch)):
            data = data_batch[i]
            if dim == 1:
                dist = data.distance
            elif dim == 2:
                dist = np.array(data.distance)
                # reshape
                dist = dist.reshape(-1,16)
                
                # repeat coloums and rows according to desired width
                dist = dist.repeat(width//16, axis=0)
                dist = dist.repeat(width//16, axis=1)
                
                # add channel axis for the gray scale image
                dist = dist[np.newaxis, :,:]
```

**Loss function Selection**
In this section, we will select a proper loss function (mse, mae, logcosh) to be used in following experiments.

- The sequence length is 16
- the model is a two layer MLP (hidden=2048)
- The RMSE, R-square score, and training time will be used as metrics to evluate the performance of a variety of loss functions.
- The model structure and random seed will be fixed and **K-folod cross validation** will be used to ensure the reliability of results.

results:
```python
# for mse
Final:: mae_ave: 0.0393882576429; mse_ave: 0.00679258274779
Final:: rmse_ave: 0.081454744786; r2_ave: 0.824955055714
# for mae
Final:: mae_ave: 0.0313820576535; mse_ave: 0.00797236233351
Final:: rmse_ave: 0.0878631743257; r2_ave: 0.790437716416
# for logcosh
Final:: mae_ave: 0.0386716082692; mse_ave: 0.00676009133059
Final:: rmse_ave: 0.081341353144; r2_ave: 0.825874607904
```
总结：
- 以上结果显示，使用mse进行训练：训练结果偏向少量数据（较大数值0.5）预测较准，因此MSE极小，MAE和RMSE都很大
- 适用MAE训练，预测结果偏向大部分数据(0附近)，因此MAE值较小，RMSE与MAE差距较大
- 所以问题就转换为，到底是要让大部分数据预测准，还是少部分数据预测准（ＭSE，训练被0.5的数据主导）
- 其实说白了就是你用哪个指标训练，哪个指标的验证效果就好。。。

**Data Sequence Length Selection**
In this section, Multiple MLP models with difference inputs are trained to verify that model performance can be improved by using temporal information.

- The MLP model with one hidden layer takes ultrasonic data as input and predicts angular velocity as control command to steer our robot.
- For the input of model, we tried 1，16, 32，48 four different lengths of time-continuous ultrasonic data sequences， respectively.
- Considering the difference in input dimensions between models, the number of neurons of hidden layer is properly selected to achieve best performance for each model.
- To evluate the performance of difference models, mean absolute error and adjusted R-square sorce are used as evaluation metrics and 5-fold cross-validation is performed in all experiments.

**Another two reasons of using MLP structure**
- Selecting best data sequence length
- As baseline

**Note:** The evaluation results on the validation dataset are not sufficient to demonstrate the feasibility of the model when appling it on a robot. Then we test our model performance in simulation experiments.

- The experiments results show that the MLP model structure cannot successfully drive the robot moving along the corridor environment.
- We suspect that the reason is that the MLP structure cannot perceive (is not aware of) the information of the spatial structure of the input data.

**实验结果：**
```python
# 实验参数：dim = 1； lengths = [1, 16, 32, 48]； n_hiddens = [64, 128, 256, 512, 1024, 2048]
# 最佳结果：
======= 16_1024_5 ======
val_mse_best: 0.0118676724564, epoch: 5.0
val_rmse_best: 0.108302955947, epoch: 5.0
val_mae_best: 0.0743651275869, epoch: 4.8
val_r2_best: 0.695606936727, epoch: 5.0
======= 16_128_5 ======
val_mse_best: 0.0188753482407, epoch: 4.8
val_rmse_best: 0.136441955843, epoch: 4.8
val_mae_best: 0.0975546103077, epoch: 4.8
val_r2_best: 0.501459904058, epoch: 4.8
```

## 总结
以上基于track-I和 track-I-II的预实验，大致体现了一些前期在选择损失函数，最优序列长度过程中的过程。总的来说，部分实验数据让前期无从下手的我从部分细节看到了一点点规律，但因为实验设计的不合理、不充分，这些规律是破碎不成体系的。

## Future Work
在Paper-I 中，我将基于Track-I-I，采用两层全连接，基于早停法的思想选择隐藏单元数量，结合5折交叉验证提高准确度，模型训练同样采用早停法，最后rmse, mae, r2指标分别如下：
- 0.0783 0.0376 0.8455
对比预实验中的结果，说明这次的实验的设计有其合理之处；

这个实验的整体思想是找到针对每种输入序列，找到其最佳模型，构建对标签的最佳描述，最后比较逼近效果； 但基于早停法思想选择隐藏单元数量不一定合理，因为随着隐藏单元数量增加，逼近效果不一定是线性的；

我们可以换种思路，不找到最佳模型，就看最原始的数据能够有多大的标签描述能力！wx+b