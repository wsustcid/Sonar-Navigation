# 1. 赛道I实验

## 1.1 MLP模型

### 1.1.1 实验一：序列长度+隐藏单元数量+损失函数 多变量组合搜索实验
```python
# 实验参数设置：
batch_size = 128
epochs = 200 # 固定训练轮数
# 待组合变量
lengths = [1, 16, 32, 48]
num_hiddens = [80, 160, 240]
loss_functions = ['mean_squared_error', 'mean_absolute_error', 'mean_squared_logarithmic_error']

# 数据集
"/media/ubuntu16/Documents/datasets/SonarSet/SIM/Track-I/2019-06-06.csv"
Now we have 6509 data after slicing with length 1
The training set size is 5207
The validation set size is 1302

# 基础模型
model.add(Dense(num_hidden, activation='relu', input_dim=length*16))
model.add(Dense(1))

# 优化器及评测指标
optimizer='adam'
metrics=['mse']

# 最佳模型保存位置（已删除）
model_path = "/home/ubuntu16/catkin_ws/src/sonar_navigation/h5/"

# 模型验证
使用最后一轮训练模型分别对训练集和验证集进行验证
```
**结论**：
*关于loss func: （对Loss func的结论可用作小论文结论）*
1. 不适合使用msle作为loss function，因为使用此函数的目标是预测值取log之后与真值取log的值相同，log函数具有缩小大数值之间的差距的功能，因此预测值若差距较大，经过mse放大为大数值差距，最后会造成mse较大。(因为我最后的验证指标是mse --> 不太合理，应选择这三个loss都不强相关的指标)

2. 当length为1和16时，mse最为loss func 效果好于mae; 当length=32 or 48 时， mae好于mse (可能原因是由于短时序时包含较多的回调数据，数值较大，所以mse效果好，长时序回调数据少，大部分都是0，所以mae好)

3. 因为预测值大多在0附近，因此使用MSE会缩小误差，误差小，更新的梯度也小，所以参数很快就不更新了，使得系统收敛较快，所以训练集效果好，但可能此时并没有训练的很好

4. 使用MAE训练，系统收敛较慢,跳动较大，但其代表的是真正的误差；

5. 训练集上效果好(mse指标)的大都使用MSE进行训练，验证集上效果好(mse指标)的大都使用MAE进行训练； 因为mae将小值和大值同等对待，导致对小数值的预测也比较好，因此，转化到mse评测时，对于小数值较多的数据集，结果会比较小；

6. 总之，mse对大数值约束较大训练更smooth，mae对小数值约束大训练多跳变；二者不易说孰优孰劣，根据具体情况选择，一般还是mse好一些； 换言之，二者对训练造成的差距并不是很明显，主要关注的还需是你的模型设计等关键环节！！

**关于length和num_hidden**
1. 总的来说，验证集上排名靠前的均是输入维度较大的，且隐藏层个数较多
2. 但单方面的增加输入维度或隐藏层个数，并不存在明显的线性递进提升关系，还是要细化区间，逐一网格搜索；但对应MLP合理隐藏单元数量的选择估计没有一个很合理的准则。

**详细训练结果**
1. 附录1
2. MLP_grid_search_1.csv



### 1.1.2 实验二：使用MAE训练对典型

```python
# 训练轮次
epochs = 500
# 参数组合
lengths = [1, 16, 32, 48]
num_hiddens = [160, 240, 320, 480]
# 训练指标
loss=mae; metrics=mae
# 最佳模型保存(2020年已删除，可回溯至2019年版本见output/track-I/MLP)
model_path = "/home/ubuntu16/catkin_ws/src/sonar_navigation/h5/"
prefix = 'mlp'+'-'+str(length)+'-'+str(num_hidden)
# 模型验证
saved_model = load_model(model_path+prefix+'-'+'best_model.h5')
```

**结论**
1. 使用单一时刻数据效果最差，证明了时序信息的重要性；
2. 最好前三个结果是 'mlp-32-480-train': 'mlp-48-480-train':'mlp-16-480-train'；说明一般来讲隐藏单元数量与拟合效果在到达过拟合之前基本是越多越好；
3. 最好的前6个结果32占据了4个，说明时序信息并不是越多越好；而是存在一个最佳值的；重要性：时序信息>隐藏单元数量；

总之，目前的这种基于某些参数的实验不够严谨，仅能证明序列长度不一定越大越好，最佳隐藏单元数量和输入维度更是一个非线性关系。需要进一步设计更为合理的实验。

**详细训练过程**
1. 见附录2
2. 及 MLP_grid_search_2.csv



## 1.2 基于PilotNet及修改版本对序列长度的对应实验

```python
# 原始数据读取
distance = D[i:i+length]
# use the value at the final time sequence as a label
angular = agl[i+length-1]
array = np.array(data.distance)
data_repeat = array.repeat(4, axis=0) # 最小尺寸 64x64
data_repeat = data_repeat.repeat(4, axis=1)
X.append(data_repeat)
Y.append(data.angular)          
# batch_size, H, W = X.shape
X = np.array(X)
# add new axis for the gray scale image
X = X[:, np.newaxis, :,:] # 注意，必须加在轴1上
Y = np.array(Y)

# 数据集
"/media/ubuntu16/Documents/datasets/SonarSet/SIM/Track-I/2019-06-06.csv"

# 待选参数
lengths = [16,32,48]

# 训练轮数
epochs = 500

# 模型参数
model = Sequential()
# Normalization [-0.5, 0.5]
model.add(Lambda(lambda x: (x-2.5)/5.0, input_shape=(1, length*4, 64)))
# Conv1
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv2
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv3
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# conv4
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv5
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Flatten
model.add(Flatten())
# FC1
model.add(Dense(100, activation='relu'))
# FC2
model.add(Dense(50, activation='relu'))
# FC3
model.add(Dense(10, activation='relu'))
# output
model.add(Dense(1))
# compile        
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
Total params: 142,219

# 模型保存 （./saved_model/）
prefix = 'cnn'+'-'+str(length)
model_path+prefix+'-'+'best_model.h5'
save_best_only=True

```
**结论**：
1. 随着时序的增加，拟合效果越来越好；
2. 48要稍微好于32，在0.01附近，最优值应该是0.008
3. 这种二维表示方式比我后续3维的表示方式效果要好很多，后期继续验证是模型的原因还是数据表示方式的原因；
**结果：**

<img src=./assets/pilotnet-16.png  width=400><img src=./assets/pilotnet-32.png  width=400>

<img src=./assets/pilotnet-48.png  width=400>

## 1.3 PilotNet-1 

```python
# 数据参数与上一步相同，仅模型不同 

# 模型参数(缩小卷积核大小，全部使用3x3)
model = Sequential()
# Normalization [-0.5, 0.5]
model.add(Lambda(lambda x: (x-2.5)/5.0, input_shape=(1, length*4, 64)))
# Conv1
model.add(Conv2D(24, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv2
model.add(Conv2D(36, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv3
model.add(Conv2D(48, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# conv4
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv5
    model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Flatten
model.add(Flatten())
# FC1
model.add(Dense(100, activation='relu'))
# FC2
model.add(Dense(50, activation='relu'))
# FC3
model.add(Dense(10, activation='relu'))
# output
model.add(Dense(1))
# compile        
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
Total params: 151,563

# 模型保存
prefix = 'mcnn-'+str(length)
model_path+prefix+'-'+'best_model.h5'
save_best_only=True

# 结果：
'mcnn-16-train': 0.013888884615153075,
 'mcnn-16-val': 0.012282487703487277,
 'mcnn-32-train': 0.007835213950602337,
 'mcnn-32-val': 0.016118508856743576,
 'mcnn-48-train': 0.007497835910180583,
 'mcnn-48-val': 0.012401922186836601

```

**结论：**
1. 这里16时序是最好的，说明其实最佳时长也和模型有关系
2. 3种时长的拟合效果都不怎么样，都要差于PilotNet
3. 这里模型对卷积核做了改变，也许是大的卷积核偏向于大的时长，小卷积核偏向于小的时长； 因此输入尺寸也很关键！要注意控制变量



**结果：**

<img src=./assets/pilotnet_1-16.png  width=400><img src=./assets/pilotnet_1-32.png  width=400>

<img src=./assets/pilotnet_1-48.png  width=400 >



## 1.4 PilotNet-2

```python
# 数据参数与上一步相同，仅模型不同 

# 模型参数(缩小卷积核大小，全部使用3x3；去掉最后一层卷积和一层全连接)
model = Sequential()
# Normalization [-0.5, 0.5]
model.add(Lambda(lambda x: (x-2.5)/5.0, input_shape=(1, length*4, 64)))
# Conv1
model.add(Conv2D(24, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv2
model.add(Conv2D(36, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Conv3
model.add(Conv2D(48, kernel_size=(3,3), strides=(2,2), padding='valid', 
                     data_format='channels_first', activation='relu'))
# conv4
model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='valid', 
                     data_format='channels_first', activation='relu'))
# Flatten
model.add(Flatten())
# FC1
model.add(Dense(100, activation='relu'))
# FC3
model.add(Dense(10, activation='relu'))
# output
model.add(Dense(1))
# compile        
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
Total params: 826,885 # 注意少了一个卷积，参数骤增！

# 模型保存
prefix = 'm1cnn-'+str(length)
model_path+prefix+'-'+'best_model.h5'
save_best_only=True

# 结果：
'm1cnn-16-train': 0.012801294273231179,
 'm1cnn-16-val': 0.013154120370745658,
 'm1cnn-32-train': 0.008093868725700304,
 'm1cnn-32-val': 0.018047218304127456,
 'm1cnn-48-train': 0.00886583692044951,
 'm1cnn-48-val': 0.011797982221469282
    
```

**结论：**
1. 效果也没有好到哪里去
2. 说明瞎改没有任何出路，必须首先基于标准模型，建立Baseline, 确定多个分支(1-D, 2-D, 3-D)，在每个分支上进行模型设计；



**结果：**

<img src=./assets/pilotnet_2-16.png  width=400><img src=./assets/pilotnet_2-32.png  width=400>

<img src=./assets/pilotnet_2-48.png  width=400 >



## 1.5 ResNet8

```python
# 原始数据读取
data_repeat = array.repeat(12, axis=0) # 最小尺寸 192x192
data_repeat = data_repeat.repeat(12, axis=1)

# 数据集
"/media/ubuntu16/Documents/datasets/SonarSet/SIM/Track-I/2019-06-06.csv"

# 待选参数
lengths = [16,32,48]

# 训练轮数
epochs = 1000

# 模型参数
# Input
img_input = Input(shape=(img_channel, img_height, img_width))

x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same', data_format='channels_first')(img_input)
x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2], data_format='channels_first')(x1)

# First residual block
x2 = BatchNormalization(axis=1)(x1)
x2 = Activation('relu')(x2)
x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

x2 = BatchNormalization(axis=1)(x2)
x2 = Activation('relu')(x2)
x2 = Conv2D(32, (3, 3), padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same', data_format='channels_first')(x1)
x3 = add([x1, x2])

# Second residual block
x4 = BatchNormalization(axis=1)(x3)
x4 = Activation('relu')(x4)
x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

x4 = BatchNormalization(axis=1)(x4)
x4 = Activation('relu')(x4)
x4 = Conv2D(64, (3, 3), padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same', data_format='channels_first')(x3)
x5 = add([x3, x4])

# Third residual block
x6 = BatchNormalization(axis=1)(x5)
x6 = Activation('relu')(x6)
x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

x6 = BatchNormalization(axis=1)(x6)
x6 = Activation('relu')(x6)
x6 = Conv2D(128, (3, 3), padding='same',
                data_format='channels_first',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same', data_format='channels_first')(x5)
x7 = add([x5, x6])

x = Flatten(data_format='channels_first')(x7)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

# Steering channel
steer = Dense(1)(x)

# Define steering model
model = Model(inputs=[img_input], outputs=[steer])
    
optimizer = optimizers.Adam(decay=1e-5)
    
# compile        
model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
Total params: 313,697

# 模型保存
prefix = 'resnet8'+'-'+str(length)
model_path+prefix+'-'+'best_model.h5'
save_best_only=True

# 结果
'resnet8-16-train': 0.017079434217885138,
 'resnet8-16-val': 0.016340231616050006,
 'resnet8-32-train': 0.01484352839179337,
 'resnet8-32-val': 0.019369325879961253,
 'resnet8-48-train': 0.014933129190467298,
 'resnet8-48-val': 0.015442268829792738
    
resnet8-16-mae: Train: 0.0154573955806, validation: 0.0186197816394
resnet8-32-mae: Train: 0.0129737922456, validation: 0.0158780260012
resnet8-48-mae: Train: 0.0129472742672, validation: 0.0209106724709

            
注： 这几个结果有点乱，暂时不作为参考依据！
            

```

**结论：**
1. 和之前实验基本类似，基本都是48最好，32最差，16和48相差不大，所以可能之间的关系是一个钟型曲线；
2. 但其实从实际值来看，都是可以接受的值；
3. 如何确定最佳时序是一个值得探讨的问题！
4. 后期利用MLP画出所有时序的拟合精度曲线，看这几个实验是否符合；也可作为辅助证明，证明时序和模型无关
5. 验证集结果在0.015~0.02 之间，


# 附录1

```python
FOR THIS MODEL: length is: 1, num_hidden is: 80, loss_function is: mean_squared_error
Total params: 1,441
Trainable params: 1,441
Non-trainable params: 0
_________________________________________________________________
1-80-mean_squared_error: Train: 0.00393989387667, validation: 0.00380872183014
mse: Train: 0.00393989387667, validation: 0.00380872183014
 
FOR THIS MODEL: length is: 1, num_hidden is: 80, loss_function is: mean_absolute_error
Total params: 1,441
Trainable params: 1,441
Non-trainable params: 0
_________________________________________________________________
1-80-mean_absolute_error: Train: 0.0328527365811, validation: 0.0323974657804
mse: Train: 0.00388228606316, validation: 0.00418544844724
 
FOR THIS MODEL: length is: 1, num_hidden is: 80, loss_function is: mean_squared_logarithmic_error
Total params: 1,441
Trainable params: 1,441
Non-trainable params: 0
_________________________________________________________________
1-80-mean_squared_logarithmic_error: Train: 0.0226602169219, validation: 0.0245607210323
mse: Train: 12.5550201178, validation: 12.5786252975
 
FOR THIS MODEL: length is: 1, num_hidden is: 160, loss_function is: mean_squared_error
Total params: 2,881
Trainable params: 2,881
Non-trainable params: 0
_________________________________________________________________
1-160-mean_squared_error: Train: 0.00390105216065, validation: 0.0044699089136
mse: Train: 0.00390105216065, validation: 0.0044699089136
 
FOR THIS MODEL: length is: 1, num_hidden is: 160, loss_function is: mean_absolute_error
Total params: 2,881
Trainable params: 2,881
Non-trainable params: 0
_________________________________________________________________
1-160-mean_absolute_error: Train: 0.0497246042825, validation: 0.0489315599203
mse: Train: 0.00560946523328, validation: 0.00550911815371
 
FOR THIS MODEL: length is: 1, num_hidden is: 160, loss_function is: mean_squared_logarithmic_error
Total params: 2,881
Trainable params: 2,881
Non-trainable params: 0
_________________________________________________________________
1-160-mean_squared_logarithmic_error: Train: 0.0231114120688, validation: 0.0252004425973
mse: Train: 0.820811805129, validation: 0.837759107351
 
FOR THIS MODEL: length is: 1, num_hidden is: 240, loss_function is: mean_squared_error
Total params: 4,321
Trainable params: 4,321
Non-trainable params: 0
_________________________________________________________________
1-240-mean_squared_error: Train: 0.00314754118444, validation: 0.00395990809193
mse: Train: 0.00314754118444, validation: 0.00395990809193
 
FOR THIS MODEL: length is: 1, num_hidden is: 240, loss_function is: mean_absolute_error
Total params: 4,321
Trainable params: 4,321
Non-trainable params: 0
_________________________________________________________________
1-240-mean_absolute_error: Train: 0.0403840491548, validation: 0.042769786343
mse: Train: 0.00435119047179, validation: 0.00524976234883
 
FOR THIS MODEL: length is: 1, num_hidden is: 240, loss_function is: mean_squared_logarithmic_error
Total params: 4,321
Trainable params: 4,321
Non-trainable params: 0
_________________________________________________________________
1-240-mean_squared_logarithmic_error: Train: 0.0235415348783, validation: 0.025653475523
mse: Train: 6.34436625242, validation: 6.37881317139
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6494 data after slicing with length 16
The training set size is 5195
The validation set size is 1299
FOR THIS MODEL: length is: 16, num_hidden is: 80, loss_function is: mean_squared_error
Total params: 20,641
Trainable params: 20,641
Non-trainable params: 0
_________________________________________________________________
16-80-mean_squared_error: Train: 0.00285298884264, validation: 0.00437941113487
mse: Train: 0.00285298884264, validation: 0.00437941113487
 
FOR THIS MODEL: length is: 16, num_hidden is: 80, loss_function is: mean_absolute_error
Total params: 20,641
Trainable params: 20,641
Non-trainable params: 0
_________________________________________________________________
16-80-mean_absolute_error: Train: 0.036939233914, validation: 0.0384403534234
mse: Train: 0.00355629278347, validation: 0.00402063641232
 
FOR THIS MODEL: length is: 16, num_hidden is: 80, loss_function is: mean_squared_logarithmic_error
Total params: 20,641
Trainable params: 20,641
Non-trainable params: 0
_________________________________________________________________
16-80-mean_squared_logarithmic_error: Train: 0.0238877607509, validation: 0.0224922470748
mse: Train: 434.403378296, validation: 432.089630127
 
FOR THIS MODEL: length is: 16, num_hidden is: 160, loss_function is: mean_squared_error
Total params: 41,281
Trainable params: 41,281
Non-trainable params: 0
_________________________________________________________________
16-160-mean_squared_error: Train: 0.00297178578912, validation: 0.00412316981237
mse: Train: 0.00297178578912, validation: 0.00412316981237
 
FOR THIS MODEL: length is: 16, num_hidden is: 160, loss_function is: mean_absolute_error
Total params: 41,281
Trainable params: 41,281
Non-trainable params: 0
_________________________________________________________________
16-160-mean_absolute_error: Train: 0.0656987816095, validation: 0.0681778110564
mse: Train: 0.00639947290765, validation: 0.00718383858912
 
FOR THIS MODEL: length is: 16, num_hidden is: 160, loss_function is: mean_squared_logarithmic_error
Total params: 41,281
Trainable params: 41,281
Non-trainable params: 0
_________________________________________________________________
16-160-mean_squared_logarithmic_error: Train: 0.0228452251293, validation: 0.0215298397467
mse: Train: 476.352296448, validation: 475.245004272
 
FOR THIS MODEL: length is: 16, num_hidden is: 240, loss_function is: mean_squared_error
Total params: 61,921
Trainable params: 61,921
Non-trainable params: 0
_________________________________________________________________
16-240-mean_squared_error: Train: 0.00321892380889, validation: 0.00325588756241
mse: Train: 0.00321892380889, validation: 0.00325588756241
 
FOR THIS MODEL: length is: 16, num_hidden is: 240, loss_function is: mean_absolute_error
Total params: 61,921
Trainable params: 61,921
Non-trainable params: 0
_________________________________________________________________
16-240-mean_absolute_error: Train: 0.0386602201499, validation: 0.038772887364
mse: Train: 0.0039757866296, validation: 0.00425597617868
 
FOR THIS MODEL: length is: 16, num_hidden is: 240, loss_function is: mean_squared_logarithmic_error
Total params: 61,921
Trainable params: 61,921
Non-trainable params: 0
_________________________________________________________________
16-240-mean_squared_logarithmic_error: Train: 0.0242059218232, validation: 0.0229013536125
mse: Train: 735.157345581, validation: 736.404022217
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6478 data after slicing with length 32
The training set size is 5182
The validation set size is 1296
FOR THIS MODEL: length is: 32, num_hidden is: 80, loss_function is: mean_squared_error
Total params: 41,121
Trainable params: 41,121
Non-trainable params: 0
_________________________________________________________________
32-80-mean_squared_error: Train: 0.0127381218364, validation: 0.013408508338
mse: Train: 0.0127381218364, validation: 0.013408508338
 
FOR THIS MODEL: length is: 32, num_hidden is: 80, loss_function is: mean_absolute_error
Total params: 41,121
Trainable params: 41,121
Non-trainable params: 0
_________________________________________________________________
32-80-mean_absolute_error: Train: 0.05257091932, validation: 0.0514354370534
mse: Train: 0.00519537364598, validation: 0.00454555833712
 
FOR THIS MODEL: length is: 32, num_hidden is: 80, loss_function is: mean_squared_logarithmic_error
Total params: 41,121
Trainable params: 41,121
Non-trainable params: 0
_________________________________________________________________
32-80-mean_squared_logarithmic_error: Train: 0.0228306056, validation: 0.0235639892519
mse: Train: 827.486869812, validation: 828.292175293
 
FOR THIS MODEL: length is: 32, num_hidden is: 160, loss_function is: mean_squared_error
Total params: 82,241
Trainable params: 82,241
Non-trainable params: 0
_________________________________________________________________
32-160-mean_squared_error: Train: 0.0029180947633, validation: 0.00348734511063
mse: Train: 0.0029180947633, validation: 0.00348734511063
 
FOR THIS MODEL: length is: 32, num_hidden is: 160, loss_function is: mean_absolute_error
Total params: 82,241
Trainable params: 82,241
Non-trainable params: 0
_________________________________________________________________
32-160-mean_absolute_error: Train: 0.0324504153803, validation: 0.0319076370448
mse: Train: 0.00330131104274, validation: 0.0032008151873
 
FOR THIS MODEL: length is: 32, num_hidden is: 160, loss_function is: mean_squared_logarithmic_error
Total params: 82,241
Trainable params: 82,241
Non-trainable params: 0
_________________________________________________________________
32-160-mean_squared_logarithmic_error: Train: 0.0236754627898, validation: 0.0251780649647
mse: Train: 2184.25942993, validation: 2185.87182617
 
FOR THIS MODEL: length is: 32, num_hidden is: 240, loss_function is: mean_squared_error
Total params: 123,361
Trainable params: 123,361
Non-trainable params: 0
_________________________________________________________________
32-240-mean_squared_error: Train: 0.00409070242895, validation: 0.00465061462019
mse: Train: 0.00409070242895, validation: 0.00465061462019
 
FOR THIS MODEL: length is: 32, num_hidden is: 240, loss_function is: mean_absolute_error
Total params: 123,361
Trainable params: 123,361
Non-trainable params: 0
_________________________________________________________________
32-240-mean_absolute_error: Train: 0.024661767832, validation: 0.0215164422989
mse: Train: 0.00370000166295, validation: 0.00286483881064
 
FOR THIS MODEL: length is: 32, num_hidden is: 240, loss_function is: mean_squared_logarithmic_error
Total params: 123,361
Trainable params: 123,361
Non-trainable params: 0
_________________________________________________________________
32-240-mean_squared_logarithmic_error: Train: 0.0221581512596, validation: 0.0217598583549
mse: Train: 46.1546589851, validation: 47.141268158
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6462 data after slicing with length 48
The training set size is 5169
The validation set size is 1293
FOR THIS MODEL: length is: 48, num_hidden is: 80, loss_function is: mean_squared_error
Total params: 61,601
Trainable params: 61,601
Non-trainable params: 0
_________________________________________________________________
48-80-mean_squared_error: Train: 0.00394177768612, validation: 0.00385549904313
mse: Train: 0.00394177768612, validation: 0.00385549904313
 
FOR THIS MODEL: length is: 48, num_hidden is: 80, loss_function is: mean_absolute_error
Total params: 61,601
Trainable params: 61,601
Non-trainable params: 0
_________________________________________________________________
48-80-mean_absolute_error: Train: 0.0353285984602, validation: 0.0330347111449
mse: Train: 0.00406821224024, validation: 0.00349642271176
 
FOR THIS MODEL: length is: 48, num_hidden is: 80, loss_function is: mean_squared_logarithmic_error
Total params: 61,601
Trainable params: 61,601
Non-trainable params: 0
_________________________________________________________________
48-80-mean_squared_logarithmic_error: Train: 0.0231832462363, validation: 0.0211411776021
mse: Train: 484.209944916, validation: 485.752514648
 
FOR THIS MODEL: length is: 48, num_hidden is: 160, loss_function is: mean_squared_error
Total params: 123,201
Trainable params: 123,201
Non-trainable params: 0
_________________________________________________________________
48-160-mean_squared_error: Train: 0.00727115207119, validation: 0.00693687023595
mse: Train: 0.00727115207119, validation: 0.00693687023595
 
FOR THIS MODEL: length is: 48, num_hidden is: 160, loss_function is: mean_absolute_error
Total params: 123,201
Trainable params: 123,201
Non-trainable params: 0
_________________________________________________________________
48-160-mean_absolute_error: Train: 0.0234205225017, validation: 0.0220134850591
mse: Train: 0.0031105079921, validation: 0.00246171660256
 
FOR THIS MODEL: length is: 48, num_hidden is: 160, loss_function is: mean_squared_logarithmic_error
Total params: 123,201
Trainable params: 123,201
Non-trainable params: 0
_________________________________________________________________
48-160-mean_squared_logarithmic_error: Train: 0.0238209155854, validation: 0.0218684801832
mse: Train: 3575.47188721, validation: 3564.88779297
 
FOR THIS MODEL: length is: 48, num_hidden is: 240, loss_function is: mean_squared_error
Total params: 184,801
Trainable params: 184,801
Non-trainable params: 0
_________________________________________________________________
48-240-mean_squared_error: Train: 0.0060123202391, validation: 0.00575895495713
mse: Train: 0.0060123202391, validation: 0.00575895495713
 
FOR THIS MODEL: length is: 48, num_hidden is: 240, loss_function is: mean_absolute_error
Total params: 184,801
Trainable params: 184,801
Non-trainable params: 0
_________________________________________________________________
48-240-mean_absolute_error: Train: 0.0264960411005, validation: 0.0240719070658
mse: Train: 0.00393600831158, validation: 0.00332125029527
 
FOR THIS MODEL: length is: 48, num_hidden is: 240, loss_function is: mean_squared_logarithmic_error
Total params: 184,801
Trainable params: 184,801
Non-trainable params: 0
_________________________________________________________________
48-240-mean_squared_logarithmic_error: Train: 0.0247437155806, validation: 0.0211280722171
mse: Train: 6100.01395264, validation: 6148.55146484
```



# 附录2：

```Python
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6509 data after slicing with length 1
The training set size is 5207
The validation set size is 1302
==FOR THIS MODEL: length is: 1, num_hidden is: 160==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 160)               2720      
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 161       
=================================================================
Total params: 2,881
Trainable params: 2,881
Non-trainable params: 0
_________________________________________________________________
mlp-1-160-mae: Train: 0.0285213334486, validation: 0.032983440347
 
==FOR THIS MODEL: length is: 1, num_hidden is: 240==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 240)               4080      
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 241       
=================================================================
Total params: 4,321
Trainable params: 4,321
Non-trainable params: 0
_________________________________________________________________
mlp-1-240-mae: Train: 0.0275310723577, validation: 0.029144994542
 
==FOR THIS MODEL: length is: 1, num_hidden is: 320==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_5 (Dense)              (None, 320)               5440      
_________________________________________________________________
dense_6 (Dense)              (None, 1)                 321       
=================================================================
Total params: 5,761
Trainable params: 5,761
Non-trainable params: 0
_________________________________________________________________
mlp-1-320-mae: Train: 0.025659578247, validation: 0.0274252107367
 
==FOR THIS MODEL: length is: 1, num_hidden is: 480==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_7 (Dense)              (None, 480)               8160      
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 481       
=================================================================
Total params: 8,641
Trainable params: 8,641
Non-trainable params: 0
_________________________________________________________________
mlp-1-480-mae: Train: 0.0250151716638, validation: 0.0271089499816
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6494 data after slicing with length 16
The training set size is 5195
The validation set size is 1299
==FOR THIS MODEL: length is: 16, num_hidden is: 160==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_9 (Dense)              (None, 160)               41120     
_________________________________________________________________
dense_10 (Dense)             (None, 1)                 161       
=================================================================
Total params: 41,281
Trainable params: 41,281
Non-trainable params: 0
_________________________________________________________________
mlp-16-160-mae: Train: 0.0198102887487, validation: 0.0207354847342
 
==FOR THIS MODEL: length is: 16, num_hidden is: 240==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_11 (Dense)             (None, 240)               61680     
_________________________________________________________________
dense_12 (Dense)             (None, 1)                 241       
=================================================================
Total params: 61,921
Trainable params: 61,921
Non-trainable params: 0
_________________________________________________________________
mlp-16-240-mae: Train: 0.0214353031712, validation: 0.0241127106361
 
==FOR THIS MODEL: length is: 16, num_hidden is: 320==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_13 (Dense)             (None, 320)               82240     
_________________________________________________________________
dense_14 (Dense)             (None, 1)                 321       
=================================================================
Total params: 82,561
Trainable params: 82,561
Non-trainable params: 0
_________________________________________________________________
mlp-16-320-mae: Train: 0.0196245515719, validation: 0.0210965522565
 
==FOR THIS MODEL: length is: 16, num_hidden is: 480==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_15 (Dense)             (None, 480)               123360    
_________________________________________________________________
dense_16 (Dense)             (None, 1)                 481       
=================================================================
Total params: 123,841
Trainable params: 123,841
Non-trainable params: 0
_________________________________________________________________
mlp-16-480-mae: Train: 0.0198914443376, validation: 0.0199629951268
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6478 data after slicing with length 32
The training set size is 5182
The validation set size is 1296
==FOR THIS MODEL: length is: 32, num_hidden is: 160==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_17 (Dense)             (None, 160)               82080     
_________________________________________________________________
dense_18 (Dense)             (None, 1)                 161       
=================================================================
Total params: 82,241
Trainable params: 82,241
Non-trainable params: 0
_________________________________________________________________
mlp-32-160-mae: Train: 0.0200544395717, validation: 0.0202094046399
 
==FOR THIS MODEL: length is: 32, num_hidden is: 240==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_19 (Dense)             (None, 240)               123120    
_________________________________________________________________
dense_20 (Dense)             (None, 1)                 241       
=================================================================
Total params: 123,361
Trainable params: 123,361
Non-trainable params: 0
_________________________________________________________________
mlp-32-240-mae: Train: 0.0183711249847, validation: 0.0203531616367
 
==FOR THIS MODEL: length is: 32, num_hidden is: 320==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_21 (Dense)             (None, 320)               164160    
_________________________________________________________________
dense_22 (Dense)             (None, 1)                 321       
=================================================================
Total params: 164,481
Trainable params: 164,481
Non-trainable params: 0
_________________________________________________________________
mlp-32-320-mae: Train: 0.0190567307873, validation: 0.0202250580303
 
==FOR THIS MODEL: length is: 32, num_hidden is: 480==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_23 (Dense)             (None, 480)               246240    
_________________________________________________________________
dense_24 (Dense)             (None, 1)                 481       
=================================================================
Total params: 246,721
Trainable params: 246,721
Non-trainable params: 0
_________________________________________________________________
mlp-32-480-mae: Train: 0.0191022302723, validation: 0.0183122788556
 
Starting reading data from csv file...
6509 original data has been read in time order.
Now we have 6462 data after slicing with length 48
The training set size is 5169
The validation set size is 1293
==FOR THIS MODEL: length is: 48, num_hidden is: 160==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_25 (Dense)             (None, 160)               123040    
_________________________________________________________________
dense_26 (Dense)             (None, 1)                 161       
=================================================================
Total params: 123,201
Trainable params: 123,201
Non-trainable params: 0
_________________________________________________________________
mlp-48-160-mae: Train: 0.0203136660159, validation: 0.0229922264814
 
==FOR THIS MODEL: length is: 48, num_hidden is: 240==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_27 (Dense)             (None, 240)               184560    
_________________________________________________________________
dense_28 (Dense)             (None, 1)                 241       
=================================================================
Total params: 184,801
Trainable params: 184,801
Non-trainable params: 0
_________________________________________________________________
mlp-48-240-mae: Train: 0.019572846964, validation: 0.0230008708313
 
==FOR THIS MODEL: length is: 48, num_hidden is: 320==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_29 (Dense)             (None, 320)               246080    
_________________________________________________________________
dense_30 (Dense)             (None, 1)                 321       
=================================================================
Total params: 246,401
Trainable params: 246,401
Non-trainable params: 0
_________________________________________________________________
mlp-48-320-mae: Train: 0.0204875028925, validation: 0.0221650186926
 
==FOR THIS MODEL: length is: 48, num_hidden is: 480==
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_31 (Dense)             (None, 480)               369120    
_________________________________________________________________
dense_32 (Dense)             (None, 1)                 481       
=================================================================
Total params: 369,601
Trainable params: 369,601
Non-trainable params: 0
_________________________________________________________________
mlp-48-480-mae: Train: 0.0187256350648, validation: 0.0185314732604
```

