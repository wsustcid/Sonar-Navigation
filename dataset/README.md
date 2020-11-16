<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-10-31 20:22:09
 * @LastEditTime: 2020-11-16 17:22:58
 * @Description:  
-->
# Spatio-Temporal Ultrasonic Navigation Dataset
This is a benchmark ultrasonic datasets that consist of spatial and temporal data to verify the usability of spatial and temporal ultrasonic cues.

## Download
This dataset can be downloaded by this [link](https://rec.ustc.edu.cn/share/4c187020-27ed-11eb-92b0-3fa94912241b)


## Collection Log

### 2019-06-06.csv
采集目标：外圈单方向角速度预测
**序号1-1976：**
- 外圈逆时针完整一圈
- 定速0.4

**序号1977-6509：**
- 外圈逆时针8圈，仅在转弯处收集数据（为保证数据均衡）
- 速度0.4

<img src=assets/20190606-1.png width=300/> <img src=assets/20190606-2-9.png width=300/>

数据统计：（可使用excel-插入-图表查看）

- 总计9圈，36次转弯，4段完整直行
- w=0 共计4433个

### 2019-06-28.csv
同时采集Track-I 和 Track-II正反方向行驶数据，包含正常行驶数据和转向数据，尽量做到了转向数据和直行数据的均衡。

数据样本个数：
- -1: 40262
- -16：40154
- -32：40069
- -48： 39965


### 2019-07-03.csv
是在2019-06-28的基础上继续采集车道线保持数据（小角度回调）和避障数据（大角度回调）合并而成：
