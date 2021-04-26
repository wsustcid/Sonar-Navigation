<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-04-26 16:34:53
 * @LastEditTime: 2021-04-26 15:57:10
 -->

# Spatio-Temporal Ultrasonic Dataset: Learning Driving from Spatial and Temporal Ultrasonic Cues

## 1. Introduction
<div align=center> <img src=./doc/figure/cover.png width=600, height=300 /></div>

Recent works have proved that combining spatial and temporal visual cues can significantly improve the performance of various vision-based robotic systems. However, for the ultrasonic sensors used in most robotic tasks, there is a lack of benchmark ultrasonic datasets that consist of spatial and temporal data to verify the usability of spatial and temporal ultrasonic cues. 

In this project, we collected a **Spatio-Temporal Ultrasonic Navigation Dataset (STUND)** to promote the use of ultrasonic sensors in robot navigation research. Based on this dataset, we trained an end-to-end driving model to autonomously drive the mobile robot in unknown indoor environments. This is inspired by the phenomenon that bats are able to navigate in complex environments by extracting spatial and temporal cues from sonar echoes.

## 2. Requirements
- Ubuntu 16.04
- ROS kinetic
- python 2.7
- tensorflow 1.18
- keras 
- numpy

## 3. Dataset
  - To download our STUND, see the `./dataset/README.md` file for more dateils.

## 4. Methodology
In this project, 
  - we first propose a Spatio-Temporal Ultrasonic Navigation Dataset (STUND), which aims to develop the ability of ultrasonic sensors by mining spatial and temporal information from multiple ultrasonic measurements.
  - Then we investigate various representations of the ST ultrasonic data to uncover hidden patterns and structures within data, which is beneficial for subsequent tasks
  - Finally, we present an end-to-end driving model that learns driving policies by extracting spatial and temporal ultrasonic cues from the STUD. 

With the help of our STUND and our driving model, various indoor navigation tasks can be tackled in a cheaper and efficient way, which is unachievable by simply using the existing ultrasonic datasets. 

## 5. Train & Evaluation
```python
# train
python run.py --xx ...

# evaluation
python evaluate.py
```

## 6. Drive
To drive your robot by the neural network, your should deploy the well-trained driving model on the robot controller. The driving model will output driving velocity and steering angles while perceiving the environmental data using ultrasonic sensors.

```python
# active ROS ENV
roscore
ros run xxx

# start your sonar sensor to public 2D ultrasonic point clouds

# start your robot to receive driving commands

# run the driving model
roslaunch sonar_navigation drive.launch
```

## 7. Citation
```
@INPROCEEDINGS{9340765,  author={Wang, Shuai and Qin, Jiahu and Zhang, Zhanpeng},  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},   title={Spatio-Temporal Ultrasonic Dataset: Learning Driving from Spatial and Temporal Ultrasonic Cues},   year={2020},  volume={},  number={},  pages={1976-1983},  doi={10.1109/IROS45743.2020.9340765}}
```


## 8. License
This Project is released under the [Apache licenes](LICENSE).
