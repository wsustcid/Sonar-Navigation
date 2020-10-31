<!--
 * @Author: Shuai Wang
 * @Github: https://github.com/wsustcid
 * @Version: 1.0.0
 * @Date: 2020-04-26 16:34:53
 * @LastEditTime: 2020-10-31 20:31:04
 -->

# Spatio-Temporal Ultrasonic Dataset: Learning Driving from Spatial and Temporal Ultrasonic Cues

## Overview
<div align=center> <img src=./doc/cover.png width=500, height=300 /></div>

In this project, we collected a Spatio-Temporal Ultrasonic Dataset
(STUD) and trained an end-to-end driving model to autonomously drive the mobile robot in unknown indoor environments. This is inspired by the phenomenon that bats are able to navigate in complex environments by extracting spatial and temporal cues from sonar echoes.


Recent works have proved that combining spatial and temporal visual cues can significantly improve the performance of various vision-based robotic systems. However, for the ultrasonic sensors used in most robotic tasks, there is a lack of benchmark ultrasonic datasets that consist of spatial and temporal data to verify the usability of spatial and temporal ultrasonic cues. 

In this project, 
- we first propose a Spatio-Temporal Ultrasonic Dataset (STUD), which aims to
develop the ability of ultrasonic sensors by mining spatial and temporal information from multiple ultrasonic measurements.
- Then we investigate various representations of the ST ultrasonic data to uncover hidden patterns and structures within data, which is beneficial for subsequent tasks.
- Finally, we present an end-to-end driving model that learns driving policies by extracting spatial and temporal ultrasonic cues from the STUD. 

With the help of our STUD and our driving model, various indoor navigation tasks can be tackled in a cheaper and efficient way, which is unachievable by simply using the existing ultrasonic datasets. 

**Citation**
If you find our work useful in your research, please consider citing:
```latex
Shuai, Wang, Jiahu Qin, and Zhanpeng Zhang, "Spatio-Temporal Ultrasonic Dataset: Learning Driving from Spatial and Temporal Ultrasonic Cues," In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 1-8. IEEE, 2020.
```


## Prerequisites

What things you need to install the software and how to install them

```python
# sjjd
Give examples
```

## Usage
A step by step series of examples that tell you how to get a development env running

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

