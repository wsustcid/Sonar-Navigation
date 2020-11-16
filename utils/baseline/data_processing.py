'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2019-08-12 10:36:54
@LastEditTime: 2019-08-12 10:36:55
'''

import csv
from math import sqrt
import numpy as np

class UltrasonicData:
    """ A custom data type
    Data member:
      .distance: distance from the car to the surrounding wall which is measured by 
                 16 ultrasonic sensors at a sequential time step
      .angular:  angular velocity control commands of the car for each sample.
    """
    def __init__(self, distance, angular):
        self.distance = distance
        self.angular  = angular
        
        
def csv_read(csv_path, length):
    """ Reading ultrasonic data from csv file
    Args: 
     - csv_path: the path of the csv file
     - length: the length of time-ordered sequence data to be read for each sample

    Return: 
     - Return a list of dataset which is represented in UltrasonicData 
       data type.
    """
    
    print("Reading data from {}".format(csv_path))
        
    # save time-independent samples and corresponding labels in
    # UltrasonicData type
    Data = []
    
    with open(csv_path, 'r') as f:
        f_reader = csv.reader(f)
        for line in f_reader:
            # a list which saves time-dependent distance data
            dist = [float(line[j+i*18]) for i in range(length) for j in range(16)]
            # use the last angular value in sequence data as label
            agl  = float(line[17+(length-1)*18])
            #[float(line[17+i*18]) for i in range(length)]
            Data.append(UltrasonicData(dist, agl))
    
    print("Now {} samples with length {} have been saved.".format(len(Data), length))

    return Data


def data_generator(Data, dim=2, target_height=None, target_width=None, batch_size=128, random_state=None):
    """ A python data generator
    Arg: 
    - Data: a list of dataset in which data is saved with custom data type
    - batch_size: the size of data batch
    - dim: the dimension of samples used for training or validation. Selectable
           value is 1, 2 or 3.
    - target_height, target_width: target height and width of input samples
    Return:
    yield a batch of sample arrays and label arrays when it is called every time
    """

    while True:
        # choosing a batch of data randomly
        if random_state is not None: np.random.seed(random_state)
        data_batch = np.random.choice(a=Data, size=batch_size, replace=False)
        # saving distance and angular velocity data seperately
        X = []
        Y = []

        # unpacking Ultrasonic data and resizeing data to target shape 
        for i in range(len(data_batch)):
            data = data_batch[i]
            if dim == 1:
                dist = data.distance
            
            elif dim == 2:
                dist = np.array(data.distance, dtype=np.float32)
                # reshape
                dist = dist.reshape(-1,16)
                height, width = dist.shape
                if target_height == None: target_height = height
                if target_width  == None: target_width  = width 
                # repeat column elements and row elements to the target shape
                dist = dist.repeat(target_height//height, axis=0)
                dist = dist.repeat(target_width//width, axis=1)
                # add channel axis for the gray scale image (channels_first)
                dist = dist[np.newaxis, :,:]
                
            elif dim == 3:
                dist = np.array(data.distance, dtype=np.float32)
                # reshape
                dist = dist.reshape(-1,16)
                height, width = dist.shape
                if target_height == None: target_height = height
                if target_width  == None: target_width  = width
                # add channel axis for the color image
                dist = dist[np.newaxis, :,:]
                # repeat column elements and row elements to the target shape
                dist = dist.repeat(target_height//height, axis=1)
                dist = dist.repeat(target_width//width, axis=2)
                dist = dist.repeat(dim, axis=0)
                
            else:
                raise IOError("Unidentified dim value: use 1, 2 or 3")
            
            # save to list to add axis of batch size (4 dims tensor)
            X.append(dist)
            Y.append(data.angular)
        
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        
        yield X, Y