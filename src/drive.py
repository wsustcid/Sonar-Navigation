#!/usr/bin/env python
""" In this node, the ultrasonic data will be saved as a numpy array by subscribing 
to the /RosAria/sonar topic. Once the untrasonic data is updated, the desired angular
velocity will be predicted by the pretrained NN model. Then the control msg will 
be published to the /RosAria/cmd_vel topic to control a wheeled robot.

Parameters:
  - model_path: the path of saved model
  - model_name: the name of saved model
  - input_mode: the array shape of saved untrasonic data 
               1: used for Onde-dimensional MLP
               2: used for Two-dimensional CNN
               3: used for Three-dimensional CNN
  - length: the length of time sequence for the data
  - velocity: the predefined velocity


Remark:
1. When start this node, the path of venv should be specificed.
   For example: rosrun sonar_navigation drive.py --venv /home/ubuntu16/py2env
2. For the input array of the model.predict(), a new axies should be added to 
   represent the batch dimension.
"""

# Required before starting our code
from venv_utils import activate_virtual_env

# activate virtual environment
venv_path = '/home/ubuntu16/py2env'
venv_status = activate_virtual_env(path=venv_path)

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud

import os
import sys 
sys.path.append("/home/ubuntu16/catkin_ws/src/sonar_navigation/")
from models.mpilotnet import mpilotnet


from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from math import sqrt, ceil, floor
from multiprocessing import Queue

class Drive():
    def __init__(self):
        rospy.init_node("NN_Driver")

        rospy.on_shutdown(self.shutdown)

        # Params
        self.weights_path = rospy.get_param("~weights_path", 
                        "/home/ubuntu16/catkin_ws/src/sonar_navigation/output/V200/weights/1_weights_66_0.8542.h5")

        self.length     = rospy.get_param("~length", 12) 
        self.dim = rospy.get_param("~dim", 2) # 1, 2 or 3
        self.channels = rospy.get_param("~channels",1)
        self.height = rospy.get_param("~height", 66)
        self.width = rospy.get_param("~width", 200)
        self.velocity   = rospy.get_param("~velocity", 0.3)

        if self.dim ==3:
            self.qsize = 3*self.length
        else:
            self.qsize = self.length

        # A queue used to save data
        self.Q_data = Queue(self.qsize)
        
        # get the current tf graph
        self.graph = K.tf.get_default_graph()

        # Define model
        self.model = mpilotnet(self.channels, self.height,self.width)
        self.model.compile(optimizer=Adam(lr=1e-3),loss='mse')
        if self.weights_path !='' and os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print("Loading model from {}".format(self.weights_path))
        else:
            print("Specifying the model path!!")

        # Initialize a control msg
        self.cmd_vel = Twist()

        
        # Subcriber
        self.sub_sonar = rospy.Subscriber("/RosAria/sonar", 
                                          PointCloud, 
                                          self.sonar_callback,
                                          queue_size =1)

        # Publisher
        self.pub_cmd = rospy.Publisher("/RosAria/cmd_vel", Twist, queue_size=1)


    # sonar callback
    def sonar_callback(self, msg):
        """
        1. Computing current distance from the car to the surrounding obstacles.
        2. Saving the distance data over a desired time sequence.
        3. Reshaping saved data to the desired input shaped of the NN.
        """

        dist = [sqrt((msg.points[i].x)**2+(msg.points[i].y)**2) for i in range(16)]

        h_repeats = int(ceil(self.height/float(self.length)))
        w_repeats = int(ceil(self.width/float(16)))
        
        h_delta = h_repeats*self.length-self.height
        w_delta = w_repeats*16-self.width
    
        h_start = int(floor(h_delta/2.0))
        h_end = h_repeats*self.length-(h_delta-h_start)
    
        w_start = int(floor(w_delta/2.0))
        w_end = w_repeats*16-(w_delta-w_start)

        if self.length == 1:
            dist_array = np.array(dist, dtype=np.float32)

            input_array = dist_array.reshape(1,-1)
            
        else: 
            if self.Q_data.qsize() == self.qsize-1:
                # put new data
                self.Q_data.put(dist)
                # get all data from queue
                data=[self.Q_data.get() for i in range(self.qsize)]
                # put latest length-1 data into queue
                for i in range(1,self.qsize):
                    self.Q_data.put(data[i])

                dist_array = np.array(data, dtype=np.float32)

                if self.dim == 2:
                    dist_array = dist_array.reshape(1,self.channels,self.length,16)

                    dist_array = dist_array.repeat(h_repeats,axis=2)
                    dist_array = dist_array.repeat(w_repeats,axis=3)

                    input_array = dist_array[:,:,h_start:h_end,w_start:w_end]
                elif self.dim ==3:
                    dist_array = dist_array.reshape(1,self.channels,self.length,16)

                    dist_array = dist_array.repeat(h_repeats,axis=2)
                    dist_array = dist_array.repeat(w_repeats,axis=3)

                    input_array = dist_array[:,:,h_start:h_end,w_start:w_end]
                
                # prediction
                with self.graph.as_default():
                    prediction = self.model.predict(input_array, batch_size=1)
            
                # retrun a numpy array with the same dimesion as the input array. 
                self.angular  = prediction[0][0]
                print("Current predicted commands: angular={}".format(self.angular))
            
                # publish control cmd
                self.cmd_vel.linear.x = self.velocity
                self.cmd_vel.angular.z = self.angular
                self.pub_cmd.publish(self.cmd_vel)

            else:
                self.Q_data.put(dist)


    def shutdown(self):
        #self.pub_cmd(Twist())
        print("Shutdown!")
        #self.pub_cmd(Twist())



if __name__ == '__main__':
    try:
        Drive()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
        
