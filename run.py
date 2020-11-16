'''
@Description:  
@Author: Shuai Wang
@Github: https://github.com/wsustcid
@Date: 2020-01-10 10:10:55
@LastEditTime: 2020-01-17 10:32:14
'''


import os 
import tensorflow as tf 


FLAGS = tf.app.flags.FLAGS

#  
tf.app.flags.DEFINE_string('mode', 'train', "optional: train, eval, drive, save ")
tf.app.flags.DEFINE_string('root_dir', '/home/ubuntu16/catkin_ws/src/sonar_navigation/', " ")
tf.app.flags.DEFINE_string('version', 'V200', " ")
tf.app.flags.DEFINE_string('output_path', '', " ")

# params for data generator
tf.app.flags.DEFINE_string('data_path', '/media/ubuntu16/Documents/datasets/Sonar/SIM/Track-I-II/2019-07-03-48.csv', "")
tf.app.flags.DEFINE_integer("length", 12, " ")
tf.app.flags.DEFINE_integer("dim", 2, " ")
tf.app.flags.DEFINE_integer("height", 66, " ")
tf.app.flags.DEFINE_integer("width", 200, " ")
tf.app.flags.DEFINE_integer('channels', 1, "number of channels of input image")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch")

# train
tf.app.flags.DEFINE_string('optimizer', 'adam', " ")
tf.app.flags.DEFINE_float('learning_rate', 1e-4, "lr")
tf.app.flags.DEFINE_integer('max_epochs', 3, " ")
#tf.app.flags.DEFINE_float('steer_weight', 1, "")
#tf.app.flags.DEFINE_float('speed_weight', 1, "")

# restore
tf.app.flags.DEFINE_string('restore_weights_path', '', "restore model")


# eval
tf.app.flags.DEFINE_string('eval_weights_path', '', " ")
tf.app.flags.DEFINE_string('eval_output_path', '', " ")
# drive
tf.app.flags.DEFINE_string('target_ip', '192.168.1.102', "ip of sim env")

# save model
tf.app.flags.DEFINE_string('save_weights_path', '', " ")
tf.app.flags.DEFINE_string('model_path', '', '')



def main(argv=None):
    
    FLAGS.output_path = os.path.join(FLAGS.root_dir, 'output', FLAGS.version)

    if FLAGS.mode == 'train':
        from train import train_model
        train_model(FLAGS)

    elif FLAGS.mode == 'drive':
        from drive import drive
        drive(FLAGS)

    elif FLAGS.mode == 'eval':
        from evaluate import eval_model
        eval_model(FLAGS)

    elif FLAGS.mode == 'save':
        from save_model import save_model_with_weights
        save_model_with_weights(FLAGS)
        
    

if __name__ == '__main__':
    tf.app.run()



