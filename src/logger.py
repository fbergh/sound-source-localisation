# Code from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
import tensorflow as tf
import numpy as np
import scipy.misc 
from io import BytesIO

class Logger(object):
    
    def __init__(self, log_dir, suffix = ""):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir, filename_suffix=suffix)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()