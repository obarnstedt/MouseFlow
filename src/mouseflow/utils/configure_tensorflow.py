from typing import Literal


def configure_tensorflow(allow_memory_growth = True, logging_level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'] = 'ERROR') -> None:
    """
    Set tensorflow session options.
      - allow_memory_growth helps avoid some memory issues on the GPU: https://medium.com/@aiii/tensorflow-tips-7806eb7960e6
      - logging_level help set how much information is printed on the screen: https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.10/tensorflow/g3doc/tutorials/monitors/index.md#enabling-logging-with-tensorflow
    """
    import tensorflow as tf

    #  To evade cuDNN error message:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = allow_memory_growth
    tf.Session(config=config)

    logging_level = getattr(tf.looging, logging_level)
    tf.logging.set_verbosity(logging_level)