try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    

def config_tensorflow(log_level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'] = 'ERROR', allow_growth: bool = True):
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    level = getattr(tf.logging, log_level)  # gets a variable from a string from a module (in this case, using "ERROR" would get you tf.logging.ERROR)
    tf.logging.set_verbosity(level)