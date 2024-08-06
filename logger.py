import logging
import logging.config

from constant import LOGGING_DIR, LOGGING_LEVEL

LOGGING_DIR.mkdir(parents=True, exist_ok=True)


class T(logging.Filter):
    def filter(self, record):
        return my_filter(record)


def my_filter(record):
    if record.name == "matplotlib":
        return False
    elif "matplotlib" in record.name.split("."):
        return False
    else:
        return True


def gen_log_dict(name, level="DEBUG"):
    level = level.upper()
    name += "_" if name else ""
    (LOGGING_DIR / name).parent.mkdir(parents=True, exist_ok=True)

    LOGGING = {
        'version': 1,
        'disable_existing_loggers': False,

        'formatters': {
            'standard': {
                'format': '[%(levelname)s] [%(asctime)s] [%(filename)s] [%(funcName)s] [%(lineno)d] [%(thread)d]> %(message)s'
            },
            'simple': {
                'format': '[%(levelname)s] [%(asctime)s] > %(message)s'
            },
            'print': {
                'format': '%(message)s'
            }
        },

        'filters': {
            # 'a': {'name': 'a'}
            'a': {'()': T}
        },

        'handlers': {

            'stream': {
                'filters': ('a',),
                'level': level,
                'class': 'logging.StreamHandler',
                'formatter': 'print'
            },

            'access': {
                'filters': ('a',),
                'level': level,
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'simple',
                'filename': LOGGING_DIR / f"{name}access.log",
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'encoding': 'utf-8',
            },

            'boss': {
                'level': 'ERROR',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'standard',
                'filename': LOGGING_DIR / f"{name}boss.log",
                'maxBytes': 1024 * 1024 * 5,
                'backupCount': 5,
                'encoding': 'utf-8',
            },
        },

        'loggers': {

            '': {
                'handlers': (['stream', 'access', 'boss'] if level == "DEBUG" else ['access', 'boss']),
                # 'handlers': [ 'access', 'boss'],
                'level': level,
                'propagate': True,
            },

        }
    }

    return LOGGING


def init_log_conf(name="", level="DEBUG"):

    logging.config.dictConfig(gen_log_dict(name, level))
    logger = logging.getLogger()
    # logger.handlers[1].addFilter(my_filter)
    logger.info('#' * 30)
    return logger

# init_log_conf(BASE_PATH.stem)
