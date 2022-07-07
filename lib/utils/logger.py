import logging
import datetime


def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()


def setup_logger(logger_name, log_file, level=logging.INFO, debug=False):
    l = logging.getLogger(logger_name)
    # logging.Formatter.converter = beijing
    formatter = logging.Formatter('%(asctime)s : %(message)s')

    if not debug:
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
        l.addHandler(fileHandler)
    l.setLevel(level)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.addHandler(streamHandler)
    return l

