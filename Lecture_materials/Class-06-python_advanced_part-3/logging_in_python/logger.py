import logging

logging.basicConfig(
    filename='app.log',                                       # specify the log file name
    filemode='w',                                             # 'w' for write mode, 'a' for append mode
    level=logging.DEBUG,                                      # minimum severity to log
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',  # this is a customizable format for log messages
    datefmt='%Y-%m-%d %H:%M:%S',                              # format for date and time
    )