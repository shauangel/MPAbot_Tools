import logging

# LDA model settings
PASSES = 10
CHUNKSIZE = 5
ITERATION = 2000
TOPIC_NUM = 3
ETA = 0.1
ALPHA = 1
TOPIC_NUM_LIST = [3, 4, 5, 6, 7]
TOPIC_TERM_NUM = 10

# Log setting
LOG_MODE = logging.DEBUG
FORMAT = '%(levelname)s: %(message)s'
DATE_FORMAT = '%H:%M:%S'

# plot settings
COLOR_LIST = ['red', 'blue', 'orange','forestgreen', 'yellow',
              'aqua', 'purple', 'deeppink', 'dimgray', 'lime']
LINE_STYLE_TUPLE_LIST = [
    ('solid', (0, ())),
    ('dotted', (0, (1, 1))),
    ('dashed', '--'),
    ('dashdot', '-.'),
    ('loosely_dotted', (0, (1, 10))),
    ('densely_dotted', (0, (1, 1))),
    ('long_dash_with_offset', (5, (10, 3))),
    ('loosely_dashed', (0, (5, 10))),
    ('densely_dashed', (0, (5, 1))),
    ('loosely_dashdotted', (0, (3, 10, 1, 10))),
    ('dashdotted', (0, (3, 5, 1, 5))),
    ('densely_dashdotted', (0, (3, 1, 1, 1))),
    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
    ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
]
MARKER_STYLE_LIST = ['1', 'x', '|', 4, 10,
                     '.', 'o', '8', 's', 'p', '*', 'h', 'd', 'P', 'X', ',']
BAR_FILL_STYLE = ['/', '///', '.', 'xx', '*', 'o', '++']


