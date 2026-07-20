import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ob/openbots_ws/src/dreamer_SPOT_implementation/informed-dreamer/install/dreamerv3'
