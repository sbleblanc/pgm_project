import os
import sys
from utils.template import template

template_name = sys.argv[1]
template_dict = dict(arg.split('=') for arg in sys.argv[2:])

config = dict()
with open(template_name) as f:
    exec(template(f.read(), template_dict), config)
