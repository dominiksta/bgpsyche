from matplotlib import pyplot as plt

from bgpsyche.logging_config import logging_setup

logging_setup()

plt.rcParams.update({
    'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavuserif',
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif'
})
