'''
read_tensorboard.py

This script launches TensorBoard and opens it in the default web browser.

Usage:
python read_tensorboard.py <log_directory>

Example:
python read_tensorboard.py logs
'''

import os
import sys
import tensorboard
import webbrowser
from threading import Timer


def launch_tensorboard(log_dir, port=6006):
    tensorboard_cmd = f'tensorboard --logdir={log_dir} --port={port}'
    os.system(tensorboard_cmd)

    def open_browser():
        webbrowser.open(f'http://localhost:{port}')

    Timer(2, open_browser).start()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <log_directory>")
        sys.exit(1)

    log_directory = sys.argv[1]
    launch_tensorboard(log_directory)
