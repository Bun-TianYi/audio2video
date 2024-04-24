import glob

import ffmpeg
import os
import sys

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(current_path)
    raw_data = os.path.join(root_path, 'data/raw/videos/')
    video_id = glob.glob(os.path.join(raw_data, '*'))
    ll = 0
