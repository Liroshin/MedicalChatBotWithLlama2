import os
from pathlib import Path
import logging

logging.basicConfig(Level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files=[
    "src/_init_.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html"
]

for filepath in list_of_files:
    Path(filepath)

