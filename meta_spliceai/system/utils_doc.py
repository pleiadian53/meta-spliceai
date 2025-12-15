# pip install python-pptx

import os, sys

import pandas as pd 
import numpy as np

from pathlib import Path

from rich.console import Console  # pip install rich 
from rich.panel import Panel

# Create an instance of Console
console = Console()


def print_with_indent(message, indent_level=0, indent_size=4):
    indent = ' ' * (indent_level * indent_size)
    console.print(f"{indent}{message}")


def print_section_separator():
    console.print(Panel("-" * 85, style="bold"))


def print_emphasized(text, style='bold', edge_effect=True, symbol='='):
    styles = {
        'bold': '\033[1m',
        'underline': '\033[4m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    end_style = '\033[0m'  # Reset to default style

    if edge_effect:
        edge_line = symbol * len(text)
        print(edge_line)
        print(f"{styles.get(style, '')}{text}{end_style}")
        print(edge_line)
    else:
        print(f"{styles.get(style, '')}{text}{end_style}")