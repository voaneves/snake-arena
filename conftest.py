"""Deixa `import snakeai` funcionar rodando `pytest` da raiz do repositório."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
