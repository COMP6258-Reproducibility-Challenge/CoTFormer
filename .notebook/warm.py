# <project>/.notebook/warm.py — kernel pre-import config (V1.1: auto-generated).
# Each import block wrapped in try/except — failures log but do not abort.

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("[warm] numpy + matplotlib ready")
except Exception as e:
    print(f"[warm] numpy/matplotlib failed: {e}")

