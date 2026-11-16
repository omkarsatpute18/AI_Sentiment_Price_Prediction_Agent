
import sys

import pandas as pd
import numpy as np
import sklearn
import streamlit
import plotly
from apify_client import ApifyClient
from exa_py import Exa
from dotenv import load_dotenv

print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("NumPy version:", np.__version__)
print("Scikit-learn version:", sklearn.__version__)

print("\nAll major libraries imported successfully!")
print("Apify client: Ready")
print("Exa client: Ready")
print("Environment configuration: Ready")