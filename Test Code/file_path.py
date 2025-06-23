import pandas as pd
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # Hide the root window

# Open the file selection dialog
file_path = filedialog.askopenfilename(title="Select a file", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))

df = pd.read_csv(file_path,  parse_dates=['start_date', 'end_date'])