# main.py

import tkinter as tk
from tkinter import scrolledtext
import data_analysis

# Create a Tkinter window
app = tk.Tk()
app.title("EDA Results")

# Create a text box to display EDA results
results_text = scrolledtext.ScrolledText(app, width=60, height=15)
results_text.pack()

# Function to display the EDA results
def show_eda_results(eda_function, eda_name):
    result_text = eda_function()
    results_text.insert(tk.INSERT, f"{eda_name} Analysis Completed\n")
    results_text.insert(tk.INSERT, result_text)  # Display the result text in the text box

# Create buttons for different EDAs
eda_buttons = [
    {"name": "Class Distribution", "function": data_analysis.analyze_class_distribution},
    {"name": "Class Imbalance", "function": data_analysis.analyze_class_imbalance},
    {"name": "Average Image Size", "function": data_analysis.analyze_average_image_size},
    {"name": "Sample Images", "function": data_analysis.analyze_sample_images},
    {"name": "Average Pixel Values", "function": data_analysis.analyze_average_pixel_values},  # Added button for Q5
]

for eda_button in eda_buttons:
    tk.Button(
        app,
        text=eda_button["name"],
        command=lambda button=eda_button: show_eda_results(button["function"], button["name"])
    ).pack()

# Start the Tkinter event loop
app.mainloop()
