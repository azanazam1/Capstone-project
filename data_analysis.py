# data_analysis.py

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_distribution():
    # Define the path to the dataset
    data_dir = "Dataset"  # The folder where you store your dataset

    # List the classes in the dataset
    classes = os.listdir(data_dir)

    # Create a dictionary to store the count of each class
    class_count = {c: len(os.listdir(os.path.join(data_dir, c)) ) for c in classes}

    # Question 1: What is the distribution of different classes in the dataset?
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_count.keys()), y=list(class_count.values()))
    plt.title("Distribution of Classes in the Dataset")
    plt.xticks(rotation=45)
    plt.show()

def analyze_class_imbalance():
    # Define the path to the dataset
    data_dir = "Dataset"  # The folder where you store your dataset

    # List the classes in the dataset
    classes = os.listdir(data_dir)

    # Create a dictionary to store the count of each class
    class_count = {c: len(os.listdir(os.path.join(data_dir, c)) ) for c in classes}

    # Question 2: Are there any class imbalances in the dataset?
    class_count_df = pd.DataFrame.from_dict(class_count, orient="index", columns=["Count"])
    class_count_df.plot.pie(y="Count", autopct='%1.1f%%', figsize=(8, 8), legend=False)
    plt.title("Class Imbalance in the Dataset")
    plt.show()

def analyze_average_image_size():
    # Define the path to the dataset
    data_dir = "Dataset"  # The folder where you store your dataset

    # List the classes in the dataset
    classes = os.listdir(data_dir)

    # Initialize a list to store image sizes
    image_sizes = []

    # Load and preprocess the dataset
    for c in classes:
        for image_name in os.listdir(os.path.join(data_dir, c)):
            img = cv2.imread(os.path.join(data_dir, c, image_name))
            image_sizes.append(img.shape)

    image_sizes = np.array(image_sizes)
    average_size = np.mean(image_sizes, axis=0)
    std_size = np.std(image_sizes, axis=0)
    
    result_text = f"Average Image Size: {average_size}, Standard Deviation: {std_size}\n"
    
    return result_text  # Return the result text

def analyze_sample_images():
    # Define the path to the dataset
    data_dir = "Dataset"  # The folder where you store your dataset

    # List the classes in the dataset
    classes = os.listdir(data_dir)

    # Question 4: Visualize sample images from different classes
    plt.figure(figsize=(12, 8))
    for i, c in enumerate(classes):
        img_path = os.path.join(data_dir, c, os.listdir(os.path.join(data_dir, c))[0])
        img = cv2.imread(img_path)
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(c)
        plt.axis('off')
    plt.show()

    #Q5 Calculating and visualizing the average pixel values within the same class
def analyze_average_pixel_values():
    # Define the path to the dataset
    data_dir = "Dataset"  # The folder where you store your dataset

    # List the classes in the dataset
    classes = os.listdir(data_dir)

    # Function to calculate the average pixel values for a class
    def calculate_average_pixel_values(class_dir):
        image_paths = os.listdir(class_dir)
        average_pixels = []

        for image_path in image_paths:
            img = cv2.imread(os.path.join(class_dir, image_path))
            average_pixel_value = np.mean(img)
            average_pixels.append(average_pixel_value)

        return average_pixels

    # Visualize the average pixel values for a few sample classes
    sample_classes = classes[:3]  # You can choose a few classes to visualize

    plt.figure(figsize=(12, 6))
    for i, c in enumerate(sample_classes):
        class_dir = os.path.join(data_dir, c)
        average_pixels = calculate_average_pixel_values(class_dir)
        plt.subplot(1, len(sample_classes), i + 1)
        plt.hist(average_pixels, bins=20, color='skyblue')
        plt.title(f"Average Pixel Values for Class: {c}")
        plt.xlabel("Average Pixel Value")
        plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
