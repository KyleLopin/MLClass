# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
from docx import Document
from docx.shared import Pt
import pandas as pd


def create_document(student_name, fish_data, iris_data, output_file="midterm.docx"):
    """
    Create a Word document with tables for fish and iris prediction datasets for a student.

    Args:
        student_name (str): The name of the student.
        fish_data (pd.DataFrame): Prediction dataset for fish (without Weight).
        iris_data (pd.DataFrame): Prediction dataset for iris (without Target).
        output_file (str): Name of the output Word document.
    """
    # Initialize the document
    doc = Document()

    # Add a title with the student's name
    doc.add_heading(f"Prediction Data Sets for {student_name}", level=1)

    # Add Fish Dataset
    doc.add_heading("Fish Prediction Data", level=2)
    table = doc.add_table(rows=1, cols=len(fish_data.columns) + 1)  # Add one column for Weight
    table.style = 'Light Grid Accent 1'

    # Add column headers
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(fish_data.columns):
        hdr_cells[i].text = col
    hdr_cells[len(fish_data.columns)].text = "Weight (To Predict)"  # Blank column

    # Add rows for fish data
    for _, row in fish_data.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
        row_cells[len(fish_data.columns)].text = ""  # Leave Weight blank

    # Add a new line
    doc.add_paragraph("\n")

    # Add Iris Dataset
    doc.add_heading("Iris Prediction Data", level=2)
    table = doc.add_table(rows=1, cols=len(iris_data.columns) + 1)  # Add one column for Target
    table.style = 'Light Grid Accent 2'

    # Add column headers
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(iris_data.columns):
        hdr_cells[i].text = col
    hdr_cells[len(iris_data.columns)].text = "Target (To Predict)"  # Blank column

    # Add rows for iris data
    for _, row in iris_data.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            row_cells[i].text = str(val)
        row_cells[len(iris_data.columns)].text = ""  # Leave Target blank

    # Save the document
    doc.save(f"{student_name}_{output_file}")
    print(f"Document saved as: {student_name}_{output_file}")


if __name__ == '__main__':
    # Example fish prediction data (excluding Weight)
    fish_prediction_data = pd.DataFrame({
        "Length": [30.1, 31.5, 29.8, 32.2, 30.7],
    })

    # Example iris prediction data (excluding Target)
    iris_prediction_data = pd.DataFrame({
        "sepal length (cm)": [5.1, 5.5, 6.0, 6.2, 5.8],
        "sepal width (cm)": [3.5, 3.8, 3.2, 3.0, 3.3],
        "petal length (cm)": [1.4, 1.5, 1.6, 1.7, 1.8],
        "petal width (cm)": [0.2, 0.3, 0.4, 0.5, 0.6]
    })

    # Student name
    student_name = "John Doe"

    # Create the document
    create_document(student_name, fish_prediction_data, iris_prediction_data, output_file="predictions.docx")
