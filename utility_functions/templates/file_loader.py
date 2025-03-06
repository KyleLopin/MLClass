# Copyright (c) 2024 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""
utility functions to get templates uploaded
"""

__author__ = "Kyle Vitautas Lopin"

# standard libraries
import pickle
import os
from tempfile import NamedTemporaryFile


PICKLE_FILE = "templates.pkl"
PICKLE_KEY = "midterm_review_25"
DOC_FILE = f"{PICKLE_KEY}.docx"


# Read existing data
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as file:
        templates = pickle.load(file)
        print(templates.keys())
else:  # use empty template
    templates = {}

# Get new document
with open(DOC_FILE, "rb") as docx_file:
    docx_data = docx_file.read()
# add to template
templates[PICKLE_KEY] = docx_data

# Step 3: Write to a temporary file
with NamedTemporaryFile("wb", delete=False) as temp_file:
    pickle.dump(templates, temp_file)
    temp_file_path = temp_file.name

# Step 4: Replace the original file with the temporary file
os.replace(temp_file_path, PICKLE_FILE)
print("Templates updated and saved using a temporary file.")
