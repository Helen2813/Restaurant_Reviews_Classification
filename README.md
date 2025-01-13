
# Restaurant Reviews Classification

This project performs restaurant review classification using machine learning. A Naive Bayes classifier is used to determine whether a review is positive or negative.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Data](#data)
- [Results](#results)

## Description

The goal of this project is to classify restaurant reviews using textual data. The project involves text preprocessing (cleaning, stemming) and a Naive Bayes classification algorithm.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd Restaurant_Reviews_Classification
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   .\venv\Scripts\activate  # For Windows
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure that the `Restaurant_Reviews.tsv` file is in the project root directory.
2. Run the script:
   ```bash
   python main.py
   ```
3. The model's accuracy will be printed in the console.

## Dependencies

The list of all dependencies can be found in [requirements.txt](requirements.txt). Key libraries used include:
- `numpy`
- `pandas`
- `nltk`
- `scikit-learn`
- `matplotlib`

## Data

The data is sourced from the `Restaurant_Reviews.tsv` file, which contains reviews and their labels (positive or negative). The file must be in TSV format (tab-separated) and contain two columns:
- `Review`: The review text
- `Liked`: The label (1 for positive review, 0 for negative review)

## Results

The model outputs its accuracy, which is displayed in the console upon running the script.
Accuracy:  0.73
