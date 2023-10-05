#  Rating-the-Review-by-Sentiment-Analysis

The project involved sentiment analysis of Starbucks customer reviews, aiming to rate feedback on a 1 to 5 scale. It utilized a multiclass classification approach and employed resampling techniques for improved model performance. Achieving a remarkable 96% accuracy, the model was deployed using Flask, HTML, and CSS on a local host.

## Dataset
This dataset contains a comprehensive collection of consumer reviews and ratings for Starbucks, a renowned coffeehouse chain. The data was collected through web scraping and includes textual reviews, star ratings, location information, and image links from multiple pages on the ConsumerAffairs website. It offers valuable insights into customer sentiment and feedback about Starbucks locations.

## Column Description
There are 11 columns in this dataset.

* Name: The reviewer's name, if available.
* Location: The location or city associated with the reviewer, if provided.
* Date: The date when the review was posted.
* Rating: The star rating given by the reviewer, ranges from 1 to 5.
* Review: The textual content of the review, captures the reviewer's experience and opinions.
* Image Links: Links to images associated with the reviews, if available.

## SKills

- Python - 3.11
- Pandas
- Numpy
- Matplotlib
- scikit-learn
- Data visualization
- Data Preprocessing
- sentiment analysis
- Content Vectorization
- Data Modeling
- Multinomial naive_bayes
- RandomForestClassifier
- resampling techniques for improved model performance.
- Frontend - HTML, CSS
- Framwork - Flask

## Installation
```bash
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

from flask import Flask, render_template, request
import pickle
```
## Resampling Technique:

To address class imbalance, incorporated resampling techniques.
Specifically, resampling was applied to ensure that each sentiment class (star rating) had a balanced representation in the training data, enhancing the model's ability to accurately classify reviews across all classes.

## Deployment

![Deployment Image 1](https://github.com/rachanabv07/Rating-the-Review-by-Sentiment-Analysis/blob/main/image-.png)

## Source    
https://www.kaggle.com/datasets/harshalhonde/starbucks-reviews-dataset/data
