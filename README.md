# Social Media Sentiment Analysis & Emotion Detection on Remote Working
This repository contains code/report prepared for our project on "Sentiment Analysis & Emotion Detection on Remote Working" for the "Text Analytics" course at University of Pisa.

## Project Goal
The goal of this project is to build an emotion detection system that can accurately detect and classify emotions from text scraped from social media discussions related to remote working. We will analyze posts from platforms like X (previously known as Twitter) and Reddit to identify their sentiments (positive/negative/neutral) and categorize a range of emotions, including:
- Happiness: Feelings of joy and contentment regarding work-life balance and flexibility.
- Loneliness: Sentiments reflecting social isolation or disconnection from colleagues.
- Sadness: Feelings of loss associated with in-person interactions and office environments.
- Etc.

## Project Structure

The project is organized into the following directories:

### 1. **Data Collection**
   - **Purpose**: This folder contains contains the data files used in the project and python scripts to retrieve the data.
   - **Files & Folders**:
      - `Reddit_data.ipynb`: script that retrieves data from the subreddits /remotework, /workfromhome, /remotejobs using PRAW API.
      - `Reddit_data.csv`: data retrieved from Reddit.
      - `twitter_data.py`: script that retrieves tweets from X using keywords or hashtags like #remotework #wfm using Twikit package.
      - `twitter_data.csv`: data retrieved from X
      - `twitter_config.ini`: contains information about user authentication.
      - `cookies.json`: contains information about the already authenticated user, so as to avoid repeated calls to login method.
      - `goemotions` Folder:
          - `full dataset` subfolder: contains the three raw files of data retrieved from Go Emotions dataset.
          - `emotions.txt`: contains mapping of emotions
          -  `goemotions_train.tsv`: contains training labeled data.
