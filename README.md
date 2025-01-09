# Social Media Emotion Detection on Remote Working
This repository contains code/report prepared for our project on "Emotion Detection on Remote Working" for the "Text Analytics" course at University of Pisa.

## Project Goal
Our goal is to gain insight and shed light on how workers feel about remote work is essential, as their emotional well-being directly influences their job performance, overall satisfaction, and mental health. To do so, we will develop an emotion model that detects a range of emotions based on posts and discussions on social media platforms that remote workers engage in. The model will rely on the “Basic Emotions” framework developed by psychologist Parrott, which includes the following recognized primary emotions:  

- Anger: Characterized by furrowed brows, clenched jaws, and a narrowed gaze. Anger often arises from frustration, threat, or injustice.
- Love: Characterized by soft eyes, a gentle smile, and relaxed facial expressions. Love is an intense emotion of deep affection, care, and connection.
- Fear: Expressed through wide eyes, raised eyebrows, and a parted mouth. Fear is a response to perceived danger or threat.
- Joy: Indicated by smiling, raised cheeks, and crinkled eyes. Joy is a positive emotion associated with pleasure and satisfaction.
- Sadness: Characterized by drooping eyebrows, downturned mouth, and a tearful gaze. Sadness is often a response to loss, disappointment, or grief.
- Surprise: Expressed through raised eyebrows, wide eyes, and an open mouth. Surprise is a brief emotion triggered by unexpected events.

A table visualizing Parrott's emotion framework can be found [here](https://www.researchgate.net/figure/Parrotts-emotion-framework_tbl1_266657790).
 
Our research and data analysis, will seek to answer this primary question - **How do employees feel about remote working after the COVID-19 pandemic?**

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
       
### 2. **Data Preprocessing**
   - **Purpose**: This folder contains all the necessary steps to prepare the extracted data for use in emotion detection models. It includes data cleaning, tokenization, stop words removal, and emoji and special characters mapping.
   - **Files & Folders**:
      - `data_preprocessing.ipynb`: Contains the complete data processing workflow, including the following steps:
         - **Data Cleaning**: Removes unnecessary characters to normalize the text.
         - **Tokenization**: Converts the text into a list of words or subwords.
         - **Stop Words Removal**: Eliminates common words that do not add semantic value (e.g., "the," "and," "of," etc.).
         - **Emoji and Special Characters Mapping**: Maps emojis and special characters to equivalent words or removes them as appropriate.


### 3. **Pre Trained Model Implementation**
   - **Purpose**: This folder contains the notebooks used to implement the pre-trained models for ground truth generation over the dataset. Particularly, two pre-trained models were implemented: Google's T5 and (distilRoBERTa) Yangswei, trained specifically on Parrot's emotion tree.

   -  **Files & Folders:**:
      - `distilRoBERTa.ipynb`: Contains the implementation of the distilRoBERTa pre-trained model on the dataset.
      - `T5_model.ipynb`: Contains the implementation of the T5 model on the dataset.
      - `exploration_labels.ipynb`: Contains a preliminary exploration of the results of the distilRoBERTa model.
      - `labeled_data_reddit_text_yangswei_85.csv`: Dataset with the results of the distilRoBERTa model.
      - `t5_model_final.csv`: Dataset with the results of the T5 model.

### 4. **Model Implementation**
   - **Purpose**: This folder contains notebooks and scripts dedicated to implementing our emotional detection models which fall into two categories deep learning (RNN, BiRNN, LSTM) and SVM for both datasets Yangswei_85 and T5.
   - **Files & Folders**:
      -  Every model has its own notebook implementing the model, hypertuning the parameters, training, testing, and evaluating the performance.
      - `data`: contains the files split into train & test for both datasets.
      - `models`: contains the best model archicterure from each model saved in h5 or keras formats.
      - `results`: contains the results from the tests of the models which include respective plots, metrics, and matrices. Each model contains these materials within its own subfolder.

### 5. **Explainability**
   - **Purpose**: This folder contains notebooks dedicated to explainability techniques, specifically LIME, enabling the interpretation and analysis of predictions made by our emotion detection models. We choose one specific instance and analysed how different models predicted and "rationalized" their predictions.
   - **Files & Folders**:
      - `NN_LIME.ipynb`: Demonstrates the use of LIME to explain predictions made by the deep learning emotion detection models (RNN, BiRNN and LSTM) for both datasets.
      - `SVM_LIME.ipynb`: Applies LIME to analyze and interpret predictions from the SVM-based emotion detection model.
      - `Results`: contains the results from applying LIME to SVM and NN models saved into two subfolders for better clarity: SVM and NN.

### 6. **NRC**
  - **Purpose**: This folder contains the notebook and files used for implementing the NRC Lexicon used for comparative analysis.
   - **Files & Folders**:
      - `Basic NRC.ipynb`: contains the notebook where NRC is implemented on both datasets.
      - `t5_metrics_nrc.txt`: contains the metric results from testing T5.
      - `yangswei_85_metrics_nrc.txt`: contains the metric results from testing Yangswei 85.
      - `NRC-Emotion-Lexicon-Wordlevel-v0.92.txt`: contains the NRC lexicon utilized.
        
### 7. **general_utils**
   - **Purpose**: This folder contains scripts with utility functions that are used throughout the files in our project.
   - **Files & Folders**:
      - `metrics_plot_uils.py`: script that computes metrics, plots a confusion matrix, plot a training-validation losses curve plot for a given model.
      - `preprocessing.py`: script that contains utility functions for preprocessing the  NN models such as fixing contractions, tokenization, padding etc.
      - `svm_utils.py`: script that contains utility functions for the SVM models such as preprocessing text, saving and loading pickle files.
