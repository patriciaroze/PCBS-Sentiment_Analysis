# Sentiment Analysis with Supervised learning models 

## Introduction 
The goal of this project is to compare the acuracy of three supervised learning models in a Sentiment Analysis classification task. The dataset I am working on is made of ~ 500,000 reviews of Amazon food products, written from Oct 1999 to Oct 2012 (see reference below). 

### Downloading the dataset
The dataset was too big to upload to Github, but you can download it here : https://drive.google.com/file/d/1JEipy9qa67FuMeqkL_uuLxQIwEEcBo3J/view?usp=sharing

To execute the script, download the csv under the name "reviews.csv" and place it in a folder named "data" in your directory. 

### Definitions  
Sentiment Analysis consists in predicting the opinion expressed in a text. In this project, I will use supervised learning models to predict the polarity - positive or negative - of the reviews. 
I define as positive the reviews rated 4 and 5, and as negative the reviews rated 1 to 3. Therefore, the task performed by the models will be a binary classification. 
The acuracy of a model is defined by the % of good predictions madeover the testing set. 

### Models 

In this project, I will use three models : two Naive Bayes classifiers - Gaussian Naive Bayes and Multinomial Naive Bayes, and a Random Tree model. NB : Over the project. I've "manually" tuned the hyper parameters of the model, meaning that I've manually tried different values and settled on the values that seemed optimal in a complexity/accuracy trade-off (minimizing the model's complexity and maximizing the accuracy of the prediction). This is not a very precise methodology, but my goal was mostly to roughly assess the performance of each model for the task, rather than achieving perfect accuracy. 

#### Description of the models : 
##### Naive Bayes classifiers 
Naive Bayes classifiers are a simple class of classifiers, based on Bayes' theorem. Naive Bayes classifiers assume strong independence between the features. 
Naive Bayes classifiers are a common baseline for text classification. 

1. Gaussian Naive Bayes
Gaussian Naive Bayes models are usually used when dealing with continuous data. They assume that the continuous values associated with each class are distributed following a Gaussian distribution. 

2. Multinomial Naive Bayes
In a multinomial Naive Bayes, the features of the vectors are assumed to follow a multinomial distribution. In a text-classification context, the points of the vector represent the frequency of occurence of each word within a specific context (document, dataset etc...). 

##### RandomForest
RandomForest classifiers operate by constructing a succession of decision trees. The outcome of the model can be seen as an averaging of multiple deep decision trees. 


I settled down on the use of those three models because they are the most common in classification tasks and are very popular regarding text classification. The suport vector machine model is also a very popular model that I decided not to implement because it can be very time-consuming. 


## 1. Exploring the dataset

The first part of the project consists in exploring the dataset. For that purpose, I computed a series of metrics listed below : 
- Ratings and length distributions : 
    - distributon of ratings 
    - distributions of review length in characters and word counts
    - distributions of helpfullness per ratings 
- Wordclouds per ratings (most common words per ratings)

The aime of this step is to understand the dataset, but also to figure out if our task could be "hacked". If we had found that positive reviews are highly correlated with the helpfulness score for instance, it might have been enough to try a simple linear regression method of predicting sentiment from helpfulness score. 
Hopefully, the exploration of the dataset doesn't show such significant correlations, and a supervised learning approach is relevant to fulfill the task.


## 2. Pre-processing

Before training our models, I had to pre-process the texts of the reviews. This means cleaning the data and vectorizing it so it can be numerically processed with functions. 
To clean the data, I manually removed empty cells of the dataset and converted all text to lowercase. I then used the nltk package to tokenize and lemmatize the text. 
From there, I used the Tfidf method to vectorize the text. The Tfidf method converts a list of tokens (i.e a review) into a vector were each point corresponds to the weighted frequency of the corresponding word. Each word is converted into a number corresponding to the frequency of the word within the review, weighted by its specificity (the inverse sum of all the reviews in which it occurs). The Tfidf model that I used also takes care of removing the stopwords of the text. 

## 3. Training, Testing : Summaries

I first applied the three models to the summaries (three - five word long titles given to the reviews) of the reviews. The idea was to compare the acuracy of the models on summaries vs on whole reviews. 
I found the following acuracy metrics (see jupyter notebook for the full confusion matrix) : 
- Gaussian Naive Bayes : 0.564
- RandomForest : 0.789
- Multinomial Naive Bayes : 0.784

Unsurprisingly, the RandomForest and the Multinomial Nauve Bayes classifiers have the best acuracy, since the gaussian naive bayes is based on hypotesis (continuity of features value and gaussian distribution of those values) that don't fit well the task. 

## 4.  Training, Testing : Full reviews

I then applied the three models to the full reviews. In this step, the number of features was considerably higher, which could predict a higher acuracy. Indeed, the acuracy was way better : 
- Gaussian Naive Bayes : 0.736
- RandomForest : 0.902
- Multinomial Naive Bayes : 0.800

## 5. Results 

Results show that a longer text can significantly improve acuracy of all models for Sentiment analysis. Acuracy improved as follows : 
- Gaussian NB : + 30%
- RandomForest : +12%
- Multinomial NB : + 2,6%

It's interesting to highlight the huge gain in acuracy of the gaussian naive bayes model, while the Multinomial Naive Bayes' acuracy only grows by 2,6%. 

This simulation shows that the RandomForest model is the most acurate - and therefore the one that should be favoured - for a binary sentiment analysis task. 

## Conclusion and feedbacks 

Before this project, I had only little experience with machine learning models. I feel that a big missing part of the project, that I would have achieved if I had more time, would have been to discuss the hyper-parameters of each model and try to discuss mathematically why each model behaved as it did and why its acuracy increased/not with more features. 
This project was a good introduction to NLP and the techniques of tex-preprocessing and vectorization. I could have used a word2vec vectorizer to achieve better acuracy, but I preferred to go with the Tfidf method that I was feeling more comfortable with. 

Regarding the course, I really enjoyed the variety of skills that we were introduced to, especially regarding image processing, game generation etc. I would have prefered having more exercises/TDs to hand in in order to learn a bit more on those topics that we are very rarely taught (since the curriculum is more focused on data analysis). 


## References 
- J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews (http://i.stanford.edu/~julian/pdfs/www13.pdf). WWW, 2013.
