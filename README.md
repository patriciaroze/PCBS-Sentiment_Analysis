# Sentiment Analysis with Supervised learning models 

## Introduction 
The goal of this project is to compare the acuracy of three supervised learning models in a Sentiment Analysis classification task. The dataset I am working on is made of ~ 500,000 reviews of Amazon food products, written from Oct 1999 to Oct 2012 (see reference below). 

### Downloading the dataset
The dataset was too big to upload to Github, but you can download it here : https://drive.google.com/file/d/1JEipy9qa67FuMeqkL_uuLxQIwEEcBo3J/view?usp=sharing

To execute the script, download the csv under the name "reviews.csv" and place it in the folder "data" in your directory (yhe folder should be created automatically when you clone this repository). 

### Definitions  
Sentiment Analysis consists in predicting the opinion expressed in a text. In this project, I will use supervised learning models to predict the polarity - positive or negative - of the reviews. 
I define as positive the reviews rated 4 and 5, and as negative the reviews rated 1 to 3. Therefore, the task performed by the models will be a binary classification. 

### Models 

In this project, I will use three models : two Naive Bayes classifiers - Gaussian Naive Bayes and Multinomial Naive Bayes, and a Random Tree model. NB : Over the project. I've "manually" tuned the hyper parameters of the model, meaning that I've manually tried different values and settled on the values that seemed optimal in a complexity/accuracy trade-off (minimizing the model's complexity and maximizing the accuracy of the prediction). This is not a very precise methodology, but my goal was mostly to roughly assess the performance of each model for the task, rather than achieving perfect accuracy. 

#### Description of the models : 
##### Naive Bayes classifiers 
Naive Bayes classifiers are a simple class of classifiers, based on Bayes' theorem. Naive Bayes classifiers assume strong independence between the features. 
{\displaystyle {\text{posterior}}={\frac {{\text{prior}}\times {\text{likelihood}}}{\text{evidence}}}\,}


## 1. Exploring the dataset



## 2. Pre-processing

## 3. Training, Testing : Summaries

## 4.  Training, Testing : Full reviews

## 5. Results 

## Conclusion and feedbacks 

## References 
- J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews (http://i.stanford.edu/~julian/pdfs/www13.pdf). WWW, 2013.
