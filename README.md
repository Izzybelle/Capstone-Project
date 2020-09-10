# General Assembly Capstone Project: NLP and Classification. 
Predicting if a Yelp Business Review is useful or not.

This repository contains the documentation and code for my Capstone project on Yelp business reviews. This is split into 2 parts: 
- Part 1: Parse the data, EDA and Cleaning 
- Part 2: Feature Engineering and Modelling

## Context
The dataset is a selection of Yelp's businesses, reviews and user data, spanning over 5 years (2013 - 2017). It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In this dataset you'll find information about businesses across 11 metropolitan areas in 4 countries.

## Content
The dataset contains 5 json files: Business, Checkin, Review, Tip, User. 

In total, there are:
- 8,021,122 user reviews
- Information on 209,393 businesses
- The data spans 10 metropolitan areas

## Inspiration
Natural Language Processing and working with Big Data.

What's in a review? What constitutes a useful review? Is it positive or negative? The Yelp reviews contain a lot of metadata that can be mined and used to infer meaning, business attibutes, and sentiment.

Another motivation to work on this dataset was to take an angle that has not been exploited and do some of my own unique analysis. 

## The project layout

### Part 1: Pitch and Problem Statement 
Define the problem statement, potential audience, goals, success metrics and data sources. Host a lightning talk presentation describing two of these proposals.

### Part 2a: Parse the data, EDA and Cleaning 
Source and format the required data for you project, perform preliminary data munging and cleaning of your data. Describe you data keeping your                 intended audience in mind. Document your work so far in Jupyter notebook. This includes parsing 2 larger files at 3.27 GB and 6.33 GB, splitting into equal segments via terminal to enable Python to upload them in turn to a PostresSQL database. 

### Part 2b: EDA and Preliminary Analysis
Quantitatively describe and visualise your data, maintain perspective on your goals and scope accordingly. This includes identifying foreign languages with Python package Lang Detect (and removing those) and sampling 100k observations out of 7.9 million rows.

### Part 3: Modelling
Detail your model and approach with concisely commented code, beginning with executive summary. Evaluate model performance and discuss results. Submit a complete notebook of the model. This include Feature Engineering with Count Vectorizer and TF-IDF and Classification models (Logistic Regression, Bernoulli NB, Linear SVM and RBF SVM).

### Part 4: Presentation
Host a short, well rehearsed presentation of your project for a non-technical audience. Cover goals, success criteria, data, approach, basic description of model, findings, risks/limitations, impact, next steps and conclusions.


