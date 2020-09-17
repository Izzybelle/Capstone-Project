# General Assembly Capstone Project: NLP and Binary Classification of Yelp business reviews
Predicting if a Yelp Business Review is useful or not.

This repository contains the documentation and code for my Capstone project on Yelp business reviews. This is split into 4 parts: 
- Pitch and problem statement
- Parse the data, EDA and Cleaning 
- Feature Engineering and Modelling
- Presentation on findings

## Abstract
The Capstone project was a 4-week final project for my General Assembly Immersive programme. The project involved using the Yelp dataset to classify whether a review was classed as useful or not based on it's content. I used a number of different natural languange processing (NLP) techniques including, count vectorizing, term-frequency - indirect document frequenct (TF-IDF), along with Logistic Regression (and other binary classification models). The result was a 62% accuracy score, which was a 7% increase on baseline. I used to ROC curve to evaluate the score.

The code for the projcet can be found [here](https://github.com/Izzybelle/Projects)

## Project Overview
The project was undertaken as part of my study for the General Assembly Data Science Immersive. My aim for the project was two-fold. First, I wanted to gain experince following a typical data science workflow. Second, I wanted to practice using various tools and techniques to tackle a problem requiring natural language processing.

## The Problem

The yelp dataset contains business reviews where customers have left their feedback. The business reviews can be given a 1 - 5 star rating and the customer reviews of the business can be voted as as a useful, cool, and/ or funny, where the reviewer can rate the reviews as 1 or 0, in case of the project (1 for useful and 0 being the assumption for the review not being useful). I decided to look at the useful rating as this has not been done previously.

### Context
The dataset is a selection of Yelp's businesses, reviews and user data, spanning over 5 years (2013 - 2017). It was originally put together for the Yelp Dataset Challenge which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In this dataset you'll find information about businesses across 10 metropolitan areas in 4 countries.

### Content
The dataset contains 5 json files: Business, Checkin, Review, Tip, User.

In total, there are:
- 8,021,122 user reviews
- 209,393 businesses
- 1,320,761 tips
- 1,968,703 users

### Business json

Provides information about the businesses being review on Yelp.

```javascript
{
    // string, 22 character unique string business id
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // string, the business's name
    "name": "Garaje",

    // string, the full address of the business
    "address": "475 3rd St",

    // string, the city
    "city": "San Francisco",

    // string, 2 character state code, if applicable
    "state": "CA",

    // string, the postal code
    "postal code": "94107",

    // float, latitude
    "latitude": 37.7817529521,

    // float, longitude
    "longitude": -122.39612197,

    // float, star rating, rounded to half-stars
    "stars": 4.5,

    // integer, number of reviews
    "review_count": 1198,

    // integer, 0 or 1 for closed or open, respectively
    "is_open": 1,

    // object, business attributes to values. note: some attribute values might be objects
    "attributes": {
        "RestaurantsTakeOut": true,
        "BusinessParking": {
            "garage": false,
            "street": true,
            "validated": false,
            "lot": false,
            "valet": false
        },
    },

    // an array of strings of business categories
    "categories": [
        "Mexican",
        "Burgers",
        "Gastropubs"
    ],

    // an object of key day to value hours, hours are using a 24hr clock
    "hours": {
        "Monday": "10:00-21:00",
        "Tuesday": "10:00-21:00",
        "Friday": "10:00-21:00",
        "Wednesday": "10:00-21:00",
        "Thursday": "10:00-21:00",
        "Sunday": "11:00-18:00",
        "Saturday": "10:00-21:00"
    }
}
```

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


