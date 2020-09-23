# General Assembly Capstone Project: NLP and Binary Classification of Yelp business reviews
Predicting if a Yelp Business Review is useful or not. What chracterizes a useful review?

This repository contains the documentation and code for my Capstone project on Yelp business reviews. This is split into 4 parts: 
- Pitch and problem statement
- Parse the data, EDA and Cleaning 
- Feature Engineering and Modelling
- Presentation on findings

## Abstract
The Capstone project was a 4-week final project for my General Assembly Immersive programme. The project involved using the Yelp dataset to classify whether a review was classed as useful or not based on it's content. I used a number of different natural languange processing (NLP) techniques including, count vectorizing, term-frequency - indirect document frequenct (TF-IDF), along with Logistic Regression (and other binary classification models). The result was a 62% accuracy score, which was a 7% increase on baseline. I used to ROC curve to evaluate the score.

The code for the projcet can be found here: [Part 2a & b - EDA](https://github.com/Izzybelle/Capstone-Project/blob/master/Capstone%20-%20Yelp%20-%20Part%202a%20%26%20b%20-%20Parse%20the%20data%2C%20EDA%20and%20Cleaning.ipynb) | [Part 3 - Modelling](https://github.com/Izzybelle/Capstone-Project/blob/master/Capstone%20-%20Yelp%20-%20Part%203%20-%20Modelling.ipynb)

## Project Overview
The project was undertaken as part of my study for the General Assembly Data Science Immersive. My aim for the project was two-fold. First, I wanted to gain experince following a typical data science workflow. Second, I wanted to practice using various tools and techniques to tackle a problem requiring natural language processing.

## The Problem
The Yelp Dataset contains business reviews from customers who have left their feedback. The business reviews can be given a 1 - 5 star rating and the customer reviews of the business can be voted by others as useful, cool, and/ or funny. The rating of the review is a 1 or 0, 1 for useful and 0 being the assumption that the review is not useful. I decided to look at the useful rating as this has not been done previously.

### Context
The dataset is a selection of Yelp's businesses, reviews and user data, spanning over 5 years (2013 - 2017). It was originally put together for the Yelp Dataset Challenge a publically available dataset, which is a chance for students to conduct research or analysis on Yelp's data and share their discoveries. In this dataset you'll find information about businesses across 10 metropolitan areas in 4 countries.

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

### Review.json

Provides user_id review for business_id , as well as star rating, and number of votes received for useful, funny or cool  user review. 


```javascript
{
    // string, 22 character unique review id
    "review_id": "zdSx_SD6obEhz9VrW9uAWA",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // integer, star rating
    "stars": 4,

    // string, date formatted YYYY-MM-DD
    "date": "2016-03-09",

    // string, the review itself
    "text": "Great place to hang out after work: the prices are decent, and the ambience is fun. It's a bit loud, but very lively. The staff is friendly, and the food is good. They have a good selection of drinks.",

    // integer, number of useful votes received
    "useful": 0,

    // integer, number of funny votes received
    "funny": 0,

    // integer, number of cool votes received
    "cool": 0
}
```

### User.json

Provides the user_id metadata associated with the user.

```javascript
{
    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "Ha3iJu77CxlrFm-vQRs_8g",

    // string, the user's first name
    "name": "Sebastien",

    // integer, the number of reviews they've written
    "review_count": 56,

    // string, when the user joined Yelp, formatted like YYYY-MM-DD
    "yelping_since": "2011-01-01",

    // array of strings, an array of the user's friend as user_ids
    "friends": [
        "wqoXYLWmpkEH0YvTmHBsJQ",
        "KUXLLiJGrjtSsapmxmpvTA",
        "6e9rJKQC3n0RSKyHLViL-Q"
    ],

    // integer, number of useful votes sent by the user
    "useful": 21,

    // integer, number of funny votes sent by the user
    "funny": 88,

    // integer, number of cool votes sent by the user
    "cool": 15,

    // integer, number of fans the user has
    "fans": 1032,

    // array of integers, the years the user was elite
    "elite": [
        2012,
        2013
    ],

    // float, average rating of all reviews
    "average_stars": 4.31,

    // integer, number of hot compliments received by the user
    "compliment_hot": 339,

    // integer, number of more compliments received by the user
    "compliment_more": 668,

    // integer, number of profile compliments received by the user
    "compliment_profile": 42,

    // integer, number of cute compliments received by the user
    "compliment_cute": 62,

    // integer, number of list compliments received by the user
    "compliment_list": 37,

    // integer, number of note compliments received by the user
    "compliment_note": 356,

    // integer, number of plain compliments received by the user
    "compliment_plain": 68,

    // integer, number of cool compliments received by the user
    "compliment_cool": 91,

    // integer, number of funny compliments received by the user
    "compliment_funny": 99,

    // integer, number of writer compliments received by the user
    "compliment_writer": 95,

    // integer, number of photo compliments received by the user
    "compliment_photos": 50
}
```

### Checkin.json

Provides the timestamps for checkin data on a business.

```javascript
{
    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg"

    // string which is a comma-separated list of timestamps for each checkin, each with format YYYY-MM-DD HH:MM:SS
    "date": "2016-04-26 19:49:16, 2016-08-30 18:36:57, 2016-10-15 02:45:18, 2016-11-18 01:54:50, 2017-04-20 18:39:06, 2017-05-03 17:58:02"
}
```

### Tips.json

Provides tips written by a user on a business. Tips are shorter reviews and convey quick suggestions.

```javascript
{
    // string, text of the tip
    "text": "Secret menu - fried chicken sando is da bombbbbbb Their zapatos are good too.",

    // string, when the tip was written, formatted like YYYY-MM-DD
    "date": "2013-09-20",

    // integer, how many compliments it has
    "compliment_count": 172,

    // string, 22 character business id, maps to business in business.json
    "business_id": "tnhfDv5Il8EaGSXZGiuQGg",

    // string, 22 character unique user id, maps to the user in user.json
    "user_id": "49JhAJh8vSQ-vM4Aourl0g"
}
```

## The project layout

### Part 1: Pitch and Problem Statement 
> Define the problem statement, potential audience, goals, success metrics and data sources. Host a lightning talk presentation describing two of these proposals. The presentation for this is [here.](https://docs.google.com/presentation/d/1DdK5IffWKxk05gCQQ703wcEMzuKLIz9f-KdcksXILsA/edit?usp=sharing)

### Part 2a: Parse the data
> Source and format the required data for you project, perform preliminary data munging and cleaning of your data. Describe you data keeping your intended audience in mind. Document your work so far in Jupyter notebook. This includes parsing 2 larger files at 3.27 GB and 6.33 GB, splitting into equal segments via terminal to enable Python to upload them in turn to a PostresSQL database. 
#### Parse the data
Upon deciding finally to use the Yelp dataset for my Capstone, the initial challenge I encountered was the size of the data. Pandas in Python isn't able to process such such files. For those that were in the gigabytes I had to perform chunking, which splits the json files intro smaller chunks. Then I could iteratively load these smaller chunks into a Pandas dataframe and write to Postgres. The code for this is [here.](https://github.com/Izzybelle/Projects/blob/master/Capstone%20-%20Yelp%20-%20Part%202a%20%26%20b%20-%20Parse%20the%20data%2C%20EDA%20and%20Cleaning.ipynb)
 
### Part 2b: EDA and Preliminary Analysis
> Quantitatively describe and visualise your data, maintain perspective on your goals and scope accordingly. This includes identifying foreign languages with
> Python package Lang Detect (and removing those) and sampling 100k observations out of 7.9 million rows.
#### EDA
This dataset contains some reviews in foreign languages but it isn't clear what the proportion is. For the purpose of my project, I wanted to focus on English language reviews since the data is of businesses in North America. 

I used python package LangDetect to identify that 99% of reviews are in English. The remaining 1% of foreign languages were removed equating to ~ 79,000 reviews, with 7.9 millions reviews remaining. The code for this is [here.](https://github.com/Izzybelle/Projects/blob/master/Capstone%20-%20Yelp%20-%20Part%202a%20%26%20b%20-%20Parse%20the%20data%2C%20EDA%20and%20Cleaning.ipynb)

With the remaining rows, the dataset was fairly balance (45/55), useful and not useful, respectively, which meant I could take a sample of the dataset equally. 

##### Countplot of the categorical features:

![alt text](https://github.com/Izzybelle/Capstone-Project/blob/master/Countplot%20of%20categroical%20features.png "Logo Title Text 2")

This chart shows that there is a skew amoung the distribution of the features. At a glance of the useful rated reviews, the majority are not marked as review  useful but there is still a substantial that do vote that a review is useful. Every review with a useful rating above 0, I will assume that these are useful, no matter how many times it was voted on. Making this a Binary Classification.

Briefly reviewing the other chart, for stars we can see that reviewers have positive bias and are more likely to give a rating of 5 star. 

The other ratings of cool and funny have a skew in the distribution. I decided that it will be difficult to gauge context and therefore it would be diffuclt to predict whether a review is cool or funny.

##### Scatter plot of correlations:

![alt text](https://github.com/Izzybelle/Capstone-Project/blob/master/Scatter%20plot%20of%20correlations.png)

There are no strong correlation between variables. It is not immediately noticeable from the scatter matrix and since useful votes are either 0 or 1. so anything greater than 1, doesn't make it more useful than something with just 1 vote of being useful. It will be helpful to make the useful column binary and plot this again. 

##### Proportion of votes: 

![alt text](https://github.com/Izzybelle/Projects/blob/master/useful_votes.png "Logo Title Text 1")

This is mostly balanced dataset 45% and 55%, for useful and not useful reviews, respectively. 

##### Reviews by city:

![alt text](https://github.com/Izzybelle/Capstone-Project/blob/master/Reviews%20by%20city.png "Logo Title Text 3")

The majority of business reviews are for business in Las Vegas. This could be a place to focus.

Following this, I sampled the dataset and although I did not conduct a hypothesis test to find what is the ideal representative sample. I took a sample as much as my computer could handle of the 100,000 observations. 50% were of useful reviews and 50% were of not useful reviews. The code for this is [here.]

### Part 3: Modelling
> Detail your model and approach with concisely commented code, beginning with executive summary. Evaluate model performance and discuss results. Submit a 
> complete notebook of the model. This include Feature Engineering with Count Vectorizer and TF-IDF and Classification models (Logistic Regression, Bernoulli NB, Linear SVM and RBF SVM).

#### Feature Engineering & Modelling
To begin with the feature engineering, I used Count Vectorizer and TF-IDF Vectorizer which I optimised on and applied the best parameters to use in the grid searching hyperparameters for the Binary Classifiers. 

I used both Count Vectorizer and TF-IDF Vectorizer which both return a sparse matrix of features, however they work dfferently to each other. Count Vectorizer returns counts of each word. TF-IDF adds weight to the word depending how many appearances in the corpus it makes. So rarer words have more weight. It is possible to tune parameters and for both so I removed stopwords, added max features of 1000, min df of 20% and ngrams (1,1). With these parameters my basic Logistic Regression returned a result of 0.64 on Test Score and 0.629 on CV Score an improvement of 9% on the baseline. The ROC Curve used on Binary Classifers returned a score of 0.69. 

The code for this is [here.](https://github.com/Izzybelle/Projects/blob/master/Capstone%20-%20Yelp%20-%20Part%203%20-%20Modelling.ipynb)
 
### Part 4: Presentation
> Host a short, well rehearsed presentation of your project for a non-technical audience. Cover goals, success criteria, data, approach, basic description of 
> model, findings, risks/limitations, impact, next steps and conclusions. The presentation for this is found [here.](https://github.com/Izzybelle/Projects/blob/master/Yelp%20reviews%20NLP%20and%20Classification.pptx)

#### Final Word

Although, the best result from the modelling was 9% above baseline. There is substantial room for improvement and the EDA and modelling was not exhaustative. Here are some of the findings I have made and ideas I have looked into futher progressing with the project. 

Firstly, the 100k sample I chose may not be representative of the 7.9 million reviews. I would need to calculate sample size for a confidence interval of 95% or 99%. If I was to use all the data then to overcome the breadth of the sheer amount of data I would have to implement big data solutions such as using Hashing Vectorizer or move to AWS to model the larger data.

Secondly, all the classifcation models had a difficult time correctly classifying observations and made a large amount of False Positives errors. To improve on the training, I would need to further implement feature engineering, by using POS tagging to further distinguish the kind of words and grammer used. Stemming is the process of reducing a word to it word stem it is less strict than Lemmatization so for the process of improving the False Positive rate, Lemmatization would offer better precision than stemming, but at the expense of recall.

Thirdly, there were far too many features for the model to classify correctly, for further featuve engineering, I would focus on a particular business or sector. As the variety of terms used were against a variety of business and some a specific to a specific industry, so isn't a good measure of usefulness in a review.

Finally, I would check the length of reviews and remove those that are shorter reviews as they would not add value to the model in train and therefore wouldn't work particularly well in test.

## Key Learnings
In all the project went fairly well, I received a result that was 9% above baseline. There is still room for improvement on the model and I have identified methods to do this in my final word. Some key takings are:

- As part of the Data Science EDA, it is important to conduct sufficient EDA that allows to focus on a business as the business reviews will be specific for that business and won't include noise from other businesses. This can then be vizualized in a way that the audience can understand the process.
- It is important to continue developing and learning about the tools available to NLP that will aid with modelling, such as POS tagging, Stemming and/ or Lemmatization.
- Getting comfortable with AWS and working with big data is key taking.

N.B. I used some of these techniques at the end of the project to test the techniques but was not able to implement them in full. I will be attempting a new project where I can better implement these.
