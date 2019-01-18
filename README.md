# How to Predict the Price of a House

### Problem Statement

The sale price of a house can be difficult to appraise. The many features of a house can increase, decrease or do neither to the price. This is important for people who wish to buy houses at low values and resell them for higher values. I will conduct an investigation of these features which will provide people with insight on what to consider when flipping houses. The data is from Ames, Iowa Housing Dataset. After exploring, cleaning, preprocessing the dataset, I will split the dataset into train and test sets and them through my Linear Regression Model, Lasso Model and Ridge Model. I will evaluate the best model to be the one that explains the highest percentage of covariance in the data and performs the best on unseen data meaning the train and test set scores have close values. Using the best model, I will finally make recommendations on how people can best profit from buying and reselling houses.

## Predicting the Sale Price of Homes in Ames, Iowa

### Contents:

 - [Description of Data](#Description-of-Data)
 - [Exploratory Data Analysis](#Exploratory-Data-Analysis)
 - [Feature Selection and Engineering](#Feature-Selection-and-Engineering)
 - [Model Benchmark/Tuning/Production](#Model-Benchmark/Tuning/Production) 
 - [Results](#Results)
 - [Kaggle](#Kaggle)
 - [Future Steps](#Future-Steps)

### Description of Data

The Ames, Iowa Housing Dataset, taken between 2006 and 2010 was split into two datasets with train set and test set. The train set was used to model parameters to make predictions on the sale price of testing set. I obtained the training and test dataset from Kaggle Ames Housing Project. Here is the train dataset on [Kaggle](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge/download/train.csv) and here is the test dataset on
[Kaggle](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge/download/test.csv). Interpretations of the datasets are included in the data dictionary and article: Ames Assessor's Office which contains information for individual properties sold in Ames, IA. Here is the [AmesIowa](http://jse.amstat.org/v19n3/decock/DataDocumentation.txt). The size of the train dataset was 81 features of a home and 2051 homes. The size of the test dataset was 80 features of a home and 879 homes. The features include 23 nominal, 23 ordinal, 14 discrete, 20 continuous variables and 2 observation indentified.

### Target

The aim was to build the model that best explains the importance of features and best predicts the sale price of homes based on selected features by conducting analysis between the different features of the dataset in relation to the target sale price of a home and importing considerations when building the model. The predictions of each model will be compared to Naive Baseline Prediction, which is the mean of y_train and then to each previous model thereafter. After obtaining the parameters of the dataset from the model, the model will be used to make predictions on unseen test data with the sale price removed. Submissions of the predictions are uploaded to [Kaggle](https://www.kaggle.com/c/dsi-us-6-project-2-regression-challenge).

### Technologies Used
- Modeling: scikit-learn - Linear Regression, Lasso
- Model Selection: train/test split
- Data Management: pandas, numpy

### Exploratory Data Analysis

The Ames, Iowa train dataset, taken between 2006 and 2010, consists of 2051 homes with 81 features with 23 nominal, 23 ordinal, 14 discrete, 20 continuous variables and 2 observation indentified.

The dataset had many null values which I initially filled with either 'Na' or 0's by referring to the data dictionary and using my domain knowledge expertise. I proceeded to manually drop columns containing a majority of 'Na' or 0's because the feature poorly reflects the homes in this dataset, and would introduce high bias to the model. I also dropped features with high colinearity because their coefficients threw each other off. I used scatterplot to plot living area which I assumed was one of the most important predictors against sale price to search for outliers in sale price and removed them to avoid affecting my models.

### Feature Selection and Engineering

First, I split the features into living area, rooms, type, quality, and condition categories. Then I swarmplotted the features against saleprice to see if the distribution was normal and/or had a linear relationship. From there, if at least two features from that category was normal and/or had an approximate linear relationship, I would keep the features. For example, overall quality and kitchen quality are both in the quality category and since they both were normally distributed and had an approximate linear relationship, I kept all of the quality features. I also dropped one of features for features which were redundant such as dropping year built because it has high colinearity with year remodeled. Next, from the numerical features, I plotted a heatmap of each feature against sale price to observe the correlation and kept all the features with correlation of 0.5 and above. I combined my numeric features and categorical features and I proceeded to one hot encode the categorical features with a final dataframe of 42 columns.

### Model Benchmark/Tuning/Production

The first model I used was the Linear Regression model. I split the train data into train split and test split, scaled my X_train and X_test, and fit my Linear Regression model. I calculated the RMSE or how far the predicted sale price is from the true sale price and the $R^2$ score or percentage of variance that is explained by the model for the train and test split to observe any high variance and overfitting. The RMSE of my Linear Regression model was lower than the RMSE of baseline prediction, but the model indicated overfitting because the $R^2$ score was significantly greater in the train split than in the test split. Next, I decided to use the Lasso model to aggressively zero out the coefficients of unimportant features. The $R^2$ score of the train split and the test split were close in value. I compared the RMSE and $R^2$ score of my Lasso model to that of my previous Linear Regresssion model and my Lasso model also performed better. 


### Results and Conclusion

For the results, the baseline value of a house is $182762.28. I barplot the coefficients of the Lasso model's final features in order of ascending weight. The ground living area had the highest $\beta$ coefficient weight adding $28,800 per unit increase, followed by total basement squarefeet adding $17,000 per unit increase and overall quality adding $14,900 per unit increase. The results of my model proves to be reasonable because square footage of primary living space is normally one of the most important predictors in sale price. Total basement squarefeet in Iowa, I would assume to be important because Iowa resides in a tornado zone and so basements are important. Overall quality of a house is a great predictor on sale price as quality is correlated to sale price. Features like kitchen and basement that had excellent quality had reasonably positive weights. Features like garage or exterior that had typical or average quality had reasonably negative weights because lower quality results in lower sale price. The weight of the year remodeled was reasonable as the newer the romodeling, the higher the sale price. For full baths and total rooms above grade, there are outlier homes with low prices that turn the weight negative.

In conclusion, when buying houses, begin by determining the ground living area, basement squarefeet if there is a basement, and assessing the overall quality of the house, kitchen quality, basement quality. I recommend purchasing low cost houses where these features need work. Then plan for some remodeling and improving these main features to significantly up the value of the house and then resell it.

### Kaggle

Using the Lasso model I produced and the parameters gathered from the train dataset, the Lasso model was used to predict the sale price on unseen data.

### Future Steps

- Consider including more feature engineering by polynomial feature engineering
- Improve sensitivity with variance threshold and other hyper
- Implement pipeline and gridsearch to guide efficient workflow