# Checkpoint 1
# Reminder: 

# - You are being evaluated for completion and effort in this checkpoint. 
# - Avoid manual labor / hard coding as much as possible, everything we've taught you so far are meant to simplify and automate your process.
# We will be working with the same `states_edu.csv` that you should already be familiar with from the tutorial.

# We investigated Grade 8 reading score in the tutorial. For this checkpoint, you are asked to investigate another test. Here's an overview:

# * Choose a specific response variable to focus on
# >Grade 4 Math, Grade 4 Reading, Grade 8 Math
# * Pick or create features to use
# >Will all the features be useful in predicting test score? Are some more important than others? Should you standardize, bin, or scale the data?
# * Explore the data as it relates to that test
# >Create at least 2 visualizations (graphs), each with a caption describing the graph and what it tells us about the data
# * Create training and testing data
# >Do you want to train on all the data? Only data from the last 10 years? Only Michigan data?
# * Train a ML model to predict outcome 
# >Define what you want to predict, and pick a model in sklearn to use (see sklearn <a href="https://scikit-learn.org/stable/modules/linear_model.html">regressors</a>).


# Include comments throughout your code! Every cleanup and preprocessing task should be documented.
# <h2> Data Cleanup </h2>

# Import `numpy`, `pandas`, and `matplotlib`.

# (Feel free to import other libraries!)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load in the "states_edu.csv" dataset and take a look at the head of the data
df = pd.read_csv("states_edu.csv")
# You should always familiarize yourself with what each column in the dataframe represents. Read about the states_edu dataset here: https://www.kaggle.com/noriuk/us-education-datasets-unification-project
# Use this space to rename columns, deal with missing data, etc. _(optional)_

# <h2>Exploratory Data Analysis (EDA) </h2>
# Chosen one of Grade 4 Reading, Grade 4 Math, or Grade 8 Math to focus on: *Grade 8 Math*
response_variable = 'AVG_MATH_8_SCORE'

# How many years of data are logged in our dataset? 
num_years = df["YEAR"].nunique()
print(f"Number of years in the dataset: {num_years}")

# Let's compare Michigan to Ohio. Which state has the higher average across all years in the test you chose?
michigan = df[df["STATE"] == "MICHIGAN"]
ohio = df[df["STATE"] == "OHIO"]

michigan_avg = michigan[response_variable].mean()
ohio_avg = ohio[response_variable].mean()

if michigan_avg > ohio_avg:
    print("Michigan has a higher average Math score.")
else:
    print("Ohio has a higher average Math score.")

# Find the average for your chosen test across all states in 2019
nineteen = df[df["YEAR"] == 2019]
avg_2019 = nineteen[response_variable].mean()
print(f"Average 8th Grade Math Score in 2019: {avg_2019}")

# For each state, find a maximum value for your chosen test score
max_val = df.groupby("STATE")[response_variable].max()
print("Max 8th Grade Math scores by state:")
print(max_val)

# *Refer to the `Grouping and Aggregating` section in Tutorial 0 if you are stuck.
# h2> Feature Engineering </h2>

# After exploring the data, you can choose to modify features that you would use to predict the performance of the students on your chosen response variable. 

# You can also create your own features. For example, perhaps you figured that maybe a state's expenditure per student may affect their overall academic performance so you create a expenditure_per_student feature.

# Use this space to modify or create features.
df["instruction_expenditure_ratio"] = df["INSTRUCTION_EXPENDITURE"] / df["TOTAL_EXPENDITURE"]

# Made the change to see how much total expenditure is spent on instruction
# Feature engineering justification: **<In order to hypothesize that states with higher expenditure on instruction, could have better test scores>**
# <h2>Visualization</h2>

# Investigate the relationship between your chosen response variable and at least two predictors using visualizations. Write down your observations.

# **Visualization 1**
df.plot.scatter(x='instruction_expenditure_ratio', y=response_variable, alpha=0.6)
plt.xlabel('Instruction Expenditure Ratio')
plt.ylabel('8th Grade Math Score')
plt.title('8th Grade Math Score vs Instruction Expenditure Ratio')
plt.show()

# **<Instruction Expenditure Ratio vs 8th Grade Math Scores>**

# **Visualization 2**
df.plot.scatter(x='ENROLL', y=response_variable, alpha=0.6)
plt.xlabel('Enrollment')
plt.ylabel('8th Grade Math Score')
plt.title('8th Grade Math Score vs Enrollment')
plt.show()

# **<Enrollment vs 8th Grade Math Scores>**

# <h2> Data Creation </h2>

# Use this space to create train/test data_
from sklearn.model_selection import train_test_split

# X =
X = df[['instruction_expenditure_ratio', 'ENROLL']].dropna() 
# y = 
y = df.loc[X.index, response_variable]

y.fillna(y.median(), inplace=True)

# X_train, X_test, y_train, y_test = train_test_split(
#      X, y, test_size=, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# <h2> Prediction </h2>
# ML Models [Resource](https://medium.com/@vijaya.beeravalli/comparison-of-machine-learning-classification-models-for-credit-card-default-data-c3cf805c9a5a)
# import your sklearn class here
from sklearn.linear_model import LinearRegression

# create your model here

# model = 
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

## Evaluation
# Choose some metrics to evaluate the performance of your model, some of them are mentioned in the tutorial.
r2 = model.score(X_test, y_test)
print(f"R-squared: {r2}")

np.mean((model.predict(X_test) - y_test)**2)**0.5
# We have copied over the graphs that visualize the model's performance on the training and testing set. 

# Change `col_name` and modify the call to `plt.ylabel()` to isolate how a single predictor affects the model.
col_name = 'instruction_expenditure_ratio'
plt.figure(figsize=(12,6))
plt.scatter(X_train[col_name], y_train, color="red", label='True Training')
plt.scatter(X_train[col_name], model.predict(X_train), color="green", label='Predicted Training')
plt.xlabel(col_name)
plt.ylabel('8th Grade Math Score')
plt.title("Model Behavior On Training Set (Instruction Expenditure Ratio)")
plt.legend()
plt.show()

col_name = 'ENROLL'
plt.figure(figsize=(12,6))
plt.scatter(X_test[col_name], y_test, color="blue", label='True Testing')
plt.scatter(X_test[col_name], model.predict(X_test), color="black", label='Predicted Testing')
plt.xlabel(col_name)
plt.ylabel('8th Grade Math Score')
plt.title("Model Behavior on Testing Set (Enrollment)")
plt.legend()
plt.show()