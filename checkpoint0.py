# Checkpoint 0 
# These exercises are a mix of Python and Pandas practice. Most should be no more than a few lines of code! 
# here is a Python list:

a = [1, 2, 3, 4, 5, 6]

# get a list containing the last 3 elements of a
last_three_elements = a[-3:]
print(last_three_elements)

# Yes, you can just type out [4, 5, 6] but we really want to see you demonstrate you know how to use list slicing in Python
# create a list of numbers from 1 to 20
numbers = []

for num in range(1, 21) :
    numbers.append(num)

print(numbers)

# now get a list with only the even numbers between 1 and 100
# you may or may not make use of the list you made in the last cell
numbers = []

for num in range(1, 101) :
    if num % 2 == 0 :
        numbers.append(num)
    else :
        continue

print(numbers)

# write a function that takes two numbers as arguments
# and returns the first number divided by the second
def checkpoint(first, second) :
    return first / second

# fizzbuzz
# you will need to use both iteration and control flow 
# go through all numbers from 1 to 30 in order
# if the number is a multiple of 3, print fizz
# if the number is a multiple of 5, print buzz
# if the number is a multiple of 3 and 5, print fizzbuzz and NOTHING ELSE
# if the number is neither a multiple of 3 nor a multiple of 5, print the number
def fizzbuzz() : 
    for num in range(1, 31) :
        if num % 3 == 0 and num % 5 == 0 :
            print("fizzbuzz")
        elif num % 3 == 0 :
            print("fizz")
        elif num % 5 == 0 : 
            print("buzz")
        else :
            print(num)

# create a dictionary that reflects the following menu pricing (taken from Ahmo's)
# Gyro: $9 
# Burger: $9
# Greek Salad: $8
# Philly Steak: $10
menu = {
    "Gyro": "$9",
    "Burger": "$9",
    "Greek Salad": "$8",
    "Philly Steak": "$10" 
}

# load in the "starbucks.csv" dataset
# refer to how we read the cereal.csv dataset in the tutorial
import pandas as pd
df = pd.read_csv("starbucks.csv")

# select all rows with more than and including 400 calories
cals = df[df["calories"] >= 400]
print(cals)

# select all rows whose vitamin c content is higher than the iron content
vitamin = df[df["vitamin c"] > df["iron"]]
print(vitamin)

# create a new column containing the caffeine per calories of each drink
new_col = df["caffeine per calories"] = df["caffeine"] / df["calories"]
print(new_col)

# what is the average calorie across all items?
avg = df["calories"].mean()
print(avg)

# how many different categories of beverages are there?
diff_categ = df["beverage_category"].nunique()
print(diff_categ)

# what is the average # calories for each beverage category?
num_cals = df.groupby("beverage_category")["calories"].mean()
print(num_cals)
