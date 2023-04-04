# import statements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# load data
consumer = pd.read_csv('C:/Users/ecriggins/Downloads/userprofile.csv')
print(consumer.head(20).to_string())
# see column data types - you may need to coerce
print(consumer.dtypes)

# to coerce
#df['column'] = df['column'].astype(int)

# find NA values for each column
print(consumer.isnull().sum())

# decide what to do with NA values, if total is less than 10% for each column, consider deletion
# Otherwise, impute appropriate measure of central tendency
# nominal - mode
# ordinal - median
# interval/ratio - mean

# print columns
print(consumer.columns)
print(consumer.dtypes)

# mean
print(np.mean(consumer.height))
# round
print(round(np.mean(consumer.height), 2))
# with f' statement
print(f'Average Consumer Height: {round(np.mean(consumer.height), 2)}')

# median
array = np.array(consumer.height)  # save column as an array
print(f'The median of height is {np.median(array)}')
# median ignoring na values
print(f'The median of height is {np.nanmedian(array)}')

# mode
print(f'The mode for the height column is {stats.mode(array, keepdims=False)}')  # calculate mode of array

# replace missing values
# df['column name'] = df['column name'].replace(['old value'], 'new value')
consumer['height'] = consumer['height'].fillna(1.67)
print(consumer.head(20).to_string())

# histogram for weight
plt.hist(consumer.weight)
plt.xlabel('Weight')
plt.ylabel('Count')
plt.title('Histogram of Consumer Weight')
plt.ylim([0, 40])
plt.xlim([0, 150])
plt.show()

# bar chart for personality
plt.hist(consumer.personality)
plt.ylabel("Frequency")
plt.xlabel("Personality")
plt.title("Bar Chart of Consumer Personality")
plt.show()

# seaborn plot
sns.histplot(consumer.personality)
plt.show()


# subplots
# categorical data/columns we want
variables = ['dress_preference', 'ambience', 'interest', 'personality', 'religion', 'activity']

# label categories
a = 3  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

# set figure size
fig = plt.figure(figsize=(14, 10))

# loop
for i in variables:
    plt.subplot(a, b, c)
    plt.title(i.title())
    plt.xlabel(i)
    sns.histplot(consumer[i], color='indigo')
    c = c + 1
plt.tight_layout()
plt.show()