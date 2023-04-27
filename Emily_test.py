# import statements
import pandas as pd
import numpy as np
import scipy.stats as stats

radio = pd.read_csv("C:/Users/ecriggins/Downloads/TheLight909.csv")
print(radio.head().to_string())
print(radio.columns)
print(radio.shape)

# change column names to something simpler
radio.columns = ['Timestamp', 'Listened_Light', 'Music_Type', 'Find_out', 'SM_Most', 'Music', 'Sport_Broadcast',
                 'Sport_Podcast', 'Christian_Podcasts', 'Convo_CC', 'Likelihood_guestspeak', 'Weekdays',
                 'Saturday', 'Sunday', 'Listener_type', 'Year', 'Major', 'Downloaded', 'Email', 'Un19', 'Un20',
                 'Un21', 'Un22', 'Un23', 'Un24', 'Un25']

print(radio.head().to_string())
# string-split and one-hot encode
music_type = radio['Music_Type'].str.get_dummies(sep= ';')
print(music_type.to_string())
weekdays = radio['Weekdays'].str.get_dummies(sep=';').add_prefix('Weekdays ')
saturday = radio['Saturday'].str.get_dummies(sep=';').add_prefix('Saturday ')
sunday = radio['Sunday'].str.get_dummies(sep=';').add_prefix('Sunday ')
# add new columns to original data set
radio = pd.concat([radio, music_type, weekdays, saturday, sunday], axis=1)
print(radio.columns)
print(radio.head().to_string())

# list for each genre and times of day
genre = ['Christian R&B',
       'Christian alternative rock', 'Christian pop', 'Christian rap',
       'Christian rock', 'None of these options apply']

weekday_list = ['Weekdays Afternoon (4-6 PM)', 'Weekdays Lunch (12-4 PM)',
       'Weekdays Mid-Morning (10:30-12PM)',
       'Weekdays Morning (7:30am – 10:30am)',
       'Weekdays Morning (7:30am–10:30am)', 'Weekdays Night (6-10 PM)',
       'Weekdays None of these apply']

saturday_list = ['Saturday Afternoon (4-6 PM)',
       'Saturday Lunch (12-4 PM)', 'Saturday Mid-Morning (10:30-12PM)',
       'Saturday Morning (7:30am–10:30am)', 'Saturday Night (6-10 PM)',
       'Saturday None of these apply']
sunday_list = ['Sunday Afternoon (4-6 PM)',
       'Sunday Lunch (12-4 PM)', 'Sunday Mid-Morning (10:30-12PM)',
       'Sunday Morning (7:30am–10:30am)', 'Sunday Night (6-10 PM)',
       'Sunday None of these apply']

# total count per genre
for x in genre:
    s = radio[x].sum()
    print("Genre:", x, ":", s)

# total count per weekday time
for x in weekday_list:
    s = radio[x].sum()
    print(x, ":", s)

# total count per Saturday time
for x in saturday_list:
    s = radio[x].sum()
    print(x, ":", s)

# fill new columns with values rather than 0s and 1s
for x in genre:
    radio[x] = np.where(radio[x] == 1, "Yes", "No")

for x in weekday_list:
    radio[x] = np.where(radio[x] ==1, "Yes", "No")

for x in saturday_list:
    radio[x] = np.where(radio[x] ==1, "Yes", "No")

for x in sunday_list:
    radio[x] = np.where(radio[x] ==1, "Yes", "No")

print(radio.head().to_string())

# create pivot table/crosstabs based on columns
# first example
table = pd.crosstab(radio['Christian pop'], radio['Weekdays Afternoon (4-6 PM)'],
                    margins= True,
                    margins_name="Total")
print(table)

# chisquare p-value analysis
print(stats.chi2_contingency(table))

