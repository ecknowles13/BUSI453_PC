# import statements
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from bioinfokit.analys import stat
import scipy.stats as stats
import openpyxl

# load dataset
BlackFriday = pd.read_excel("C:/Users/ecriggins/Downloads/BlackFriday.xlsx") # ("/Users/emilyknowles/Downloads/BlackFriday.xlsx")
print(BlackFriday.head().to_string())
print(BlackFriday.shape)

# change NA to 0 for select columns
BlackFriday['Product_Category_1'] = BlackFriday['Product_Category_1'].fillna(0)
BlackFriday['Product_Category_2'] = BlackFriday['Product_Category_2'].fillna(0)

# confirm it worked
print(BlackFriday.head().to_string())

# one sample t-test
tt1 = stat()
tt1.ttest(df=BlackFriday, test_type=1, res= 'Purchase', mu= 410)
print(tt1.summary)

# independent samples t-test/two samples t-test
tt2 = stat()
tt2.ttest(df=BlackFriday, test_type=2, res= 'Purchase', xfac= 'Gender')
print(tt2.summary)

# paired samples t-test/related t-test
tt3 = stats.ttest_rel(BlackFriday['Product_Category_1'], BlackFriday['Product_Category_2'])
print(tt3)

# one-way ANOVA
aov_model = ols('Purchase ~ C(City_Category)', data=BlackFriday).fit()
aov_table = sm.stats.anova_lm(aov_model, typ=2)
print(aov_table)

# Tukey's HSD
tukey = pairwise_tukeyhsd(endog= BlackFriday['Purchase'],
                          groups=BlackFriday['City_Category'],
                          alpha= 0.05)
print(tukey)

# boxplots
sns.boxplot(x = BlackFriday['City_Category'],
            y = BlackFriday['Purchase'])
plt.show()

# Chi-Square
# create frequency table
table = pd.crosstab(BlackFriday['City_Category'], BlackFriday['Marital_Status'],
                    margins= True, margins_name= "Total")
print(table)

# Chi-Square/Crosstabs for association (Pearson)
ch2 = stat()
ch2.chisq(df=table)
print(ch2.summary)
print(ch2.expected_df)