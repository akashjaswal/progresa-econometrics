
# coding: utf-8

# ## Study on Progresa Program (Mexico, 1997-98)
# 
# In this study, I use data from the [Progresa program](http://en.wikipedia.org/wiki/Oportunidades),
# a government social assistance program in Mexico. This program, as well as the details of its impact, 
# are described in the paper "[School subsidies for the poor: evaluating the Mexican Progresa poverty program]
#(http://www.sciencedirect.com/science/article/pii/S0304387803001858)", by Paul Shultz. 
# 
# The goal of this study is to implement some of the basic econometric techniques to measure the causal impact 
# of Progresa on secondary school enrollment rates. The timeline of the program was:
# 
#  * Baseline survey conducted in 1997
#  * Intervention begins in 1998, "Wave 1" of data collected in 1998
#  * "Wave 2 of data" collected in 1999
#  * Evaluation ends in 2000, at which point the control villages were treated. 
#  
# The data used below are actual data collected to evaluate the impact of the Progresa program. 
# In this file, each row corresponds to an observation taken for a given child for a given year. 
# There are two years of data (1997 and 1998), and just under 40,000 children who are surveyed in each year. 

# 
### Descriptive analysis
# 
### Summary Statistics

# Importing libraries

import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pylab import *
from scipy import *
from pylab import rcParams
import statsmodels.formula.api as smf
get_ipython().magic(u'matplotlib inline')

# Loading dataset and summarizing

progresa_df = pd.read_csv("progresa_sample.csv")
progresa_summary = progresa_df.describe()
print '\n Summary Statistics (Mean and Standard Deviation) for all numeric gemographic variables in the dataset:'
progresa_summary.loc[['mean','std']][['sex','indig','dist_sec','sc','grc','fam_n','min_dist','dist_cap',
                                     'hohedu','hohwag','welfare_index','hohsex','hohage','age','grc97','sc97']]


#### Assessing baseline data:
# We assess the baseline (1997) demographic characteristics **for the poor**  different in treatment and control villages using T tests:

# Cleaning values in main dataframe for treatment effect
progresa_df.loc[progresa_df.progresa == 'basal', 'progresa'] = 1
progresa_df.loc[progresa_df.progresa == '0', 'progresa'] = 0
progresa_df.loc[progresa_df['poor'] == 'pobre', 'poor'] = 1
progresa_df.loc[progresa_df['poor'] ==  'no pobre', 'poor'] = 0

# Subsetting data per required conditions 

baseline_97_treatment = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 97) & (progresa_df['progresa'] == 1)]
baseline_97_control = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 97) & (progresa_df['progresa'] == 0)]
baseline_ttest = pd.DataFrame(columns = ['Variable name', 'Average value (Treatment villages)', 'Average value (Control villages)', 'Difference (Treat - Control)', 'p-value'])

req_variables = ['sex','indig','dist_sec','sc','grc', 'fam_n','min_dist','dist_cap','hohedu','hohwag','welfare_index','hohsex','hohage','age']

# Finding the mean values for above variables and finding the differences in the treatment and control groups via T Test

for i in req_variables:
        baseline_ttest.set_value(req_variables.index(i), 'Variable name', i)
        baseline_ttest.set_value(req_variables.index(i), 'Average value (Treatment villages)',baseline_97_treatment[str(i)].mean())   
        baseline_ttest.set_value(req_variables.index(i), 'Average value (Control villages)' ,baseline_97_control[str(i)].mean()) 
        t = stats.ttest_ind(baseline_97_treatment[str(i)][~np.isnan(baseline_97_treatment[str(i)])], baseline_97_control[str(i)][~np.isnan(baseline_97_control[str(i)])] )
        baseline_ttest.set_value(req_variables.index(i), 'Difference (Treat - Control)',  t.statistic)
        baseline_ttest.set_value(req_variables.index(i), 'p-value',  t.pvalue)

print baseline_ttest

#### Graphical exploration

# Grouping data and finding average enrollment rates by level of household head education

avg_en = pd.DataFrame(progresa_df[progresa_df['year']==97].groupby('hohedu').mean()['sc'])
avg_en.reset_index(level = 0, inplace = True)
print avg_en

# Plotting the data

matplotlib.style.use('ggplot')
plt.scatter(list(avg_en['hohedu']),list(avg_en['sc']))
plt.xlabel("Level of Household head education in years")
plt.ylabel("Average Enrollment Rate in 1997")
plt.show()

# Treatment villages in 1997
vill_en_97 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 97) & (progresa_df['progresa'] == 1)].groupby('village').mean()
vill_en_97.reset_index(level=0, inplace = True)

# Treatment villages in 1998
vill_en_98 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 98) & (progresa_df['progresa'] == 1)].groupby('village').mean()
vill_en_98.reset_index(level=0, inplace = True)

print "\n"
rcParams['figure.figsize'] = 8, 5

# Plotting histogram for avg enrollment rates by Village in 1997 and drawing a vertical line at teh mean
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.hist(vill_en_97['sc'], color = "orange")
xlabel("Avg Enrollment Rates in 1997")
ylabel("Number of Villages")
plt.axvline(vill_en_97['sc'].mean(), color='red', linestyle='dashed', linewidth=2)

# Plotting histogram for avg enrollment rates by Village in 1998 and drawing a vertical line at teh mean
ax2 = fig.add_subplot(133)
ax2.hist(vill_en_98['sc'], color = 'skyblue')
xlabel("Avg Enrollment Rates in 1998")
ylabel("Number of Villages")
plt.axvline(vill_en_98['sc'].mean(), color='red', linestyle='dashed', linewidth=2)
plt.show()

# Printing out mean of Average enrollments for the two years and running a quick T test

print "\nMean of Avg Enrollment rates in 1997:",vill_en_97['sc'].mean()
print "\nMean of Avg Enrollment rates in 1998:", vill_en_98['sc'].mean()
t = stats.ttest_ind(vill_en_97['sc'][~np.isnan(vill_en_97['sc'])], vill_en_98['sc'][~np.isnan(vill_en_98['sc'])])
print "\n",t


### Measuring Impact
# Our goal is to estimate the causal impact of the PROGRESA program on the social and economic outcomes of individuals in Mexico. 

### Simple differences: T-test

# Isolating the treatment and control villages for the year 1998
treat_98 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 98) & (progresa_df['progresa'] == 1)]
control_98 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 98) & (progresa_df['progresa'] == 0)]

# Finding and printing out mean values for the enrollment rates in 1998 across treated and control groups
treat_98_avg = treat_98['sc'].mean()
control_98_avg = control_98['sc'].mean()
print "\nAvg Enrollment in Treated Villages in 98 is: ", treat_98_avg
print "Avg Enrollment in Control Villages in 98 is: ", control_98_avg

# T test for determining differences in the two groups and their statistical significance
t = stats.ttest_ind(control_98['sc'][~np.isnan(control_98['sc'])], treat_98['sc'][~np.isnan(treat_98['sc'])])
print "\nThe Differences in Averages for Control and treatment villages (T Statistic): ",t.statistic 
print "The P value for Statistical Significance: ", t.pvalue

### Simple differences: Regression

# Data for 1998 if the poor households
data_98 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 98)]
data_98.loc[data_98.progresa == 'basal', 'progresa'] = 1
data_98.loc[data_98.progresa == '0', 'progresa'] = 0

# Running a regression model of enrollment rates on treatment assignment
lm = smf.ols(formula='sc ~ progresa', data=data_98).fit()
print lm.params
print lm.summary()


### Multiple Regression
# Including control variables

# Multiple linear regression with additional set of control variables
multiple_lm = smf.ols(formula='sc ~ progresa + sex + indig + dist_sec + fam_n + min_dist + dist_cap + hohedu + hohwag + welfare_index + age + hohage', data=data_98).fit()
multiple_lm.summary()


### Difference-in-Difference, version 1 (tabular)
# Thus far, we have computed the effects of Progresa by estimating the difference in 1998 enrollment rates across villages. 
# An alternative approach would be to compute the treatment effect using a difference-in-differences framework.

# Subsetting data for two years differently
data_97 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 97)]
data_98 = progresa_df[(progresa_df['poor'] == 1) & (progresa_df['year'] == 98)]

# Creating a new dataframe for showing results in a tabular format
index = ['Avg Enrollment Before Treatment', 'Avg Enrollment After Treatment']
cols = ['Control Group', 'Treatment Group']
double_diff = pd.DataFrame(index = index, columns = cols)

# Calculating mean values
treat_98 = data_98['sc'][data_98.progresa == 1].mean() 
treat_97 = data_97['sc'][data_97.progresa == 1].mean()
control_98 =  data_98['sc'][data_98.progresa == 0].mean()
control_97 =  data_97['sc'][data_97.progresa == 0].mean()

# Finding the difference in difference value for impact evaluation of our treatment effect
diff_in_diff = (treat_98 - treat_97) - (control_98 - control_97)
print "The result of Difference-in-Difference gives us the estimate of the treatment effect on enrollment rates! "
print "The value is: ", diff_in_diff
print "\nSee table below for individual average values for treatment and control groups before and after treatment:"
double_diff.loc["Avg Enrollment Before Treatment","Control Group"] = control_97
double_diff.loc["Avg Enrollment After Treatment","Control Group"] = control_98
double_diff.loc["Avg Enrollment Before Treatment","Treatment Group"] = treat_97
double_diff.loc["Avg Enrollment After Treatment","Treatment Group"] = treat_98
double_diff


### Difference-in-Difference, version 1 (regression)
# Now we use a regression specification to estimate the average treatment effects of the program in a difference-in-differences framework. 

progresa_df.loc[progresa_df['poor'] == 'pobre', 'poor'] = 1
progresa_df.loc[progresa_df['poor'] ==  'no pobre', 'poor'] = 0
progresa_df.loc[progresa_df.year == 97, 'post'] = 0
progresa_df.loc[progresa_df.year == 98, 'post'] = 1
pd.to_numeric(progresa_df['progresa'])
pd.to_numeric(progresa_df['poor'])
progresa_poor = progresa_df[progresa_df['poor'] == 1]
dd_lm = smf.ols(formula = 'sc ~ progresa + post + progresa:post + sex + dist_sec  + min_dist + dist_cap + hohedu + age + hohage', data=progresa_poor).fit()
dd_lm.summary()


### Difference-in-Difference, version 2
# In this part we adopt an alternative approach to compare enrollment rates in 1998 between poor and non-poor across treatment and control villages. 

progresa_df_98 = progresa_df[progresa_df['year'] == 98]
dd_lm = smf.ols(formula = 'sc ~ progresa + poor + progresa:poor + sex + dist_sec  + min_dist + dist_cap + hohedu + age', data=progresa_df_98).fit()
dd_lm.summary()

### Spillover effects 
# Thus far, we have focused on the impact of PROGRESA on poor households. Now we repeat our analysis instead focusing on the impact of PROGRESA on non-poor households. 

progresa_nonpoor = progresa_df[progresa_df['poor'] == 0]
dd_lm = smf.ols(formula = 'sc ~ progresa + post + progresa:post + sex + dist_sec  + min_dist + dist_cap + hohedu + age + hohage', data=progresa_nonpoor).fit()
print "The number of non-poor households in treatment group: ", len(progresa_nonpoor[progresa_nonpoor['progresa'] == 1])
dd_lm.summary()

### Summary 

# ---------
# This was an interesting study, and yes, the progresa program did have causal impact on the enrollment rates of the poor households in Mexico in 1998. In addition to that it may have also boosted the overall enrollment across the villages in both treatment and control groups.
# However, the magnitude of the impact is not drastically high as growth in enrollment rates occurs over a period of time as the scale of the program grows and more households are treated with the subsidy.
# There are multiple things to consider to ensure results of the study (causal impact of treatment) being accurately quantified and also the process is implemented without intended flaw. Assuming that the study was conducted with good randomization and the treatment and control groups do not differ significantly allows us to measure the causal impact accurately. 
# In the end, the return on investment for the stakeholders (in this case the government pumping money) calculation matters. If the government is paying more than what the enrollment costs then the amount may be used for other purposes. Also, there is no way to validate if the money is actually used for enrollment purposes (for instance, the government subsidizes 100 households, but only few tens actually get their children enrolled -- the progresa program still has an impact but the magnitude is not as high as it is supposed to be).
# Final verdict: Good program, great start, many things to look forward to!
# -----
