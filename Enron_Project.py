
# coding: utf-8

# # Identify Fraud - ENRON SCANDAL
# 
# [Summary link - Wikipedia](https://en.wikipedia.org/wiki/Enron_scandal)
# 
# >The Enron scandal was a financial scandal that eventually led to the bankruptcy of the Enron Corporation, an American energy company based in Houston, Texas, and the de facto dissolution of Arthur Andersen, which was one of the five largest audit and accountancy partnerships in the world. In addition to being the largest bankruptcy reorganization in American history at that time, Enron was cited as the biggest audit failure.
# 
# >Enron was formed in 1985 by Kenneth Lay after merging Houston Natural Gas and InterNorth. Several years later, when Jeffrey Skilling was hired, he developed a staff of executives that – by the use of accounting loopholes, special purpose entities, and poor financial reporting – were able to hide billions of dollars in debt from failed deals and projects. Chief Financial Officer Andrew Fastow and other executives not only misled Enron's Board of Directors and Audit Committee on high-risk accounting practices, but also pressured Arthur Andersen to ignore the issues.
# 
# >Enron shareholders filed a 40 billion dollar lawsuit after the company's stock price, which achieved a high of US90.75 per share in mid-2000, plummeted to less than 1 dollar by the end of November 2001. The U.S. Securities and Exchange Commission (SEC) began an investigation, and rival Houston competitor Dynegy offered to purchase the company at a very low price. The deal failed, and on December 2, 2001, Enron filed for bankruptcy under Chapter 11 of the United States Bankruptcy Code. Enron's 63.4 billion dollars in assets made it the largest corporate bankruptcy in U.S. history until WorldCom's bankruptcy the next year.
# 
# >Many executives at Enron were indicted for a variety of charges and some were later sentenced to prison. Enron's auditor, Arthur Andersen, was found guilty in a United States District Court of illegally destroying documents relevant to the SEC investigation which voided its license to audit public companies, effectively closing the business. By the time the ruling was overturned at the U.S. Supreme Court, the company had lost the majority of its customers and had ceased operating. Enron employees and shareholders received limited returns in lawsuits, despite losing billions in pensions and stock prices. As a consequence of the scandal, new regulations and legislation were enacted to expand the accuracy of financial reporting for public companies. One piece of legislation, the Sarbanes–Oxley Act, increased penalties for destroying, altering, or fabricating records in federal investigations or for attempting to defraud shareholders. The act also increased the accountability of auditing firms to remain unbiased and independent of their clients.

# <a id='top'></a>
# 
# Table of Contents
# <br><br>
# [Project Goal](#Goal) | 
# <br>
# [Dataset Questions](#Questions) | 
# <br>
# [Dataset Information](#Info) | 
# <br>
# [Feature Statistics](#Stats) |
# <br>
# [Explore Features](#Features) |
# <br>
# - [Salary](#Salary)
# - [Bonus](#Bonus)
# - [Total Payments](#Total Payments)
# - [Exercised Stock Options](#Stock Options)
# - [Total Stock Value](#Total Stock Value)
# - [Total Bonus and Exercised Stock Options](#Total BE)
# - [Total Payments and Stock Value in Millions](#Total Millions)
# - [Shared Receipt with POI](#Shared Receipt)
# - [To Messages](#To Messages)
# - [From Messages](#From Messages)
# - [Fraction to POI](#FTP)
# - [Fraction from POI](#FFP)
# 
# <br>
# [Outliers](#Outliers) | 
# <br>
# [Transform, Select, and Scale](#TSS) | 
# <br>
# [Algorithm Selection](#Algorithm) | 
# <br>
# [Evaluation Metrics](#Metrics) | 
# <br>
# [Performance Test](#Test) |
# <br>
# [Parameter Tuning](#Tuning) | 
# <br>
# [Final Analysis](#Analysis) |
# <br>
# [Validating Our Analysis](#Validating) | 
# <br>
# [Final Thoughts](#Thoughts) | 
# <br>
# ____________________________________________________________________________

# <a id='Goal'></a>
# 
# ## Project Goal
# 
# The goal of this project is to use the Enron dataset to train our machine learning algorithm to detect the possiblity of fraud (identify person's of interest.)  Since we know our persons of interest (POIs) in our dataset, we will be able to use supervised learning algorithms in constructing our POI identifier.  This will be done by picking the features within our dataset that separate our POIs from our non-POIs best.  
# <br>
# We will start out our analysis by answering some questions about our data.  Then, we will explore our features further by visualizing any correlations/outliers.  Next, we will transform/scale our features and select those that will be most useful in our POI identifier, engineering new features and adding them to the dataset if provided to be useful for our analysis.  We will identify at least two algorithms that may be best suited for our particular set of data and test them, tuning our parameters until optimal performance is reached.  In our final analysis, the algorithm we have fit will be validated using our training/testing data.  Using performance metrics to evaluate our results, any problems will be addressed and motifications made.  In our final thoughts, the performance of our final algorithm will be discussed. 
# <br>
# 

# In[1]:


"""Import pickle and sklearn to get started.
Load the data as enron_dict"""
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
import pickle
import sklearn
import pprint
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


with open("final_project_dataset.pkl", "rb") as data_file:
    enron_dict = pickle.load(data_file)


# In[3]:


df = pd.DataFrame(enron_dict)
df


# <a id='Questions'></a>
# 
# ## Dataset Questions
# 
# After getting our data dictionary loaded, we can start exploring our data.  We'll answer the following questions:
# <br>
# 1. How many people do we have in our dataset?
# 2. What are their names?
# 3. What information do we have about these people?
# 4. Who are the POIs in our dataset?
# 5. Who are the highest earners?  Are they POIs?
# 6. Whos stock options had the highest value (max exercised_stock_options)?
# 7. Are there any features we can ignore due to missing data?
# 8. What is the mean salary for non-POIs and POIs?
# 9. What features might be useful for training our algorithm?
# 10. Are there any features we may need to scale?
# 
# [Top](#top)

# In[4]:


print ('Number of People in Dataset: ', len(enron_dict) )


# In[5]:


pretty = pprint.PrettyPrinter()


# In[6]:


#sort names of Enron employees in dataset by first letter of last name
names = sorted(enron_dict.keys())  

print ('Sorted list of Enron employees by last name')
pretty.pprint(names) 


# In[7]:


print ('Example Value Dictionary of Features')
pretty.pprint(enron_dict['ALLEN PHILLIP K']) 


# Before we go any further, let's transform our dictionary into a pandas dataframe to explore further.

# In[8]:


list(enron_dict['METTS MARK'].keys())


# In[9]:


df.T.head()


# In[10]:


"""Write Enron Dictionary to CSV File for Possible Future Use and Easily Read into Dataframe"""

df1 = df.T
df1.index.names = ['name']
pd.DataFrame.to_csv(df1, 'enron1.csv')


# In[11]:


enron_dict.keys()


# In[12]:



fieldnames = ['name'] + list(enron_dict['METTS MARK'].keys())

with open('enron.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for name in enron_dict.keys():
        if name != 'TOTAL':
            n = {'name': name}
            n.update(enron_dict[name])
            writer.writerow(n)      


# In[13]:


#read csv into pandas dataframe 
enron = pd.read_csv('enron.csv')


# In[14]:


#added/combined feature, total bonus and exercised_stock_options
enron['total_be'] = enron['bonus'].fillna(0.0) + enron['exercised_stock_options'].fillna(0.0)


# In[15]:


#added feature, fraction of e-mails to and from poi
enron['fraction_to_poi'] = enron['from_this_person_to_poi'].fillna(0.0)/enron['from_messages'].fillna(0.0)
enron['fraction_from_poi'] = enron['from_poi_to_this_person'].fillna(0.0)/enron['to_messages'].fillna(0.0)


# In[16]:


#added feature, scaled total compensation
enron['total_millions'] = (enron['total_payments'].fillna(0.0) + enron['total_stock_value'].fillna(0.0))/1000000


# <a id='Info'></a>
# 
# ## Dataset Information

# In[17]:


#data information/types

enron.info()


# [Top](#top)
# 
# Just by looking at our dataset information above, we can quickly point out a few ways to narrow down our feature selection.  Some of our features have lots of <b>missing data</b>, so those may be ones that we can remove.  Features like "restricted_stock_deferred", "loan_advances", and "director_fees" may be some that we can take out altogether.  There are also a few features that seem to be giving us the same information, like "shared_receipt_with_poi","to_messages", "from_messages", "from_this_person_to_poi", and "from_poi_to_this_person" all tell us about the person's e-mail behavior and all have the same data count, 86.  We may be able to narrow those features down to just one or two, or <b>create a new feature</b> from them (see feature added above.)
# <br><br>
# Features that may be most useful, since we're dealing with corporate fraud, are those features that tell us about the money.  Let's follow the money!  Features that will give us that money trail will be "salary", "total_payments", "exercised_stock_options", "bonus", "restricted_stock", and "total_stock_value".
# <br><br>
# For now, let's continue to explore our dataset before making our final selection.

# <a id='Stats'></a>
# 
# ## Feature Statistics

# In[18]:


#number of POI in dataset
print ('There are 18 POI in our Dataset as you can see by our "True" count')
enron['poi'].value_counts()


# In[19]:


#set a baseline by extracting non-POIs and printing stats

non_poi = enron[enron.poi.isin([False])]

non_poi_money = non_poi[['salary','bonus','exercised_stock_options','total_stock_value',                         'total_payments','total_be','total_millions']].describe()
non_poi_money


# In[20]:


non_poi_email_behavior = non_poi[['shared_receipt_with_poi','to_messages',                                  'from_messages','fraction_from_poi','fraction_to_poi']].describe()
non_poi_email_behavior


# I thought it was interesting to see someone with 100% of their e-mails going to persons of interest.  Below, I printed out some features associated with this person.  After a little research, I found that Gene Humphrey was one of the first employees of Enron.  So, it makes sense that all of his e-mails were to persons of interest who had been with the company from the beginning.  Those were the only people he worked with.

# In[21]:


enron[(enron['fraction_to_poi']>0.7)][['name','salary','total_be',                                       'restricted_stock','total_stock_value','to_messages','poi']]


# In[22]:


#POI stats

poi_info = enron[enron.poi.isin([True])]
poi_money = poi_info[['salary','bonus','exercised_stock_options','total_stock_value',                      'total_payments','total_be','total_millions']].describe()
poi_money


# In[23]:


poi_email_behavior = poi_info[['shared_receipt_with_poi','to_messages',                                'from_messages','fraction_from_poi','fraction_to_poi']].describe()
poi_email_behavior


# In[24]:


#difference in non-poi compensation and poi compensation

difference_in_money = poi_money - non_poi_money
difference_in_money


# We can see from the table above that money matters!  The mean difference is especially telling.  And, although upper management tends to have greater compensation, you can't help but be shocked by the tremendous gap seen here.

# In[25]:


#difference in non-poi email behavior and poi behavior

difference_in_email = poi_email_behavior - non_poi_email_behavior
difference_in_email


# My original email behavior table was a bit less telling than the money table, but I was able to scale features to reflect e-mail behavior more accurately.  The updated tables can be seen above with the fraction of emails sent to and from POIs.

# In[26]:


#poi name, salary, bonus, stock options, total bonus and options, from messages, 
# and fraction to poi, ordered by total descending

poi_info[['name','salary','bonus','exercised_stock_options','total_be','total_millions',          'from_messages','fraction_to_poi']].sort_values('total_millions',ascending=False)


# Although we don't have the salary and bonus data for Joseph Hirko, his exercised stock options is second to Kenneth Lay.  Since "exercised_stock_options" seems to be a key indicator of a POI when salary/bonus is unavailable, that is definitely a feature we'll want to include in our final feature selection.  These features may be even more robust when taking the total of bonus and options.  In fact, let's add the feature, total_be, to our dataset and maybe it will come in handy.  *I went back and added this feature to the top of my code in order to include it in the info and analysis.*
# <br>
# Also, it was interesting to see that POIs don't have as many "from_messages" as non-POIs.  David Delainey is the only one that has well over 500 emails from him.  This may be a telling behavior of POIs, as they may prefer to talk face-to-face with others.
# <br>
# [Top](#top)

# <a id='Features'></a>
# 
# ## EDA - Explore Features
# 
# In this section, we'll visualize some of our features in order to explore them further.

# <a id='Salary'></a>
# 
# ### Salary

# In[27]:


average_salary = enron.groupby('poi').mean()['salary']
average_salary


# In[28]:


sns.boxplot(x='poi',y='salary',data=enron)


# <a id='Bonus'></a>
# 
# ### Bonus

# In[29]:


average_bonus = enron.groupby('poi').mean()['bonus']
average_bonus


# In[30]:


sns.boxplot(x='poi',y='bonus',data=enron)


# Wow!  An 8 million dollar bonus seems a bit much, but you can see the difference between John and Ken in other ways.  Below, you can see the difference between all other financial features is significant.  And, bonuses among POIs are still higher on average than non-POIs.  So, despite our non-POI outlier, this feature may still be useful in training our algorithm.

# In[31]:


enron[(enron['bonus']>6000000)][['name','salary','bonus','exercised_stock_options','restricted_stock','total_stock_value','poi']]


# <a id='Total Payments'></a>
# 
# ### Total Payments

# In[32]:


average_total_payments = enron.groupby('poi').mean()['total_payments']
average_total_payments


# In[33]:


sns.boxplot(x='poi',y='total_payments',data=enron)


# Wow!  That last boxplot has an outlier that is obviously pulling the mean waaaaaaaay up.  Who is that?

# In[34]:


enron[(enron['total_payments']>40000000)][['name','total_payments','poi']]


# In[35]:


#take Ken Lay out of the poi boxplot

kl_not_in = enron[(enron['total_payments']<40000000)]

sns.boxplot(x='poi',y='total_payments',data=kl_not_in)


# Well, at least now we can see the boxplots more clearly.  There's not much of a difference between POI and non-POI here when we take Ken Lay out, so total_payments probably won't be a feature we'll use.

# <a id='Stock Options'></a>
# 
# ### Exercised Stock Options

# In[36]:


average_optionsvalue = enron.groupby('poi').mean()['exercised_stock_options']
average_optionsvalue


# In[37]:


sns.boxplot(x='poi',y='exercised_stock_options',data=enron)


# Exercised stock options definitely looks to be higher among POIs, so this will definitely be a feature to include in our list of features for our algorithm.  

# <a id='Total BE'></a>
# 
# ### Total Bonus and Exercised Stock Options

# In[38]:


average_total_sbe = enron.groupby('poi').mean()['total_be']
average_total_sbe


# In[39]:


sns.boxplot(x='poi',y='total_be',data=enron)


# Total Bonus and Exercised Stock Options might be useful, but it might also just add to the noise.  So, maybe we won't use this one.  

# <a id='Total Stock Value'></a>
# 
# ### Total Stock Value

# In[40]:


average_stockvalue = enron.groupby('poi').mean()['total_stock_value']
average_stockvalue


# In[41]:


sns.boxplot(x='poi',y='total_stock_value',data=enron)


# Total stock value for POIs on average is much higher than non-POIs.  This feature is another good option for our POI identifier.

# <a id='Total Millions'></a>
# 
# ### Total Payments and Stock Value in Millions

# In[42]:


average_total_comp = enron.groupby('poi').mean()['total_millions']
average_total_comp


# In[43]:


sns.boxplot(x='poi',y='total_millions',data= enron)


# Let's try that one again without Ken Lay...

# In[44]:


sns.boxplot(x='poi',y='total_millions',data= kl_not_in)


# Hmmmm, maybe we didn't need to add this feature.  We can look closer by using lmplot and pairplot later on in our analysis.

# <a id='Shared Receipt'></a>
# 
# ### Shared Receipt with POI

# In[45]:


average_shared_receipt = enron.groupby('poi').mean()['shared_receipt_with_poi']
average_shared_receipt


# In[46]:


sns.boxplot(x='poi',y='shared_receipt_with_poi',data= enron)


# <a id='To Messages'></a>
# 
# ### To Messages

# In[47]:


average_to = enron.groupby('poi').mean()['to_messages']
average_to


# In[48]:


sns.boxplot(x='poi',y='to_messages',data= enron)


# <a id='From Messages'></a>
# 
# ### From Messages

# In[49]:


average_from = enron.groupby('poi').mean()['from_messages']
average_from


# In[50]:


sns.boxplot(x='poi',y='from_messages',data= enron)


# <a id='FTP'></a>
# 
# ### Fraction to POI

# In[51]:


average_fraction_to = enron.groupby('poi').mean()['fraction_to_poi']
average_fraction_to


# In[52]:


sns.boxplot(x='poi',y='fraction_to_poi',data= enron)


# Fraction_to_poi looks like a good feature to add to our list, since most of the poi distribution is in the upper range of the non-poi distribution.

# <a id='FFP'></a>
# 
# ### Fraction from POI

# In[53]:


average_fraction_from = enron.groupby('poi').mean()['fraction_from_poi']
average_fraction_from


# In[54]:


sns.boxplot(x='poi',y='fraction_from_poi',data= enron)


# [Top](#top)
# 
# <b>Pairplot Analysis</b>
# 
# Now, let's take a look at some of our features in the following pairplot.  Maybe it will help us make our final decisions for our features list.

# In[55]:


import seaborn as sns; sns.set(style="ticks", color_codes=True)

g = sns.pairplot(enron, vars=['bonus','exercised_stock_options','from_messages','fraction_to_poi'],
                 dropna=True, diag_kind='kde', hue='poi', markers=['x','o'])


# <a id='Outliers'></a>
# 
# ## Outliers
# 
# When looking at the stats for poi and non-poi for the first time, I noticed that the non-poi stats were much higher than the poi stats.  That's when I remembered I didn't account for the "TOTAL" key.  So, I went back and skipped over that key when writing to my csv.  I figured I'd just pop it out of my dictionary later if I need to.  After doing that, my stats were as expected.  Now, let's see what other outliers we can find.  In the pairplot above, there were two POIs that really stood out.  Let's take a closer look in the following lmplot.

# In[56]:


sns.lmplot(x='bonus', y= 'salary', hue='poi', data=enron, palette='Set1',size=10,markers=['x','o'])
plt.title('Salary/Bonus for POI and non-POI', fontsize=18)
plt.xlabel('Bonus', fontsize=16)
plt.ylabel('Salary', fontsize=16)


# In[57]:


#Who are the two outliers in blue with the high salary AND high bonus?  Ken Lay and Jeff Skilling of course!

enron[(enron['salary']>1000000)][['name','salary','bonus','poi']]


# <img src='Ken_Lay.jpg' width=100 height =150>

# These are two of our persons of interest, so we definitely don't want to take them out of our dataset.  

# According to [Executive Excess 2002](http://d3n8a8pro7vhmx.cloudfront.net/ufe/legacy_url/629/Executive_Excess_2002.pdf?1448073268)
# >Top executives at 23 companies under investigation for their accounting practices earned far more during the
# past three years than the average CEO at large companies. CEOs at the firms under investigation earned an
# average of 62.2 million dollars during 1999-2001, 70 percent more than the average of 36.5 million dollars for all
# leading executives for that period.

# We may also be able to find a few datapoints that are just causing noise by checking for lots of missing values in rows.

# In[58]:


#check for more than 20 missing values for each datapoint

i = 0

for row in enron.isnull().sum(axis=1):
    if row > 20:
        print (enron.iloc[i])
    i+=1


# In[59]:


#check for missing values in features

enron.isnull().sum()


# Loan advances has 142 missing values!  That's definitely a feature we can remove before we run our tests.

# <b>Money and Messages Regression Model</b>

# In[60]:


sns.lmplot(x='bonus', y='fraction_to_poi', hue='poi', data=enron, palette='Set1',size=10,markers=['x','o'])
plt.title('Money & Messages', fontsize=18)


# [Top](#top)
# 
# <a id='TSS'></a>
# 
# ## Transform, Select, and Scale
# 
# Now let's transform, select, and scale our features.  

# In[61]:


import sys
import pickle

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[62]:


enron.columns.tolist()


# In[63]:


features_full_list = enron.columns.tolist()
features_full_list.pop(0) #take out 'name'
features_full_list.pop(19) #take out 'email_address'
features_full_list.pop(11) #take out 'loan_advances' because of missing values
features_full_list.pop(15) #take out 'director_fees' because of missing values
features_full_list.pop(14) #take out 'poi' for now and add to beginning of list
features_list = ['poi']
for n in features_full_list:
    features_list.append(n)
features_list


# In[64]:


### Remove outliers that corrupt the data
enron_dict.pop('TOTAL', 0)


# In[65]:


#remove datapoints that create noise

enron_dict.pop('LOCKHART EUGENE E',0)


# In[66]:


#take out all 'loan_advances' because of missing values

for name in enron_dict:
    enron_dict[name].pop('loan_advances',0)


# In[67]:


### Create new feature(s)

#add fraction of emails from and to poi
#idea for this added feature taken from course materials

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if poi_messages != 'NaN' and all_messages != 'NaN':
        fraction = float(poi_messages)/all_messages


    return fraction


# In[68]:


for name in enron_dict:

    data_point = enron_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    
    enron_dict[name]["fraction_from_poi"] = fraction_from_poi
  
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )

    enron_dict[name]["fraction_to_poi"] = fraction_to_poi


# In[69]:


#add total_be to dictionary

for name in enron_dict:
    data_point = enron_dict[name]
    
    bonus = data_point['bonus']
    if bonus == 'NaN':
        bonus = 0.0
    options = data_point['exercised_stock_options']
    if options == 'NaN':
        options = 0.0
    total = bonus+options

    enron_dict[name]['total_be'] = total
    
    


# In[70]:


#add total compensation in millions to dataset

for name in enron_dict:
    data_point = enron_dict[name]
    
    total_payments = data_point['total_payments']
    if total_payments == 'NaN':
        total_payments = 0.0
    total_stock = data_point['total_stock_value']
    if total_stock == 'NaN':
        total_stock = 0.0
    total = (total_payments + total_stock)/1000000

    enron_dict[name]['total_millions'] = total


# <b>SELECT FEATURES</b>

# I've selected a few lists that may be useful in training our classifiers.  Each of the features selected may be able to give us some insight into the compensation and behavior of a POI.  The total compensation (total_millions) shows us that, on average, POIs are compensated more highly than non-POIs.  The same holds true for individual payments, like salary and bonus.  And, when it comes to stock behavior, POIs are more active in their exercising of stock options(exercised_stock_options.)  Other features, like from_messages, show a kind of pattern in e-mail behavior.  POIs do not send many messages.  However, the ones they do send are often to other POIs(fraction_to_poi).  These are all features we'll test before making our final feature selection.
# <br><br>
# <b>We will start out with our full list as a baseline and test that against our selected lists' metrics in order to find our final feature list for our POI identifier.  Our lists were chosen based on our stats and plots generated in our analysis.  Those features that showed a greater overall difference between POI and non-POI stats were chosen</b>

# In[71]:


print (features_list) #list of features available for testing


# In[72]:


### Select what features to use
first_list = ['poi','total_millions','fraction_to_poi','from_messages']
second_list = ['poi','total_be','fraction_to_poi','from_messages']
third_list = ['poi','salary','bonus','fraction_to_poi','from_messages']
fourth_list = ['poi','bonus','exercised_stock_options','fraction_to_poi']
features_final_list = fourth_list
print ("Final List", features_final_list)


# <a id="Algorithm"></a>
# 
# ## Algorithm Selection
# 
# <a id='Metrics'></a>
# 
# ### Evaluation Metrics

# In[73]:


#Evaluation metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.cross_validation import StratifiedShuffleSplit


# In[74]:


#Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[75]:



def test_list(classifier, feature_list, enron_dict):
    
    my_dataset = enron_dict
    data = featureFormat(my_dataset, feature_list, sort_keys = True) 
    labels, features = targetFeatureSplit(data) 
    
    X = np.array(features)
    y = np.array(labels)
    sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
    for train_index, test_index in sss:
        features_train, features_test = X[train_index], X[test_index]
        labels_train, labels_test = y[train_index], y[test_index]
        
    clf = classifier
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    
    if classifier == DecisionTreeClassifier():
        return {'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
                'Recall': recall_score(labels_test,pred), 'Feature Importance': clf.feature_importances_}
    
    return {'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
            'Recall': recall_score(labels_test,pred)}
    
    
   


# <a id='Test'></a>
# 
# ### Performance:  Accuracy, Precision, and Recall
# 
# Below, we will test each list using three different algorithms: Naive Bayes, Decision Tree, and KNearest Neighbors.  
# - Our accuracy score will show us our ratio of correctly predicted observation to the total observations.  
#     Accuracy = TP+TN/TP+FP+FN+TN
# - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations.  
#     Precision = TP/TP+FP
# - And, recall is the ratio of correctly predicted positive observations to all observations in the class.  
#     Recall = TP/TP+FN
# 
# When trying to identify POIs, we want to see as few falsely identified positives.  We don't want to falsely identify anyone as a POI.  So, I'd say precision is a bit more important here.  Let's see how each list does using each classifier.

# In[76]:


#first list
print (first_list)
print ('GaussianNB: ', test_list(GaussianNB(),first_list,enron_dict))
print ('DecisionTree: ', test_list(DecisionTreeClassifier(),first_list,enron_dict))
print ('KNeighbors: ', test_list(KNeighborsClassifier(),first_list,enron_dict))


# In[77]:


#second list
print (second_list)
print ('GaussianNB: ', test_list(GaussianNB(),second_list,enron_dict))
print ('DecisionTree: ', test_list(DecisionTreeClassifier(),second_list,enron_dict))
print ('KNeighbors: ', test_list(KNeighborsClassifier(),second_list,enron_dict))


# In[78]:


#third list
print (third_list)
print ('GaussianNB: ', test_list(GaussianNB(),third_list,enron_dict))
print ('DecisionTree: ', test_list(DecisionTreeClassifier(),third_list,enron_dict))
print ('KNeighbors: ', test_list(KNeighborsClassifier(),third_list,enron_dict))


# In[79]:


#fourth_list
print (fourth_list)
print ('GaussianNB: ', test_list(GaussianNB(),fourth_list,enron_dict))
print ('DecisionTree: ', test_list(DecisionTreeClassifier(),fourth_list,enron_dict))
print ('KNeighbors: ', test_list(KNeighborsClassifier(),fourth_list,enron_dict))


# With an accuracy score of 92%, a precision score of 75%, and a recall score of 60% ...
# 
# Our Final List includes <b>'poi', 'bonus', 'exercised stock options', and 'fraction to poi'</b>
# <br>
# Our Final Classifier with be <b>KNeighbors</b>

# <a id='Validating'></a>
# 
# ## Validation
# 
# We've already implemented our validation process, but here we will discuss its importance.  Without validating our classifier using training/testing data, we have no way of measuring its accuracy and reliability.  Training and testing the classifier against the same data will only yield overfitting results.  This is why validation is important.  By using StratifiedShuffleSplit to split our data into training and testing data, we can make sure that our classes are allocated by the same ratio set for training/testing and that each datapoint in the class is randomly selected.  Because of our small dataset, setting the iterations to 1000 will give us more reliable results in the end, as we will have trained and tested on almost all of our datapoints.  The only downside is the run time.

# In[80]:


### Store to my_dataset for easy export below.
my_dataset = enron_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_final_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[81]:


#Validation using StratifiedShuffleSplit in order to evenly dispurse the classes between training and test data
X = np.array(features)
y = np.array(labels)
sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
for train_index, test_index in sss:
    features_train, features_test = X[train_index], X[test_index]
    labels_train, labels_test = y[train_index], y[test_index]

#check for accuracy
    
clf = KNeighborsClassifier()
clf.fit(features_train, labels_train)

print (clf.score(features_test, labels_test))


# [Top](#top)
# 
# <a id="Tuning"></a>
# 
# ## Tuning
# 
# Even though we've all but settled on KNeighbors, let's see if tuning the parameters of our Decision Tree Classifier would make a difference.  Tuning parameters can sometimes significantly change our performance metrics outcome.  Parameters can control for overfitting/underfitting, so tuning them can certainly change the metrics.

# In[82]:


DecisionTreeClassifier().get_params()


# In[83]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

#set min_samples_split to 3 and increase until no longer helpful
clf = DecisionTreeClassifier(min_samples_split=9)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)


# In[84]:


"""  Can we do better?  [0.71052631578947367, 0.20000000000000001, 0.40000000000000002]"""

#print performance metrics
print ({'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
       'Recall': recall_score(labels_test,pred)})


# In[85]:


import scikitplot as skplt


# In[86]:


#plot confusion matrix
skplt.metrics.plot_confusion_matrix(labels_test, pred, normalize=True)


# How about our KNeighbors Classifier?  What are the parameters?  If we changed any of them, would it make a difference?

# In[87]:


KNeighborsClassifier().get_params()


# In[88]:


"""Can we do better without overfitting?  [0.92105263157894735, 0.75, 0.59999999999999998]"""

#set n_neighbors to 2 and increase until metrics show overfitting
clf = KNeighborsClassifier(n_neighbors=6)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print ({'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
       'Recall': recall_score(labels_test,pred)})


# ^OVERFIT!  

# In[89]:


"""Can we still do better?  [0.92105263157894735, 0.75, 0.59999999999999998]"""

#set n_neighbors to 2,3, and 4

def test_param(n):
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)

    return [accuracy_score(labels_test,pred), precision_score(labels_test,pred), recall_score(labels_test,pred)]


# In[90]:


print ("2: ", test_param(2))
print ("3: ", test_param(3))
print ("4: ", test_param(4))


# ### GridSearchCV

# When using GridSearchCV to find the best parameters for KNeighbors, our estimator gives us the same results.  

# In[91]:


from sklearn.model_selection import GridSearchCV

k = np.arange(10)+1
leaf = np.arange(30)+1
params = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),'leaf_size': leaf,'n_neighbors': k}

clf_params = GridSearchCV(KNeighborsClassifier(), params, cv=5)
clf_params.fit(features_train,labels_train)


# In[92]:


clf_params.best_estimator_


# Let's see what kind of results we get using the parameters "suggested" by GridSearchCV.

# In[93]:


clf = KNeighborsClassifier(algorithm='auto', leaf_size=1, metric='minkowski',
                           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                           weights='uniform')
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print ({'Accuracy': accuracy_score(labels_test,pred),'Precision': precision_score(labels_test,pred),
       'Recall': recall_score(labels_test,pred)})


# These are the same results as our default settings, so that's what we'll use.  

# [Top](#top)
# 
# <a id="Analysis"></a>
# 
# ### FINAL FEATURES and ALGORITHM SELECTION
# 
# Our final features, based on our feature analysis and testing, will be:
# 
# - Bonus
# - Exercised Stock Options
# - Fraction to POI
# 
# The KNeighbors Classifier, even without any parameter tuning, had a higher accuracy score than the Decision Tree Classifier with its min_samples_split tuned to 9.  So, we're going to use KNeighbors in our POI identifier.  KNearest Neighbors will help us zero in on pockets of POIs/non-POIs within our testing data.  The features I selected work well with this particular classifier because 'bonus' and 'exercised stock options' are good for training the algorithm to pick up on POI compensation trends, and 'fraction_to_poi' will help our algorithm pick up on POI e-mail behavior.  I narrowed it down to three features so as not to create noise.  When running our longer lists of features, we saw our precision and recall drop to zero, so cutting down our features to 3 was the strategy moving forward.  As we had already gotten rid of features with lots of missing data, it was easier to narrow them down and test each list against our chosen classifiers.

# In[94]:


clf = KNeighborsClassifier()
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)

print (accuracy_score(labels_test,pred))
print (precision_score(labels_test,pred))
print (recall_score(labels_test,pred))


# In[95]:


#plot confusion matrix
skplt.metrics.plot_confusion_matrix(labels_test, pred, normalize=True)


# <a id='Thoughts'></a>
# 
# ## Final Thoughts

# With an accuracy of 85-95%, precision of 65-75%, and recall of 35-60%, I think our algorithm has done well considering the small amount of data we had to work with.  But, using our validation methods proved useful in creating a reliable algorithm. We've done pretty well, but I don't think I'd want to bet anyones life in prison on any algorithm trained on this data.  We could use more data!  Our Naive Bayes and Decision Tree Classifiers didn't perform as well.  Although, overfitting is a problem as you increase the number of 'n_neighbors' in our KNeighbors Classifier, we can avoid this by keeping the default setting of 5.  That left us with a working algorithm and some pretty solid evaluation metrics.  According to our confusion matrix, we were able to identify 97% of non-POIs and 60% of POIs.  I'd rather get a few POIs wrong than falsely identify a non-POI as a POI.  There may come a day when people will be convicted based on machine learning, so it's important that we be as accurate as possible.  This identifier only gets it wrong around 40% of the time.

# In[96]:


# 'poi', 'bonus', 'exercised stock options', and 'fraction to poi' 

enron[(enron['bonus']>5000000) & (enron['poi'] == True)][['name','salary','bonus','poi']]


# ## Predict POI 

# In[97]:


pred_proba = clf.predict_proba(pd.DataFrame(features)[y==1].values)[:,1]
pred_proba


# In[103]:


predict_df = enron[enron.poi == 1]
predict_df['proba'] = pred_proba
predict_df


# In[114]:


predict_names = predict_df[predict_df['proba']>.5]


# In[115]:


predict_names.sort_values(by ='proba', ascending=False)
predict_names


# In[116]:


# result = {}
# for i in pred_proba:
#     result['i'] = enron_dict


# In[117]:


# df_pred = pd.DataFrame.from_dict(result, orient ='index',columns=['proba']).reset_index()
# df_pred[df_pred>.5].dropna()['index']


# In[118]:


# #Print POI names predicted by model
# df_pred[df_pred>.5].dropna()


# ## Predict POI probability new executive 

# In[119]:


bonus = int(input ('bonus                    : '))
stock = int(input ('exercised stock options  : ') )


# In[120]:


fraction_poi = float(input ('mails from this person to poi : '))/float(input ('from_messages : '))


# In[121]:


if (clf.predict([[bonus, stock, fraction_poi]]) == 1):
    print ('POI!!!...')
else:
    print ('NOT POI..')

