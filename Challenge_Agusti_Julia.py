#!/usr/bin/env python
# coding: utf-8

# In[2]:


# We will use PANDAS library for this project
import pandas as pd
# We will use MatPlot library for the visualizations
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# We will import to a DataFrame only the necessary fields for this project from the dataset: 'hour' and 'click'
fields= ['hour','click']
df=pd.read_csv('./data/train.gz',compression='gzip',skipinitialspace=True,usecols=fields)
# We have a look to the data structure
df.info(verbose=True)


# In[4]:


# We format the 'hour' field into datetime data type under a new field called 'date'. This will help to process the time series
df['date'] = pd.to_datetime(df['hour'].astype(str),format='%y%m%d%H')


# In[5]:


# We redefine as index for the DataFrame the new field 'date' ( this simplifies the processing and the visualizations )
df = df.set_index(pd.DatetimeIndex(df['date']))


# In[6]:


####### PART 1: CTR over TIME ########
######################################
# Now we are ready to aggregate per hour the timeseries into a new DataFrame
# While we will not need the hour field anymore in the aggregated DataFrame we reuse it to make the 'hour' count 
# and record it in this field. We can save some memory with this method.
df_agg=df.resample('h').agg({"hour":'size',"click":'sum'})


# We rename  the field hour to 'impressions': it contains the total count of 
# advertisement visualizations (how often they are shown)
df_agg.rename({'hour':'impressions'},axis='columns',inplace=True)


# In[7]:


# We have a look to the data structure
df_agg.info(verbose=True)


# In[8]:


# We list some of the records to check that everything looks fine
df_agg.head(n=20)


# In[9]:


# We compute the CTR ( Click Through-Rates) and we store it in a new column on the DataFrame
df_agg['ctr'] = df_agg.apply(lambda row: row['click'] / row['impressions'], axis=1)


# In[10]:


# Let´s check how it looks like
df_agg.head(n=20)


# In[11]:


# Now we are ready to visualize the the resulting time series
# We use seaborn style defaults and set the default figure size to fit the screen
sns.set(rc={'figure.figsize':(16, 4)})
ax=df_agg.plot(y='ctr')
ax.set_ylabel('CTR')
ax.set_xlabel('Date')


# In[14]:


# Let´s see how the Histogram looks like ( beside in the second part of the challenge we assume that 
# it is a Gaussian Distribution )
df_agg.hist(column='ctr')
plt.title("Histogram of CTR")
plt.xlabel("CTR")
plt.ylabel("Frequency (count)")


# In[15]:


####### PART 2: Outlier Detection ########
##########################################

# First we calculate the standard deviation for the CTR
ctr_std=df_agg['ctr'].std()


# In[16]:


# We assume that the Moving Average to compute is the Simple Moving Average (SMA). It would be possible to use a 
# Exponential Moving Average ( EMA ) but it is out of this exercise scope to assess the benefits

# There is no clue for the Window Size to use to calculate the Moving Average. We can think that the SMA is a Low Pass Filter
# for a signal that discriminates the noise on this signal. We can see the Window Size (in the frequency domain for the signal)
# as the cut frequency of this Low Pass Filter. 

# One possible method to decide the Window Size for the SMA is to consider the compromise between the 'smoothness' 
# ( noise reduction ) and the 'delay' on the smoothed signal that SMA introduces depending on the Window Size. If we compute the 
# SAD ( sum of absolute differences between the SMA obtained signal and the original ) for different window sizes we 
# could find a Window Selection Criteria

# We create a DataFrame for the SAD values for different Window Sizes
sad=pd.DataFrame()

# We calculate the SMA for a Window size between 2 and 12 samples and we keep them as a new column on the agreggated
# DataFrame from the previous exercices

for i in range(2,12):
    # We calculate the SMAs for a Window size between 2 and 12 samples and we keep them as a new column on the aggregated
    # aggregated DataFrame from the previous exercice. This means that we will end up with 10 SMA TimeSeries.
    df_agg['MA{}'.format(i)]=df_agg.rolling(window=i)['ctr'].mean()
    # We calculate the SAD for the corresponding Window as the difference between the SMA and the original signal
    sad.at[format(i),'SAD']=abs(df_agg['MA{}'.format(i)]-df_agg['ctr']).sum()


# In[17]:


#Let´s have a look how the new columns with the SMAs look like
df_agg.head(n=20)


# In[18]:


# Let´s check the SDA values
sad.head(n=20)


# In[34]:


# Let´s plot the SAD. As we will see there is a point where increasing the Window Size doesn´t produce much more benefit
# in the smoothing and it takes longer to compute and it lags the original signal more.
# We will pick the smallest window size where the SAD starts to flatten out. In our case the proper window size is 8 samples, 
# as we can see in the graph bellow.
ax=sad.plot(y='SAD',use_index=True)
plt.ylabel('SAD')
plt.xlabel('Window Size')


# In[ ]:


#### Now we are ready to detect the outliers under the criteria 1.5 standard deviations apart from its calculated moving
# average ( in our case MA8 )
criteria=1.5*ctr_std

# To find the Outliers we use the method 'WHERE' over the aggregated DataFrame, using the specified criteria.
# We assign the value of the outliers on a new column called 'outliers' in the aggregated DataFrame linked to 
# the corresponding Date
df_agg['outliers']=df_agg['ctr'].where(abs(df_agg['ctr']-df_agg['MA8'])>criteria)


# In[24]:


# We can plot the outliers
ax=df_agg.plot(y='outliers',style='o')
ax.set_xlabel('Date')
ax.set_ylabel('CTR Value')


# In[25]:


# If we want to keep all the points except the outliers ( in case we want to interpolate the missing samples ) we can
# do it with a simple method called MASK
no_outliers=df_agg['ctr'].mask(abs(df_agg['ctr']-df_agg['MA8'])>criteria)
# We can plot the 'remaining' points
ax=no_outliers.plot(style='o')
ax.set_xlabel('Date')
ax.set_ylabel('CTR')


# In[26]:


# Finally we highligh the outliers as requested on the challenge
ax = df_agg.plot(y='ctr')
df_agg.plot(y='outliers',ax = ax, style='+',color='red')
ax.set_xlabel('Date')
ax.set_ylabel('CTR')


# In[27]:


# As an extra we can save the timeseries/values for the outliers and the no-outliers for further processing
df_agg['outliers'].to_csv('./data/outliers.csv', header=None,  sep=';', mode='a')
no_outliers.to_csv('./data/no_outliers.csv', header=None,  sep=';', mode='a')


# In[ ]:




