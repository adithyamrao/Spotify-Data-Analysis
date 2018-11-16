
# coding: utf-8

# # Spotify Data Exploration: the Popularity Feature
# ### Intro:
# After retrieving some data from the Spotify API (for more info about that check out [this](https://github.com/adithyamrao/Spotify-Data-Analysis/blob/master/Spotify%20Data%20Fetch.ipynb)) it's time to get some insights. In this notebook, I will use data collected during the month of Nov 2018 to identify the most popular tracks and artists on Spotify using the 'popularity' featue.
# 
# ### About the Popularity Feature:
# From the [official Spotify docs](https://developer.spotify.com/documentation/web-api/reference/search/search/):
# 
# >"The popularity of the track. The value will be between 0, for least popular, and 100 for most popular. The popularity of a track is a value between 0 and 100, with 100 being the most popular. Popularity is based mainly on the total number of playbacks. Duplicate tracks, such as both in a single and in an album, are popularity rated differently. Note: This value is not updated in real-time and may therefore lag behind in actual popularity."
# 
# ### Goal of this Notebook:
# The goal is to use the previously retrieved data to gain insights from the popularity feature such as most popular tracks and most popular artists by analyzing and visualizing the data using Python libraries Pandas, Numpy and Matplotlib.

# In[4]:


# import libraries
import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# get all csv files into one variable
path = 'Datasets/2018'
all_files = glob.glob(os.path.join(path, "*.csv"))

# create lists of columns to be used when reading/merging the csv's
columns = ['artist_name','track_id', 'track_name', 'popularity']
merge_columns = ['artist_name','track_id', 'track_name']

# create dataframes by reading the csv's in all_files
df_from_each_file = (pd.read_csv(f, usecols=columns) for f in all_files)

# create empty dataframe with the defined column structure
df = pd.DataFrame(columns=columns)

# loop over dataframes and merge into one dataframe
# outer join in order to keep the popularity column from each file
for df_, files in zip(df_from_each_file, all_files): # all_files are here to provide the column suffix (0920,0830 etc)
    df = df.merge(df_, how='outer', on=merge_columns, suffixes=('',str(files)[-8:-4]))

print('Shape: ', df.shape)
df.head()


# Since I have merged a single file based on artist and track names there shouldn't be a lot duplicates.
# 
# However, it is still worth to do a quick drop_duplicates here.

# In[5]:


# drop duplicate tracks
df.drop_duplicates(subset=['artist_name','track_name'], inplace=True)
print('Shape after dropping: ', df.shape)


# ## 1. Top 50 most Popular Tracks

# In[10]:


# sum individual popularity scores
df['popularity'] = df[['popularity2018']].sum(axis=1)

# calculate also the mean popularity score
df['popularity_mean'] = df[['popularity2018']].mean(axis=1)

# create new dataframe df_top ordered consisting of the 100 most popular tracks
df_top = df.sort_values('popularity', ascending=False).head(100)

# show the first 50 results
df_top[['artist_name', 'track_name', 'popularity', 'popularity_mean']].head(50)


# ## 2. Top Artists by Popularity
# Note: the Spotify API offers a special popularity score on artist-level as well. That score is not used here.
# 
# Instead, I have used only the popularity scores of their individual tracks.

# In[11]:


# show top 20 artists by number of tracks in top 100
df_top[['artist_name','track_name']].groupby('artist_name').count().sort_values('track_name', ascending=False).head(20)


# In[12]:


# show top 20 artists by total popularity of their tracks in top 100
df_top[['artist_name','popularity']].groupby('artist_name').sum().sort_values('popularity', ascending=False).head(20)


# ## 4. Visualizing Popularity

# In[13]:


# create a new transposed dataframe where the track names are the columns and individual popularities the rows
df_top10_pop = df_top[['track_name','popularity2018']].set_index('track_name').head(10).T

# set the figure size
plt.figure(figsize=(12,18))
 
# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot of the top10 track popularities
num=0
for track in df_top10_pop.columns:
    num+=1
 
    # find the right spot on the plot
    plt.subplot(10,1, num)
    
    # plot the individual popularities
    df_top10_pop.loc[['popularity2018'],track].plot(marker='', color=palette(num), linewidth=2.5)
    
    # same limits for every subplot
    plt.ylim(90,100)
    
    # get current position of the ticks
    locs, labels = plt.xticks()

    # add ticks with custom labels
    mylabels = ['','15th Nov'] # a bit ugly but it works
    plt.xticks(locs, mylabels)

    # not ticks everywhere
    if num in range(10) :
        plt.tick_params(labelbottom=False)
        
    # add title
    plt.title(track, loc='left', fontsize=10, fontweight=0, color=palette(num))
    
# add general title
plt.suptitle("Popularity of Top 10 Tracks during Nov 2018", fontsize=13, fontweight=0, color='black', style='italic');


# Need data from coming months so that visualization can be better(I mean Pretty!)
