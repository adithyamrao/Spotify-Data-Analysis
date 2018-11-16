
# coding: utf-8

# # Collecting Data from the Spotify Web API using Spotipy
# ### About the Spotipy Library:
# From the [official Spotipy docs](https://spotipy.readthedocs.io/en/latest/):
# 
# "Spotipy is a lightweight Python library for the Spotify Web API. With Spotipy you get full access to all of the music data provided by the Spotify platform."
# 
# ### About using the Spotify Web API:
# Spotify offers a number of [API endpoints](https://developer.spotify.com/documentation/web-api/reference/) to access the Spotify data. In this notebook, I used the following:
# 
# - [search endpoint](https://developer.spotify.com/documentation/web-api/reference/search/search/) to get the track IDs
# - [audio features endpoint](https://developer.spotify.com/documentation/web-api/reference/tracks/get-several-audio-features/) to get the corresponding audio features.<br>
# The data was collected on several days during the months of April, May and August 2018.
# 
# ### Goal of this notebook:
# The goal is to show how to collect audio features data for tracks from the [official Spotify Web API](https://developer.spotify.com/documentation/web-api/) in order to use it for further analysis/ machine learning which will be part of another notebook.
# 
# ### 1. Setting Up
# The below code is sufficient to set up Spotipy for querying the API endpoint. A more detailed explanation of the whole procedure is available in the [official docs](https://spotipy.readthedocs.io/en/latest/#installation).

# In[1]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

cid ="621685b7e9054759ac91dca2a8a0a5f8" 
secret = "3c593b4b86814535a0014c0e6b8ff0dc"

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


# ## 2. Get the Track ID Data
# The data collection is divided into 2 parts: the track IDs and the audio features. In this step, I'm going to collect 10.000 track IDs from the Spotify API.
# 
# The [search endpoint](https://developer.spotify.com/documentation/web-api/reference/search/search/) used in this step had a few limitations:
# 
# - limit: a maximum of 50 results can be returned per query
# - offset: this is the index of the first result to return, so if you want to get the results with the index 50-100 you will need to set the offset to 50 etc. <br>
# Spotify cut down the maximum offset to 10.000.
# 
# My solution: using a nested for loop, I increased the offset by 50 in the outer loop until the maxium limit/ offset was reached. The inner for loop did the actual querying and appending the returned results to appropriate lists which I used afterwards to create my dataframe.

# In[2]:


# timeit library to measure the time needed to run this code
import timeit
start = timeit.default_timer()

# create empty lists where the results are going to be stored
artist_name = []
track_name = []
popularity = []
track_id = []

for i in range(0,10000,50):
    track_results = sp.search(q='year:2018', type='track', limit=50,offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        track_id.append(t['id'])
        popularity.append(t['popularity'])
      

stop = timeit.default_timer()
print ('Time to run this code (in seconds):', stop - start)


# ## 3. Data Preparation and EDA
# In the next few cells, I'm going to do some exploratory data analysis as well as data preparation of the newly gained data.
# 
# A quick check for the track_id list:

# In[3]:


print('number of elements in the track_id list:', len(track_id))


# Ahh, looks good. Let's get it into a dataframe. 

# In[4]:


import pandas as pd

df_tracks = pd.DataFrame({'artist_name':artist_name,'track_name':track_name,'track_id':track_id,'popularity':popularity})
print(df_tracks.shape)
df_tracks.head()


# In[5]:


df_tracks.info()


# ** Sometimes, the same track is returned under different track IDs i.e., as asingle track and from album.**
# 
# This needs to be checked for and corrected if needed.

# In[6]:


# group the entries by artist_name and track_name and check for duplicates

grouped = df_tracks.groupby(['artist_name','track_name'], as_index=True).size()
grouped[grouped > 1].count()


# There are 2297 duplicate entries which will be dropped in the next cell:

# In[7]:


df_tracks.drop_duplicates(subset=['artist_name','track_name'], inplace=True)


# In[8]:


# doing the same grouping as before to verify the solution
grouped_after_dropping = df_tracks.groupby(['artist_name','track_name'], as_index=True).size()
grouped_after_dropping[grouped_after_dropping > 1].count()


# In[9]:


# One more way of checking for dupliates
df_tracks[df_tracks.duplicated(subset=['artist_name','track_name'],keep=False)].count()


# In[10]:


df_tracks.shape


# ## 4: Get the Audio Features Data
# With the [audio features endpoint](https://developer.spotify.com/documentation/web-api/reference/tracks/get-several-audio-features/) I will now get the audio features data for my 6901 track IDs.
# 
# The limitation for this endpoint is that a maximum of 100 track IDs can be submitted per query.
# 
# Again, I used a nested for loop. This time the outer loop was pulling track IDs in batches of size 100 and the inner for loop was doing the query and appending the results to the rows list.
# 
# Additionaly, I had to implement a check when a track ID didn't return any audio features (i.e. None was returned) as this was causing issues.

# In[11]:


# again measuring the time
start = timeit.default_timer()

# empty list, batchsize and the counter for None results
rows = []
batchsize = 100
None_counter = 0

for i in range(0,len(df_tracks['track_id']),batchsize):
    batch = df_tracks['track_id'][i:i+batchsize]
    feature_results = sp.audio_features(batch)
    for i, t in enumerate(feature_results):
        if t == None:
            None_counter = None_counter + 1
        else:
            rows.append(t)
            
print('Number of tracks where no audio features were available:',None_counter)

stop = timeit.default_timer()
print ('Time to run this code (in seconds):',stop - start)


# ## 5. Data Preparation
# Same as with the first dataset, checking how the rows list looks like:

# In[12]:


print('number of elements in the track_id list:', len(rows))


# In[13]:


# Load it into dataframe
df_audio_features = pd.DataFrame.from_dict(rows,orient='columns')
print("Shape of the dataset:", df_audio_features.shape)
df_audio_features.head()


# In[14]:


df_audio_features.info()


# Some columns are not needed for the analysis so I will drop them.
# 
# Also the ID column will be renamed to track_id so that it matches the column name from the first dataframe.

# In[15]:


columns_to_drop = ['analysis_url','track_href','type','uri']
df_audio_features.drop(columns_to_drop, axis=1,inplace=True)

df_audio_features.rename(columns={'id': 'track_id'}, inplace=True)

df_audio_features.shape


# In[16]:


# merge both dataframes
# the 'inner' method will make sure that we only keep track IDs present in both datasets
df = pd.merge(df_tracks,df_audio_features,on='track_id',how='inner')
print("Shape of the dataset:", df_audio_features.shape)
df.head()


# In[17]:


df.info()


# In[18]:


# Check for dulpicates
df[df.duplicated(subset=['artist_name','track_name'],keep=False)]


# Everything seems to be fine so I will save the dataframe as a .csv file.

# In[19]:


df.to_csv('SpotifyAudioFeatures11152018.csv')

