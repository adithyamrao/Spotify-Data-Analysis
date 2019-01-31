# Spotify_ML

# Problem Statement and Motivation
Our goal is to predict the number of followers a Spotify playlist will garner based on song-level categorical features and quantitative audio features. Once we have constructed our predictive model, we write an algorithm to assemble playlists from songs that are quantitatively similar to a user-specified song. Our model can then be used on these playlists to predict playlist success.

# Introduction and Description of Data

We assembled a dataset of 1628 playlists totaling 85,313 songs using the python [Spotify API](https://github.com/plamere/spotipy). We downloaded playlists created by Spotify, as these are the most visible playlists on the platform. Our dataset contains qualitative features, such as key, time signature, mode, if the track has explicit language, etc., and quantitative audio features for each song, like acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, and valence.

Starting with the song-level data, we aggregated to the playlist level by performing mean aggregation on the quantitative features and majority-vote aggregation on the categorical features. This produces a data frame consisting of 1628 rows for 1628 playlists and ∼20 column features. These include the features listed above, as well as a quantitative feature describing the number of tracks on each playlist, and six categorical features describing the use of certain key words in track titles in the playlist—e.g., “remix,” “deluxe,” etc.

The Spotify API does not provide song-level genre information, but it does provide genre information at the level of the artist. We queried the genres of each artist. We mapped artist genres to song genres based on first artist listed for each song, and from there to playlist genres considering the genre of each song in the playlist. Our dataset has 1,230 unique genres and many artists have more than one genre listed. The most popular and recognizable genres, 26 in total, were determined based on frequency of occurrence in the entire dataset, and indicator variables were made for each of these popular genres. A positive observation was assigned for each genre that was listed for an artist (so many artists/songs had multiple positive observations). These indicator variables were then treated the same as all other categorical variables by using majority-vote aggregation by playlist; for example, if most the tracks in a playlist had rock listed as a genre then that playlist was given a positive observation for rock.  

# Literature Review/Related Work
The success of our models to predict playlist popularity is dependent on being able to define the audio features of each playlist. The Echo Nest, a company that Spotify acquired in 2014, has done much of the work on algorithmically determining the audio features of a song. The [Echo Nest's Analyzer API](http://docs.echonest.com.s3-website-us-east-1.amazonaws.com/_static/AnalyzeDocumentation.pdf), which is now part of the Spotipy API, can quantify a song's musical content, such as tempo, loudness, timbre and sections, which are defined as  "large variations in rhythm or timbre, e.g. chorus, verse, bridge, guitar solo, etc..".  These quantities can be combined to form a song's second-order attributes, such as energy or danceability. The [Spotify API audio features](https://developer.spotify.com/web-api/get-audio-features/) method, which supplies second-order song attributes for each song, has been very helpful in improving our playlist-popularity regression.

Creating new playlists requires finding songs that are similar. The earliest work on machine playlist creation was in quantitatively finding similarity between songs. Logan and Salomon (2001) implemented an alogrithm to find the "distance" between songs by computing their spectral features. The assumption is that songs with similar spectral features, or signatures, are most likely similar. This method can find songs that were either in the same genre, by the same artist or on the same album. We use a similar distance measure to create playlists for each genre. Berenzweig et al. (2003) extended the work of Logan and Salomon (2001) by including subjective information about artist similarity from user surveys, and web scraping music databases and online playlists. They used these data to build similarity matrices that give the probability of two artists being on the same playlists. We use the strategy of Berenzweig et al. (2003), computing the distance between songs, in building playlists. 

# Modeling Approach and Project Trajectory
After constructing the playlist data frame, we explored the influence of different predictors on the number of followers (the response variable). The first thing we noticed is that the response variable is heavily imbalanced (right-skewed), with a small number of extremely popular playlists:

![logfollowers](/Images/Followers_hist)

 We therefore predict the log(followers) for a given playlist, rather than the absolute number of followers. Playlists with zero followers (26) were removed from the dataset because it was assumed that these playlists may be not representative of the population of followed playlists - they may be intended to be listened to without users choosing to follow them (i.e. soothing background noise).

Interaction terms were included based on observation of predictor interaction. Loudness influenced the spread of other song characteristics and danceability was influenced by the speechiness and tempo of the song (see ipython notebook for more details).

## KNN
We started with KNN, as it is one of the simplest regression models. As a baseline, a KNN model was fit to all predictors in our data set to predict the log of followers for each playlist. This model performed poorly, with an R<sup>2</sup> of 0.05 on the testing set. We then fit a KNN model on the quantitative features only for each playlist in order to prevent the predictions from being too dependent on the indicator variables. This increased the R<sup>2</sup> to 0.23. We finally included the interaction terms of loudness and danceablilty for the final KNN regression, as we identified these as important predictors in our EDA. This increased the R<sup>2</sup> once again to 0.33.

![KNN Regression](/Images/KNN_REGRESSION.png)

## Regularized (Ridge) Regression 
The next approach we evaluated was Ridge regression. We first fit the log(followers) using only playlist popularity as predictor. This yielded an R<sup>2</sup> score of 0.12 on the test set. We then added in all the predictors, as well as second-order polynomial features for the continuous predictors, and interaction terms between loudness, danceability, speechiness, and tempo, as our EDA revealed these to have important interactions. The test R<sup>2</sup> score in this case soared to 0.48. We used cross-validation to estimate the optimal regularization parameter, which we found to be 0.1. The mean cross-validation score on the training set for this model was 0.45.

We determined which predictors are most important by computing the absolute magnitudes of their fitted coefficients. These are the top 30 predictors:
![ridge_top](/Images/ridge_top_predictors.png)

An interesting result here is that the number of tracks seems to be by far the most important predictor of success -- more so than even mean song popularity -- as the top two predictors are both num_tracks variables.

These are the bottom 30 predictors:
![ridge_bot](/Images/ridge_bot_predictors.png)

Based on this figure, the Key variables seem to be largely unimportant -- four of twelve of them fall into the bottom thirty predictors. Aside from the Key variables, we see an assortment of interaction terms mainly involving loudness.

## Gradient Boosted Regression Tree
Gradient Boosted Regression was used to predict playlists to see if we could improve on typical methods of regression. All quantitative and indicator variables were used as predictors, including interaction terms discussed above. Data was split into test and training sets, and the training set was used to fit the model. Cross validation was used to determine the optimal model fit, which required a grid search for learning rate (range 0.1-0.01), maximum depth (4-12), minimum samples per leaf (3-7) and maximum number of features used for split decision (0.3-0.6). The optimal parameters were a learning rate of 0.01, maximum depth of 12, maximum number of features used of 0.3 and minimum samples per leaf of 7. The high value for minimum samples per leaf may in part have been chosen to compensate for the high maximum depth. Then a second grid search with cross validation was performed to find the optimal learning rate given the optimal parameters (range of 0.1-0.001 was used). The optimal learning rate remained 0.01, resulting in an $R^2$ score of 0.97, which is very high, likely due to overfitting to the training set, which is shown by the wide separation between test and train error in the deviance plot shown below.  

![Deviance Plot](/Images/GBRTree_deviance_plot)

The most important quantitative predictors for tree splits were popularity and number of tracks, shown in the figure below, as expected because users will generally want popular songs and a large number of songs in a playlist. The interaction of these predictors with loudness were also important. The most important indicator variables (which will generally always have a lower number of splits because there are only 2 options to choose to split on 1 or 0) were genre related - pop and rock - and word related - mention of "remastered" or "remix" in song titles.

![Predictor Importance](/Images/GBRTree_predictors)

## MLP
The new method that was implemented is a neural network method called Multi-layer Perceptron. There is an sklearn package called MLPRegressor that can be used to implement MLP regression. The process was very similar to gradient boosted regression in that parameters were fit in a two-step process. The regularization parameter (alpha, range 0.1-0.00001) was first fit, and then activation, which is the function used for the hidden layer and the number of hidden layers (range 50-200). The optimal parameters were a regularization parameter of 0.1, a logistic function for the activation function, and using 200 hidden layers. 

# Results and Conclusions

In general, all the models, after being fit for optimal parameters, did relatively well at predicting the natural logarithm of the number of followers. All models were improved by cross validation to find optimal parameters. The simplest models considered -- KNN using all predictors and Ridge using only popularity -- did a very poor job of predicting the log(followers), but both were vastly improved when new variables (specifically interaction terms) were included, and when optimal parameters were used, as shown in the KNN plot of score versus number of points used to predict. It was also observed that model scores were very sensitive to how the data was split (train/test and splits for cross-validation), which may be due to variability in the data. This could be prevented in the future by using a larger data set. 

|                | Cross validation score |
|----------------|------------------------|
| KNN            |      0.35              |
| Ridge          |      0.45              |
| Gradient Boost |      0.55              |
| MLP            |      0.44              |

The model that does the best based on cross validation scores using the training data set is Gradient Boosted Regression Tree with a 3-fold cross validation R<sup>2</sup> score on the training set of 0.55. This model was then evaluated using the test set and has a test R<sup>2</sup> score of 0.54. The residual plot, shown below, shows that the model does a good job overall. In general, the model struggles to predict the natural log of followers for playlists that have a low number of followers, shown by the larger residuals. The poor prediction for unpopular playlists could be due to factors not captured in our model. For example, some of the playlists with the lowest number of followers also have obscure and uninformative names like dw_g and dw_c.  

![Residual Plot Test Set](/Images/GBRTree_residual)

# Future Directions

With more time, there are several ways we would try to improve this work. First, we would download more playlists from archives other than Spotify. Here we work with an ensemble of ~1600 playlists, which is enough to draw broad conclusions, but our model evaluations would be more robust given more data. This would be particularly valuable for modeling playlists with extreme numbers of followers -- either very few, or large numbers. Another source of additional data would be playlists created by individual consumers, which are available on websites such as "The Art of the Mix." We could also explore user artists lists, because artists co-occurring in a user's collection are probably linked, and this information could be used to construct better playlists for evaluation by our model. Similarly, we could obtain associated acts from Wikipedia to build playlists with similar artists. Finally, it would be valuable to investigate predictors capturing playlists with very low numbers of followers. Some possibilities would be to look at playlist titles and how they are made available to users. 

We implemented a simple algorithm to generate playlists based on an input song (Implemented in Playlist_maker.ipynb) Our measure of a good playlist was one that has songs with similar attributes and genres. To create playlists, we used the distance measure of Berenzweig et al. (2003).  We measure the Euclidian distance between the input song’s audio features and the attributes of all other songs in our dataset. We then find the "closest" songs that share the same genre as the input song. Here is a playlist generated by our algorithm:

Kanye West: Through The Wire

artist | track_name 
 --- | ---
Kanye West | Hey Mama
Frank Ocean | Swim Good
Nicki Minaj | High School
Alicia Keys | Girl On Fire - Inferno Version
Chris Brown | Back To Sleep
Kanye West | Gold Digger
The Diplomats | Dipset Anthem
Juelz Santana | Dipset (Santana's Town)
Kanye West | Through The Wire
Ashanti | Helpless (feat. Ja Rule)
Sage The Gemini | Good Thing
Missy Elliott | One Minute Man (feat. Ludacris)
Kanye West | Touch The Sky
Chris Brown | Yo (Excuse Me Miss)
Busta Rhymes | Girlfriend
B.o.B | So Good
Kanye West | We Don't Care
Kanye West | Black Skinhead
Busta Rhymes | Calm Down (feat. Eminem)
Trey Songz | 1x1 

Six of the twenty songs on the playlist share the same artist (Kanye West). If we had more time, we would input the playlist to one of our models to get the predicted number of followers.  We would then add and replace tracks from the playlist from our list of “closest” tracks. Each time we added or replaced a song we would run the playlist through our model to predict the number of followers. Our final playlist would be the playlist with the most predicted followers. 
