import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regressor
from sklearn.metrics import mean_squared_error

# Reads in your dataset
  #dataset = pd.read_csv('/content/gdrive/My Drive/imdb_movies_shows.csv', header=None, skiprows=1)
#Habibuls Path
dataset = pd.read_csv('/Users/habibulhuq/Downloads/AIFinalProject/imdb_movies_shows.csv', header=None, skiprows=1)

# Assign meaningful column names
column_names = ['title', 'type', 'release_year', 'age_certification', 'runtime', 'genres', 'production_countries', 'seasons', 'imdb_id', 'imdb_score', 'imdb_votes']
dataset.columns = column_names

dataset.head()

#Breaks up the data to test/train sets

# This will drop all the rows with missing values in imdb_score and runtime
dataset = dataset.dropna(subset=['imdb_score', 'runtime'])

#predicting the IMDb scores based on the movie or show's runtime. So X is 'runtime' and y is 'imdb_score'
X = dataset['runtime'].values.reshape(-1, 1)
y = dataset['imdb_score']
print("runtime: \n", X ,"\n\n")
print("imdb_score: \n",y)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the Random Forest model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Checks / validates the accuracy   Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

#Displays a confusion matrix 1
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=X_test, scatter_kws={'alpha':0.3})
plt.xlabel("Actual IMDb Scores")
plt.ylabel("Actual RunTime")
plt.title("Actual RunTime vs Actual IMDb Scores with Regression Line")
plt.show()

#Displays a confusion matrix 2
import matplotlib.pyplot as plt

plt.scatter(y_test, X_test)
plt.xlabel("Actual IMDb Scores")
plt.ylabel("Actual RunTime")
plt.title("Actual RunTime VS Actual IMDb Scores")
plt.show()

#Displays a confusion matrix 3
import matplotlib.pyplot as plt

plt.scatter(y_pred,X_test)
plt.xlabel("Actual Runtime Scores")
plt.ylabel("Predicted IMDb Scores")
plt.title("Actual Runtime vs Predicted IMDb Scores")
plt.show()

#Displays a confusion matrix 4
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual IMDb Scores")
plt.ylabel("Predicted IMDb Scores")
plt.title("Actual IMDb Scores vs Predicted IMDb Scores")
plt.show()

#Displays a confusion matrix 5
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.3})
plt.xlabel("Actual IMDb Scores")
plt.ylabel("Predicted IMDb Scores")
plt.title("Actual IMDb Scores vs Predicted IMDb Scores with Regression Line")
plt.show()

#Pickles (Serialize) your model / encoding scheme
save_directory = '/Users/habibulhuq/Downloads/AIFinalProject/model_random_forest.pkl'

with open(save_directory, 'wb') as model_file:
    pickle.dump(model, model_file)