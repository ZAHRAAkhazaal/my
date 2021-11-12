import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


gapminder_filepath = "C:/Users/asus/Downloads/gapminder.csv"
gapminder_data = pd.read_csv(gapminder_filepath, index_col='country')
gapminder_data.head()

%%%%%----------------------------------------------------------------
gapminder_data.info()
%%%%%------------------------------------------
  

# Setting the figure size 
sns.set(rc={'figure.figsize':(10,8)})
sns.heatmap(df.corr(), cmap='RdYlBu', square=True)



X=gapminder_data.data()
y=gapminder_data.target()
%--------------------------------------
# Check the shape of y
print("The shape of the target variable is :" X.shape )
# Check the shape of X
print("The shape of the input variable is :" y.shape )

%---------------------------------------------------------
plt.scatter(X, y)
 
from sklearn import linear_model 

linear_regreesion model= linear_regreesion (X,y)

linear_regreesion mode.fit()
linear_regreesion mode.pred(X,y)
linear_regreesion mode.score(X,y)
