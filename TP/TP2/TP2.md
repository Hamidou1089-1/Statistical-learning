```python
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import pandas as pd
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import statsmodels.api as sm
from scipy import stats
import itertools
```

### ENSIMAG – Grenoble INP – UGA - Academic year 2024-2025
# Introduction to Statistical Learning and Applications ([website](https://github.com/ISLA-Grenoble/2025-main))

- Pedro L. C. Rodrigues -- `pedro.rodrigues@inria.fr`

- Alexandre Wendling -- `alexandre.wendling@univ-grenoble-alpes.fr`

***

### ⚠️ General guidelines for TPs

Each team shall upload its report on [Teide](https://teide.ensimag.fr/) before the deadline indicated at the course website. Please
**include the name of all members** of the team on top of your report.
The report should contain graphical representations and explanatory text. For each graph, axis names should be provided as well
as a legend when it is appropriate. Figures should be explained by a few sentences in the text. Answer to
the questions in order and refer to the question number in your report. Computations and
graphics have to be performed in `python`. The report should be written as a jupyter notebook. This is a file format that allows users to format documents containing text written in markdown and `python` instructions. You should include all of the `python` instructions that you have used in the document so that it may be possible to replicate your results.

***

# 🖥️ TP2: Principal components regression in genetics

The goal of this TP session is to use genetic markers to predict the geographical origin of a set of indians from South, Central, and North America. We propose to build two regression linear models to predict the latitude and longitude of an individual based on its genetic markers. Because the number of markers (p = 5709) is larger than the number of samples (N = 494), the predictors of the regression model will be the outputs of a principal component analysis (PCA) performed on the genetic markers. A genetic marker is encoded 1 if the individual has a mutation, 0 elsewhere.

## ▶️ Exercise 1: Data visualization (1 point)

NB: To do this exercise you will have to install packages `geopandas` and `geodatasets`.

Download dataset `NAm2.txt` from [here](https://github.com/ISLA-Grenoble/2025-main/blob/main/TP/TP2/NAm2.txt). Each row of the dataset corresponds to an individual and the columns have explicit names. The third column contains the names of the tribes to which each individual pertains. Columns 7 and 8 contain the latitude and the longitude and from Column 9 onwards are genetic markers, which are encoded are 0 or 1. Run the code described below and explain how it works.

```
import pandas as pd
import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt

# Load the data
file_path = 'NAm2.txt'
df = pd.read_csv(file_path, delimiter=' ')

# Extract relevant columns
latitude = df.iloc[:, 6]
longitude = df.iloc[:, 7]
tribes = df.iloc[:, 2]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(longitude, latitude))

# Plotting
world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
fig, ax = plt.subplots(figsize=(8.0, 6.5))
plt.subplots_adjust(left=0.0, right=0.90, bottom=0.10, top=0.92)
world.clip([-140, -55, -25, 75]).plot(ax=ax, color='white', edgecolor='black')
marker_list = ['o', 'v', 's']
colors_list = [f'C{i}' for i in range(9)]
for i, tribe in enumerate(gdf['Pop'].unique()):
    members_tribe = gdf[gdf['Pop'] == tribe]
    ax.scatter(members_tribe['long'], members_tribe['lat'],
               marker=marker_list[i//9],
               color=colors_list[i%9], label=tribe)
ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
ax.set_title('Tribes Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
fig.show()
```


```python
# Pack a importé


```


```python


# Load the data
file_path = 'NAm2.txt'
df_original = pd.read_csv(file_path, delimiter=' ')

df_copie = df_original.copy()

# Extract relevant columns
latitude = df_copie.iloc[:, 6]
longitude = df_copie.iloc[:, 7]
tribes = df_copie.iloc[:, 2]

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(df_copie, geometry=gpd.points_from_xy(longitude, latitude))

# Plotting
world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
fig, ax = plt.subplots(figsize=(8.0, 6.5))
plt.subplots_adjust(left=0.0, right=0.90, bottom=0.10, top=0.92)
world.clip([-140, -55, -25, 75]).plot(ax=ax, color='white', edgecolor='black')
marker_list = ['o', 'v', 's']
colors_list = [f'C{i}' for i in range(9)]
for i, tribe in enumerate(gdf['Pop'].unique()):
    members_tribe = gdf[gdf['Pop'] == tribe]
    ax.scatter(members_tribe['long'], members_tribe['lat'],
               marker=marker_list[i//9],
               color=colors_list[i%9], label=tribe)
ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
ax.set_title('Tribes Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
```




    Text(83.17879444959422, 0.5, 'Latitude')




    
![png](TP2_files/TP2_3_1.png)
    



```python
# Preparing data for stat analysis
df_modified = df_copie.drop(df_original.columns[0:7], axis=1)

# Overview
print("\n\n===============Data Info=========================================\n\n")
display(df_original.info())
print("\n\n=================================================================\n\n")
print("\n\n===============Description of the date set=======================\n\n")
display(df_original.describe())
# Null value
print("\n\n===============Checking null value===============================\n\n")
display(df_original.isnull().sum())
print("\n\n===============Overviewing the head data=========================\n\n")
display(df_original.head())
print("\n\n=================================================================\n\n")
print(f"Dimension: {df_original.shape}")


```

    
    
    ===============Data Info=========================================
    
    
    <class 'pandas.core.frame.DataFrame'>
    Index: 494 entries, Chipewyan29 to Pima1051
    Columns: 5717 entries, IndivID to L678.218
    dtypes: float64(2), int64(5712), object(3)
    memory usage: 21.6+ MB



    None


    
    
    =================================================================
    
    
    
    
    ===============Description of the date set=======================
    
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndivID</th>
      <th>PopID</th>
      <th>sex</th>
      <th>lat</th>
      <th>long</th>
      <th>L1.125</th>
      <th>L1.130</th>
      <th>L1.135</th>
      <th>L1.140</th>
      <th>L1.142</th>
      <th>...</th>
      <th>L677.255.553287981859</th>
      <th>L677.259</th>
      <th>L677.263</th>
      <th>L677.267</th>
      <th>L678.202</th>
      <th>L678.206</th>
      <th>L678.209.848101265823</th>
      <th>L678.210</th>
      <th>L678.214</th>
      <th>L678.218</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.0</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>...</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
      <td>494.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2064.495951</td>
      <td>668.153846</td>
      <td>0.0</td>
      <td>8.451194</td>
      <td>-80.973279</td>
      <td>0.002024</td>
      <td>0.002024</td>
      <td>0.002024</td>
      <td>0.002024</td>
      <td>0.002024</td>
      <td>...</td>
      <td>0.107287</td>
      <td>0.263158</td>
      <td>0.212551</td>
      <td>0.004049</td>
      <td>0.006073</td>
      <td>0.046559</td>
      <td>0.040486</td>
      <td>0.888664</td>
      <td>0.014170</td>
      <td>0.004049</td>
    </tr>
    <tr>
      <th>std</th>
      <td>649.805868</td>
      <td>309.406342</td>
      <td>0.0</td>
      <td>24.159398</td>
      <td>15.441783</td>
      <td>0.044992</td>
      <td>0.044992</td>
      <td>0.044992</td>
      <td>0.044992</td>
      <td>0.044992</td>
      <td>...</td>
      <td>0.309792</td>
      <td>0.440794</td>
      <td>0.409527</td>
      <td>0.063564</td>
      <td>0.077770</td>
      <td>0.210905</td>
      <td>0.197296</td>
      <td>0.314867</td>
      <td>0.118311</td>
      <td>0.063564</td>
    </tr>
    <tr>
      <th>min</th>
      <td>702.000000</td>
      <td>81.000000</td>
      <td>0.0</td>
      <td>-41.000000</td>
      <td>-108.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.250000</td>
      <td>811.000000</td>
      <td>0.0</td>
      <td>-10.000000</td>
      <td>-96.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2157.500000</td>
      <td>831.000000</td>
      <td>0.0</td>
      <td>9.500000</td>
      <td>-77.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2543.750000</td>
      <td>838.000000</td>
      <td>0.0</td>
      <td>17.000000</td>
      <td>-70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2800.000000</td>
      <td>849.000000</td>
      <td>0.0</td>
      <td>59.550000</td>
      <td>-52.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 5714 columns</p>
</div>


    
    
    ===============Checking null value===============================
    
    



    IndivID                  0
    PopID                    0
    Pop                      0
    Country                  0
    Continent                0
                            ..
    L678.206                 0
    L678.209.848101265823    0
    L678.210                 0
    L678.214                 0
    L678.218                 0
    Length: 5717, dtype: int64


    
    
    ===============Overviewing the head data=========================
    
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IndivID</th>
      <th>PopID</th>
      <th>Pop</th>
      <th>Country</th>
      <th>Continent</th>
      <th>sex</th>
      <th>lat</th>
      <th>long</th>
      <th>L1.125</th>
      <th>L1.130</th>
      <th>...</th>
      <th>L677.255.553287981859</th>
      <th>L677.259</th>
      <th>L677.263</th>
      <th>L677.267</th>
      <th>L678.202</th>
      <th>L678.206</th>
      <th>L678.209.848101265823</th>
      <th>L678.210</th>
      <th>L678.214</th>
      <th>L678.218</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chipewyan29</th>
      <td>2012</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan31</th>
      <td>2156</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan33</th>
      <td>2381</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan35</th>
      <td>2382</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Chipewyan37</th>
      <td>2383</td>
      <td>811</td>
      <td>Chipewyan</td>
      <td>Canada</td>
      <td>AMERICA</td>
      <td>0</td>
      <td>59.55</td>
      <td>-107.3</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5717 columns</p>
</div>


    
    
    =================================================================
    
    
    Dimension: (494, 5717)


## ▶️ Exercise 2: Multiple linear regression (2 points)

Using **only** the genetic markers as predictors, you will estimate a multiple linear regression model to predict the longitude of each individual.

You will proceed in several steps.

**(a)** First, try to estimate the coefficients of the multiple linear regression using the expression seen in class

$$\hat{\beta} = (X^\top X)^{-1}X^\top y$$

You should proceed as we did in TP1 using `numpy.linalg.solve` to obtain the values of $\beta$.

Did you run into any errors? What is going on? Relate your answer to the fact that $\text{rank}(X) < p$, where $X \in R^{N*p}$ is the data matrix.

**Mathematical interpretation**: When p > N, the matrix $X^T X$ is singular (not invertible), which means there's no unique solution to the normal equations. This is a fundamental problem in high-dimensional statistics.



```python
X_matrix_predictors = df_modified.iloc[:, 1:]
X_matrix_predictors['intercept'] = 1
y_pred = df_modified.iloc[:, 0]

#df_modified.corr()
```


```python

print("\nThe matrix X^T X is not invertible because of multicollinearity in the data and because the number of predictors (p) is greater than the number of observations (N), making rank(X_matrix_predictors) < p.\n")


rank_X = np.linalg.matrix_rank(X_matrix_predictors)
print(f"Rank de X: {rank_X}")
print(f"Number of predictors (p): {X_matrix_predictors.shape[1]}")
print(f"does the rank(X) < p? {rank_X < X_matrix_predictors.shape[1]}")

```

    
    The matrix X^T X is not invertible because of multicollinearity in the data and because the number of predictors (p) is greater than the number of observations (N), making rank(X_matrix_predictors) < p.
    
    Rank de X: 494
    Number of predictors (p): 5710
    does the rank(X) < p? True


**(b)** Use function `numpy.linalg.lstsq` to estimate the coefficients (it may take a few seconds to get a result).

And now? Did you get any errors? Why is that?

Relate your answer to the difference between functions `numpy.linalg.solve` and `numpy.linalg.lstsq`.

You can check the documention for both functions as well as [this](https://netlib.org/lapack/lug/node27.html) link for more information.


```python
beta = np.linalg.lstsq(X_matrix_predictors, y_pred, rcond=None)[0]
```


```python
len(beta)
```




    5710



## The principal difference between solve and lstsq :
- solve : resolve $Ax=b$ where A is a squared matrix invertible
- **numpy.linalg.lstsq** finds the solution that minimizes $\lVert Ax-b \rVert^{2}$, even when A is not of full rank. It uses SVD decomposition to find a solution, making it more robust than **solve**.

 **(c)** We will now use `sklearn` to do our linear regression with the help of class `sklearn.linear_model.LinearRegression` whose documentation is available [here](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html). Note that every estimator from `sklearn` has a `fit` and a `predict` method, which are used to calculate coefficients and predict values (see [here](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics) for more info). In our current case, we can do:

```
# select only the genetic markers as predictors
predictors = df.columns[8:]
# create the design matrix
X = df[predictors].values
# get the observed values to predict
y = df['long']
# fit a multiple linear regression model
lr = LinearRegression()
lr.fit(X, y)
```

You should not run into errors now, since `sklearn` also uses `lstsq` to solve the normal equations, as shown [here](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/linear_model/_base.py#L682) (though it uses the `scipy` implementation instead of the `numpy` for "historical" reasons). Check the values of the estimated coefficients stored as an attribute in `lr.coef_`, are they the same as the ones obtained in item **(b)**? Probably not. This is because `sklearn` re-centers the predictors before estimating the coefficients of the linear regression, as shown [here](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/linear_model/_base.py#L622). What would be a practical reason for doing such re-centering systematically? Hint: it has to do with how to interpret the intercept of the model.


```python
# select only the genetic markers as predictors
predictors = df_original.columns[8:]    # create the design matrix
X = df_original[predictors].values      # get the observed values to predict
y = df_original['long']

# fit a multiple linear regression model
lr = LinearRegression()
lr.fit(X, y)

beta_coef = beta[1:]


plt.scatter(beta_coef, lr.coef_)

```




    <matplotlib.collections.PathCollection at 0x74691d0491f0>




    
![png](TP2_files/TP2_14_1.png)
    


The reason sklearn centers predictors is to make the intercept interpretable as the expected value of y when all predictors are at their means. Without centering, the intercept would represent the expected value of y when all predictors are zero, which might not be meaningful in many contexts.



**Interpretation:**
- When predictors are centered, the intercept represents the expected value of y when all predictors are at their means
- When predictors are not centered, the intercept represents the expected value of y when all predictors are zero

So it's normal to not have a diagonal line on the scatter, because we didn't recenter the beta.

## ▶️ Exercise 3: Principal components analysis (5 points)

**(a)** Explain in a few words the main concepts and ideas underlying the principal component analysis (PCA). You should include both the geometric and statistical interpretations of PCA.
#### answer:
PCA is a dimensionality reduction technique that transforms a set of correlated variables into a set of uncorrelated variables called principal components:

- Geometric interpretation: PCA finds the directions (eigenvectors) in which the data varies the most. These directions become the principal components.
- Statistical interpretation: PCA finds linear combinations of original variables that maximize variance. The first principal component accounts for as much of the variability as possible, and each succeeding component accounts for as much of the remaining variability as possible while being orthogonal to the preceding components.

**(b)** Use the estimator defined in `sklearn.decomposition.PCA` to do a PCA on the dataset. Plot the first two dimensions of the projected data points on a scatterplot. The scattered points should have different markers and colors depending on which tribe they belong to. You can use the same color/marker style from **Exercise 2** or propose a new one.


```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 8))
marker_list = ['o', 'v', 's']
colors_list = [f'C{i}' for i in range(9)]

for i, tribe in enumerate(df_copie['Pop'].unique()):

    tribe_indices = df_copie['Pop'] == tribe


    plt.scatter(
        X_pca[tribe_indices, 0],
        X_pca[tribe_indices, 1],
        marker=marker_list[i//9],
        color=colors_list[i%9],
        label=tribe,
        alpha=0.7
    )


plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance expliquée)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance expliquée)', fontsize=12)
plt.title('Projection PCA des marqueurs génétiques par tribu', fontsize=14)
plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](TP2_files/TP2_18_0.png)
    


**Interpretation**:
The plot shows how different tribes cluster in the PCA space. If tribes that are geographically close also cluster together in the PCA space, this suggests that genetic markers contain information about geographical origin.

**(c)** Remember from our class that the results of PCA are affected when pre-processing transformations are applied to the data. We will illustrate this using `sklearn.preprocessing.StandardScaler` as per:
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)
```
Redo the 2D scatter plot from item **(b)** on the normalized version of the datast. How does it compare to your previous plot?


```python
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)

pca_std = PCA(n_components=2)
X_pca_std = pca_std.fit_transform(X_std)

plt.figure(figsize=(10, 8))
marker_list = ['o', 'v', 's']
colors_list = [f'C{i}' for i in range(9)]

for i, tribe in enumerate(df_copie['Pop'].unique()):

    tribe_indices = df_copie['Pop'] == tribe


    plt.scatter(
        X_pca_std[tribe_indices, 0],
        X_pca_std[tribe_indices, 1],
        marker=marker_list[i//9],
        color=colors_list[i%9],
        label=tribe,
        alpha=0.7
    )


plt.xlabel(f'PC1 ({pca_std.explained_variance_ratio_[0]:.2%} variance expliquée)', fontsize=12)
plt.ylabel(f'PC2 ({pca_std.explained_variance_ratio_[1]:.2%} variance expliquée)', fontsize=12)
plt.title('Projection PCA des marqueurs génétiques par tribu', fontsize=14)
plt.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```


    
![png](TP2_files/TP2_21_0.png)
    


**Effect of Standardization on PCA**:

When standardizing the data causes the clear tribal clusters from the original PCA to disappear, this reveals something fundamental about your genetic marker data:

- Why the Clusters Disappeared

    - ****Scale dominance****: In the original data (without standardization), certain genetic markers likely had larger variances than others. These high-variance markers dominated the PCA, creating the clear tribal separation you observed.
    - ****Variance equalization****: Standardization gives equal weight to all genetic markers by scaling them to unit variance. When all markers contribute equally to the PCA, the previously dominant patterns are diluted.
    - ****Information distribution****: This suggests that the tribal clustering was primarily driven by a subset of genetic markers with larger variances, rather than being a pattern distributed across all markers.

- Mathematical Explanation
The disappearance of clusters after standardization indicates that:

    - The genetic markers that best distinguish between tribes had naturally larger variances
    - When forced to contribute equally through standardization, the distinctive tribal signals become mixed with noise from less informative markers

**(d)** Given the results in **(b)** and **(c)**, what can you conclude regarding the necessity of standardizing the data points for the dataset consider in this TP?


```python
# Already answered throughout my interpretation
```

**(e)** Which percentage of variance is captured by the first two principal components? How many principal components would you keep if you would like to represent the genetic markers using a minimal number of principal components? To help answering this question, you can use a plot showing the cumulative percentage of variance as a function of the number of principal components.


```python
pca_full = PCA()
pca_full.fit(X)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

n_components = np.argmax(cumulative_variance >= 0.95) +1
print(n_components)
print(cumulative_variance[2])

plt.figure(figsize=(10, 8))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.plot(2, cumulative_variance[2], 'ro')
plt.axvline(x=2, color='b', linestyle='--', label='2 components Explained Variance')
plt.legend()
plt.show()
```

    410
    0.049039559288078294



    
![png](TP2_files/TP2_26_1.png)
    


****Interpretation :****
- When number_components = 410 we have more than 95% of the variance explained by our new predictors.
- When the number of components is equal to 2, the purcentage of variance is 4.9%, which is too low.

## ▶️ Exercise 4: Principal components regression (4 points)

**(a)** Predict the latitude and the longitude of all points from the dataset using the scores of the first 250 PCA axes. Plot the predicted spatial coordinates using the same style and structure from **Exercise 1** and compare the results from each plot. What can you conclude? Does the new map illustrate somehow too optimistically (or too pessimistically) the ability to find geographical origin of individuals outside the database from its genetic markers? Justify your answer.


```python
pca = PCA(n_components=250)
X_pca = pca.fit_transform(X)

# To predict latitude
lat_model = LinearRegression()
lat_model.fit(X_pca, df_original['lat'])
lat_pred = lat_model.predict(X_pca)

# To predict longitude
long_model = LinearRegression()
long_model.fit(X_pca, df_original['long'])
long_pred = long_model.predict(X_pca)


# Create a DataFrame with predicted coordinates
predicted_coords = pd.DataFrame({
    'lat_pred': lat_pred,
    'long_pred': long_pred,
    'Pop': df_original['Pop']  # Tribe information for coloring
})
```


```python
# Create two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.02)

# Common map background for both plots
for ax in axes:
    world.clip([-140, -55, -25, 75]).plot(ax=ax, color='white', edgecolor='black')
    ax.set_xlabel('Longitude')

# Set specific titles and y-axis label
axes[0].set_title('Actual Tribe Locations')
axes[0].set_ylabel('Latitude')
axes[1].set_title('Predicted Tribe Locations (250 PCA Components)')

# Plot actual locations in left subplot
for i, tribe in enumerate(df_copie['Pop'].unique()):
    members_tribe = df_copie[df_copie['Pop'] == tribe]
    axes[0].scatter(members_tribe['long'], members_tribe['lat'],
                   marker=marker_list[i//9],
                   color=colors_list[i%9],
                   label=tribe,
                   alpha=0.7)

# Plot predicted locations in right subplot
for i, tribe in enumerate(df_copie['Pop'].unique()):
    indices = df_copie['Pop'] == tribe
    axes[1].scatter(long_pred[indices], lat_pred[indices],
                   marker=marker_list[i//9],
                   color=colors_list[i%9],
                   label=tribe,
                   alpha=0.7)

# Add a single legend outside the plots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.0, 0.5))

plt.show()
```


    
![png](TP2_files/TP2_30_0.png)
    


##### **Data leakage** :
Clearly these two maps are very simular, which is an error that we expressed in class, our approach which is using the same dataset for :
   - Computing the PCA transformation
   - Training the regression model
   - Evaluating the model's performance

This lead to a data leakage because information from what should be our test data has already influenced our feature extraction process.


##### **Overfitting** :
Here we can detect overfitting for several reason,
- *High dimensionality* : we're using 250 pca to predict just two variables (lattitude and longitude ) with only 494 observations. This high ratio of features to observations makes overfitting likely.
- *No training/ Testing Split* : We're evaluating the model on the same data used to train it, which will always show better performance than on unseen data.


##### *Conclusion*:
So these prediction are too optimistic, because we are overfitting the model and data leakage so of course our model seems perfect. *But i guess that's why we have exercice 5*.




**(b)** Quantify the error of the linear regression model using the mean distance between real and predicted coordinates. Beware to use `sklearn.metrics.pairwise.haversine_distances` so to correctly measure the distances between points so to take into account the curvature of the Earth. Your answer should be given in kilometers.


```python
def convert_to_radians(lat, long):
    return np.c_[np.radians(lat), np.radians(long)]
```


```python
actual_coords_rad = convert_to_radians(df_original['lat'], df_original['long'])

predicted_coords_rad = convert_to_radians(lat_pred, long_pred)

distances_rad = haversine_distances(actual_coords_rad, predicted_coords_rad)

earth_radius_km = 6371
distance_km = distances_rad * earth_radius_km

mean_distance_km = np.mean(np.diag(distance_km))
print(f"Mean distance between actual and predicted coordinates: {mean_distance_km:.2f} km")
```

    Mean distance between actual and predicted coordinates: 625.91 km


##### *Why an error so high ?*
The hight average error of 619.20 km is not very surprising, as we compare both maps on question a), we just said that the two look like because the geographic region on tribe contains the reals location of the actual tribe, but there is a spreading in the predicted location.

This illustrates an important concept in predictive modeling: models can appear to perform well at a macro level (preserving the general distribution of tribes across the Americas) while still having significant errors at the individual level. A 619.20 km error is substantial - approximately the distance between major cities - and would place individuals in entirely different regions or countries in many cases.

The high error despite using 250 principal components suggests that either:
1. The relationship between genetic markers and geographic location is more complex than a linear model can capture
2. Some genetic markers may be shared across distant populations due to historical migration patterns
3. The genetic-geographic relationship has inherent variability that limits prediction precision

## ▶️ Exercise 5: PCR and cross-validation (6 points)

Our goal now is to build the best model to predict individual geographical coordinates. 

For this, you will run a linear regression to predict latitudes and longitudes. Note that `sklearn.linear_model.LinearRegression` can naturally handle the fact of having two sets of coefficients. We will use ten-fold cross-validation to helps us choose the number of principal axes that we should keep. You should report the errors in terms of kilometers as done in **Exercise 4(b)**.

**(a)** Recall in a few words the principle of cross-validation. Explain why this procedure is useful when building a predictive model. Your answer should mention different strategies to handle datasets in which the samples are not IID.

#### ***Cross-Validation***:
Cross-validation is a resampling procedure used to evaluate models on limited data. Given a dataset with N observations, k-fold cross-validation involves:

1. Dividing the dataset into k equal-sized folds
2. For each of the k folds:
   - Train the model using k-1 folds (training set)
   - Evaluate the model on the remaining fold (test set)
3. Average the k performance metrics to get a more robust estimate of model performance

This approach is particularly valuable because:
- It provides a more reliable estimate of model performance on unseen data
- Each observation is used for both training and testing
- The variance of the estimated performance is reduced

As we saw in Exercise 4, using the same data for both training and testing leads to overfitting and data leakage, resulting in overly optimistic performance estimates. Cross-validation helps mitigate these issues by ensuring model evaluation is performed on data not used during training.

For non-IID data (where observations are not independent and identically distributed), standard k-fold cross-validation may be inappropriate. Alternative strategies include:

- Stratified k-fold: Maintains the same proportion of target classes in each fold (for classification)
- Group k-fold: Ensures observations from the same group (e.g., same individual or tribe) stay together
- Time-series split: For temporal data, trains on past observations and tests on future ones
- Leave-one-group-out: Leaves out one group at a time, useful when observations naturally cluster

**(b)** Based on the structure of the dataset being used, such as the different countries of the individuals and the order in which the rows of the dataframe are provided, explain which choice of cross-validation iterator from [here](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) seems the most adequate for our context.

##### ***Cross validation iterators:***
Based on the structure of our dataset, the most appropriate cross-validation iterator would be Group k-fold. This choice is justified by several characteristics of our data:

1. The data is organized by tribes, with individuals from the same tribe likely having similar genetic markers due to shared ancestry.

2. The exercise specifically mentions that the order of rows in the dataframe might be significant, suggesting that samples are grouped by tribe or geographical region.

3. In this genetic study context, we want to evaluate how well our model generalizes to entirely new tribes or populations, rather than just new individuals from already-seen tribes.

Group k-fold ensures that all samples from the same tribe are either entirely in the training set or entirely in the test set for each fold. This prevents data leakage between tribes and provides a more realistic assessment of how our model would perform when predicting the geographical origin of individuals from previously unseen tribal populations.

Using standard KFold would be inappropriate as it might split tribes across training and testing sets, leading to artificially inflated performance metrics and not properly evaluating the model's generalization capabilities.

**(c)** We first assess the quality of the PCR fit for `n_components=4`. Note that you should be careful in avoiding [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) problems when doing the PCA followed by a multiple linear regression. You should use the pipeline interface from scikit-learn with `sklearn.pipeline.make_pipeline` to facilitate your task. Be sure to evaluate the errors as done in **Exercise 4(b)**.


```python
pca = PCA(n_components=410)
lr = LinearRegression()
est = make_pipeline(pca, lr)

cv = GroupKFold(n_splits=10)
groups = df_original['Pop']

y_coords = df_original[['lat', 'long']].values

train_errors = []
test_errors = []

```


```python
for train_index, test_index in cv.split(X, y_coords, groups=groups):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_coords[train_index], y_coords[test_index]


    est.fit(X_train, y_train)

    y_train_pred = est.predict(X_train)
    y_test_pred = est.predict(X_test)

    y_train_rad = np.radians(y_train)
    y_train_pred_rad = np.radians(y_train_pred)
    y_test_rad = np.radians(y_test)
    y_test_pred_rad = np.radians(y_test_pred)

    train_distances = haversine_distances(y_train_rad, y_train_pred_rad).diagonal() * 6371
    test_distances = haversine_distances(y_test_rad, y_test_pred_rad).diagonal() * 6371

    train_errors.append(np.mean(train_distances))
    test_errors.append(np.mean(test_distances))

# Overall mean errors
mean_train_error = np.mean(train_errors)
mean_test_error = np.mean(test_errors)

print(f"Mean training error: {mean_train_error:.2f} km")
print(f"Mean test error: {mean_test_error:.2f} km")
```

    Mean training error: 186.87 km
    Mean test error: 2034.21 km


**(d)** Repeat the analysis from item **(b)** but changing `n_components` between 2 and 440 in steps of 10. Plot the mean training and test errors versus the number of principal components. Attention, the errors should be given in kilometers.


```python
train_errors_mean = []
test_errors_mean = []
for n_components in range(2, 440, 10):
    pca = PCA(n_components=n_components)
    lr = LinearRegression()
    est = make_pipeline(pca, lr)

    cv = GroupKFold(n_splits=10)
    groups = df_original['Pop']

    y_coords = df_original[['lat', 'long']].values

    train_errors = []
    test_errors = []

    for train_index, test_index in cv.split(X, y_coords, groups=groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_coords[train_index], y_coords[test_index]


        est.fit(X_train, y_train)

        y_train_pred = est.predict(X_train)
        y_test_pred = est.predict(X_test)

        y_train_rad = np.radians(y_train)
        y_train_pred_rad = np.radians(y_train_pred)
        y_test_rad = np.radians(y_test)
        y_test_pred_rad = np.radians(y_test_pred)

        train_distances = haversine_distances(y_train_rad, y_train_pred_rad).diagonal() * 6371
        test_distances = haversine_distances(y_test_rad, y_test_pred_rad).diagonal() * 6371

        train_errors.append(np.mean(train_distances))
        test_errors.append(np.mean(test_distances))

    train_errors_mean.append(np.mean(train_errors))
    test_errors_mean.append(np.mean(test_errors))


components_range = list(range(2, 440, 10))

plt.figure(figsize=(12, 8))
plt.plot(components_range, train_errors_mean, 'b-', label='Training Error')
plt.plot(components_range, test_errors_mean, 'r-', label='Test Error')
plt.xlabel('Number of principal Components')
plt.ylabel('Mean Error (km)')
plt.title('Training and test errors vs number of principal components')
plt.legend()
plt.tight_layout()
```


    
![png](TP2_files/TP2_44_0.png)
    


**(e)** Which model would you keep? What is the prediction error for this model? Compare it with its corresponding training error. Plot the predicted coordinates on a map as in **Exercise 4(a)**. What can you conclude?


```python

```


```python

```

## ▶️ Exercise 6: Conclusion (2 points)

Propose a conclusion to your study. You can write a paragraph about the quality of predictors versus the number of factors, possible improvements to the approach (for instance, showing what happens when using [partial least squares](https://scikit-learn.org/1.5/auto_examples/cross_decomposition/plot_pcr_vs_pls.html) instead of PCR), comment on the performance of the regression in predictions for each country separately, etc. Note that we expect a thorough presentation of the final predictive model as well as an interpretation of it, not simply a bunch of `python` code lines.


```python

```
