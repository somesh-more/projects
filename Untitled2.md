```python
import math
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pprint import pprint
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score 
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
df
```




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
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>nameOrig</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
      <th>isFlaggedFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9839.64</td>
      <td>C1231006815</td>
      <td>170136.00</td>
      <td>160296.36</td>
      <td>M1979787155</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>1864.28</td>
      <td>C1666544295</td>
      <td>21249.00</td>
      <td>19384.72</td>
      <td>M2044282225</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>181.00</td>
      <td>C1305486145</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>C553264065</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>181.00</td>
      <td>C840083671</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>C38997010</td>
      <td>21182.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>11668.14</td>
      <td>C2048537720</td>
      <td>41554.00</td>
      <td>29885.86</td>
      <td>M1230701703</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362615</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>339682.13</td>
      <td>C786484425</td>
      <td>339682.13</td>
      <td>0.00</td>
      <td>C776919290</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>6311409.28</td>
      <td>C1529008245</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>C1881841831</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>6311409.28</td>
      <td>C1162922333</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>C1365125890</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>743</td>
      <td>TRANSFER</td>
      <td>850002.52</td>
      <td>C1685995037</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>C2080388513</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>743</td>
      <td>CASH_OUT</td>
      <td>850002.52</td>
      <td>C1280323807</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>C873221189</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6362620 rows × 11 columns</p>
</div>




```python
df['nameDest']=df['nameDest'].apply(lambda x : x[0])
```


```python
df['step'] =df['step']%24
```


```python
df['amount'] = np.log(df['amount'])
```


```python
df['amount'].replace([np.inf, -np.inf], np.nan, inplace=True)
```


```python
df['amount'].fillna(10.84, inplace=True)
```


```python
df.drop(['nameOrig','isFlaggedFraud'], axis=1, inplace=True)
```


```python
df
```




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
      <th>step</th>
      <th>type</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>nameDest</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9.194174</td>
      <td>170136.00</td>
      <td>160296.36</td>
      <td>M</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>7.530630</td>
      <td>21249.00</td>
      <td>19384.72</td>
      <td>M</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>TRANSFER</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>C</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>CASH_OUT</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>C</td>
      <td>21182.00</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>PAYMENT</td>
      <td>9.364617</td>
      <td>41554.00</td>
      <td>29885.86</td>
      <td>M</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362615</th>
      <td>23</td>
      <td>CASH_OUT</td>
      <td>12.735766</td>
      <td>339682.13</td>
      <td>0.00</td>
      <td>C</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>23</td>
      <td>TRANSFER</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>C</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>23</td>
      <td>CASH_OUT</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>C</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>23</td>
      <td>TRANSFER</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>C</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>23</td>
      <td>CASH_OUT</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>C</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6362620 rows × 9 columns</p>
</div>




```python
df= df[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type','nameDest','isFraud']]
```


```python
df
```




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
      <th>step</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type</th>
      <th>nameDest</th>
      <th>isFraud</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.194174</td>
      <td>170136.00</td>
      <td>160296.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.530630</td>
      <td>21249.00</td>
      <td>19384.72</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>21182.00</td>
      <td>0.00</td>
      <td>CASH_OUT</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>9.364617</td>
      <td>41554.00</td>
      <td>29885.86</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362615</th>
      <td>23</td>
      <td>12.735766</td>
      <td>339682.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>CASH_OUT</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>23</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>23</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>CASH_OUT</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>23</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>23</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>CASH_OUT</td>
      <td>C</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>6362620 rows × 9 columns</p>
</div>




```python
df.to_csv('file1.csv')
```


```python
df['step'] = df['step'].astype(str)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6362620 entries, 0 to 6362619
    Data columns (total 9 columns):
    step              object
    amount            float64
    oldbalanceOrg     float64
    newbalanceOrig    float64
    oldbalanceDest    float64
    newbalanceDest    float64
    type              object
    nameDest          object
    isFraud           int64
    dtypes: float64(5), int64(1), object(3)
    memory usage: 436.9+ MB



```python
X= df.iloc[:,:8]
y= df.iloc[:,-1]
```


```python
X
```




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
      <th>step</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type</th>
      <th>nameDest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.194174</td>
      <td>170136.00</td>
      <td>160296.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.530630</td>
      <td>21249.00</td>
      <td>19384.72</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5.198497</td>
      <td>181.00</td>
      <td>0.00</td>
      <td>21182.00</td>
      <td>0.00</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>9.364617</td>
      <td>41554.00</td>
      <td>29885.86</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6362615</th>
      <td>23</td>
      <td>12.735766</td>
      <td>339682.13</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>339682.13</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6362616</th>
      <td>23</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6362617</th>
      <td>23</td>
      <td>15.657870</td>
      <td>6311409.28</td>
      <td>0.00</td>
      <td>68488.84</td>
      <td>6379898.11</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6362618</th>
      <td>23</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>6362619</th>
      <td>23</td>
      <td>13.652995</td>
      <td>850002.52</td>
      <td>0.00</td>
      <td>6510099.11</td>
      <td>7360101.63</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>6362620 rows × 8 columns</p>
</div>




```python
y
```




    0          0
    1          0
    2          1
    3          1
    4          0
              ..
    6362615    1
    6362616    1
    6362617    1
    6362618    1
    6362619    1
    Name: isFraud, Length: 6362620, dtype: int64




```python
from imblearn.over_sampling import SMOTENC
sm = SMOTENC(categorical_features=[0,6,7], sampling_strategy='minority',random_state=1, k_neighbors=5)
X1, y1 = sm.fit_resample(X,y)
```


```python
X1
```




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
      <th>step</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type</th>
      <th>nameDest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.194174</td>
      <td>1.701360e+05</td>
      <td>160296.36</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.530630</td>
      <td>2.124900e+04</td>
      <td>19384.72</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.198497</td>
      <td>1.810000e+02</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5.198497</td>
      <td>1.810000e+02</td>
      <td>0.00</td>
      <td>21182.000000</td>
      <td>0.000000</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>9.364617</td>
      <td>4.155400e+04</td>
      <td>29885.86</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>PAYMENT</td>
      <td>M</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12708809</th>
      <td>4</td>
      <td>11.398498</td>
      <td>8.925055e+04</td>
      <td>0.00</td>
      <td>1073.629607</td>
      <td>90324.179841</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>12708810</th>
      <td>10</td>
      <td>14.458943</td>
      <td>1.903693e+06</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>12708811</th>
      <td>21</td>
      <td>11.964052</td>
      <td>1.570399e+05</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
    <tr>
      <th>12708812</th>
      <td>1</td>
      <td>10.026860</td>
      <td>2.276142e+04</td>
      <td>0.00</td>
      <td>7767.977465</td>
      <td>23870.414246</td>
      <td>CASH_OUT</td>
      <td>C</td>
    </tr>
    <tr>
      <th>12708813</th>
      <td>10</td>
      <td>15.692535</td>
      <td>6.561458e+06</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>TRANSFER</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
<p>12708814 rows × 8 columns</p>
</div>




```python
X1['step'] = X1['step'].astype(int)
```


```python
X1 = pd.get_dummies(X1, drop_first=True)
```


```python
X1
```




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
      <th>step</th>
      <th>amount</th>
      <th>oldbalanceOrg</th>
      <th>newbalanceOrig</th>
      <th>oldbalanceDest</th>
      <th>newbalanceDest</th>
      <th>type_CASH_OUT</th>
      <th>type_DEBIT</th>
      <th>type_PAYMENT</th>
      <th>type_TRANSFER</th>
      <th>nameDest_M</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.194174</td>
      <td>1.701360e+05</td>
      <td>160296.36</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>7.530630</td>
      <td>2.124900e+04</td>
      <td>19384.72</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>5.198497</td>
      <td>1.810000e+02</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5.198497</td>
      <td>1.810000e+02</td>
      <td>0.00</td>
      <td>21182.000000</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>9.364617</td>
      <td>4.155400e+04</td>
      <td>29885.86</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>12708809</th>
      <td>4</td>
      <td>11.398498</td>
      <td>8.925055e+04</td>
      <td>0.00</td>
      <td>1073.629607</td>
      <td>90324.179841</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12708810</th>
      <td>10</td>
      <td>14.458943</td>
      <td>1.903693e+06</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12708811</th>
      <td>21</td>
      <td>11.964052</td>
      <td>1.570399e+05</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12708812</th>
      <td>1</td>
      <td>10.026860</td>
      <td>2.276142e+04</td>
      <td>0.00</td>
      <td>7767.977465</td>
      <td>23870.414246</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12708813</th>
      <td>10</td>
      <td>15.692535</td>
      <td>6.561458e+06</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>12708814 rows × 11 columns</p>
</div>




```python
pd.concat()
```


```python

```


```python

```


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X1, y1, test_size = 0.20, random_state = 0)
```


```python
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test 
```


```python
sc = StandardScaler()
X_train[:,:6] = sc.fit_transform(X_train[:,:6])
X_test[:,:6] = sc.transform(X_test[:,:6])
```


```python
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF.fit(X_train, y_train)
```




    RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=0)




```python
RF.score(X_train, y_train)
```




    0.9999860333148717




```python
RF.score(X_test, y_test)
```




    0.9995593609632369




```python
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = RF.predict(X_test)
confusion_matrix(y_test,y_pred)
```




    array([[1268695,     910],
           [    210, 1271948]])




```python
check = pd.DataFrame(X_test[(y_test == 1) & (y_pred == 0)], columns=['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER','nameDest_M'])

```


```python
check['type_TRANSFER'].sum()
```




    7.0




```python
check['type_CASH_OUT'].sum()

```




    203.0




```python

```
