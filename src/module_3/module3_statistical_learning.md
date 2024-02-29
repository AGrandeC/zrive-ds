```python
import os
import numpy as np
import pandas as pd
from summarytools import dfSummary
import warnings 


import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


import pickle 
from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, roc_auc_score
import sklearn.metrics as metrics

pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
warnings.filterwarnings('ignore') 


data_path = os.getcwd() + "/data/"
if not os.path.exists(data_path):
    os.makedirs(data_path)
```

# Load data and keep orders with 5 items or more


```python
df_base = pd.read_parquet(data_path + 'feature_frame.parquet')
```


```python
df_base
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>vendor</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.276180</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2880544</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643254800516</td>
      <td>3893722808452</td>
      <td>2021-03-03 13:19:28</td>
      <td>2021-03-03</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>colief</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880545</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643274788996</td>
      <td>3883757174916</td>
      <td>2021-03-03 13:57:35</td>
      <td>2021-03-03</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>colief</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880546</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643283734660</td>
      <td>3874925314180</td>
      <td>2021-03-03 14:14:24</td>
      <td>2021-03-03</td>
      <td>7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>colief</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880547</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643294515332</td>
      <td>3906490826884</td>
      <td>2021-03-03 14:30:30</td>
      <td>2021-03-03</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>colief</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
    <tr>
      <th>2880548</th>
      <td>33826439594116</td>
      <td>healthcarevitamins</td>
      <td>3643301986436</td>
      <td>3914253959300</td>
      <td>2021-03-03 14:42:05</td>
      <td>2021-03-03</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.417186</td>
      <td>0.114360</td>
      <td>colief</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>34.0</td>
      <td>27.693045</td>
      <td>30.0</td>
      <td>34.0</td>
      <td>27.451392</td>
    </tr>
  </tbody>
</table>
<p>2880549 rows × 27 columns</p>
</div>




```python
items_ordered = df_base[df_base['outcome'] == 1].groupby('order_id')['variant_id'].count().reset_index().rename(columns={"variant_id": "order_size"})

df_orders = pd.merge(left=df_base, right=items_ordered, 
                    how='left', left_on=['order_id'], right_on=['order_id'])

df_orders = df_orders[df_orders['order_size'] >= 5]
df_orders.head()
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>discount_pct</th>
      <th>vendor</th>
      <th>global_popularity</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
      <th>order_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808434524292</td>
      <td>3479090790532</td>
      <td>2020-10-06 10:50:23</td>
      <td>2020-10-06</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.081052</td>
      <td>0.053512</td>
      <td>clearspring</td>
      <td>0.038462</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
dfSummary(df_orders)
```




<style type="text/css">
#T_7f732 thead>tr>th {
  text-align: left;
}
#T_7f732_row0_col0, #T_7f732_row1_col0, #T_7f732_row2_col0, #T_7f732_row3_col0, #T_7f732_row4_col0, #T_7f732_row5_col0, #T_7f732_row6_col0, #T_7f732_row7_col0, #T_7f732_row8_col0, #T_7f732_row9_col0, #T_7f732_row10_col0, #T_7f732_row11_col0, #T_7f732_row12_col0, #T_7f732_row13_col0, #T_7f732_row14_col0, #T_7f732_row15_col0, #T_7f732_row16_col0, #T_7f732_row17_col0, #T_7f732_row18_col0, #T_7f732_row19_col0, #T_7f732_row20_col0, #T_7f732_row21_col0, #T_7f732_row22_col0, #T_7f732_row23_col0, #T_7f732_row24_col0, #T_7f732_row25_col0, #T_7f732_row26_col0, #T_7f732_row27_col0 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 5%;
  max-width: 50px;
  min-width: 20px;
}
#T_7f732_row0_col1, #T_7f732_row1_col1, #T_7f732_row2_col1, #T_7f732_row3_col1, #T_7f732_row4_col1, #T_7f732_row5_col1, #T_7f732_row6_col1, #T_7f732_row7_col1, #T_7f732_row8_col1, #T_7f732_row9_col1, #T_7f732_row10_col1, #T_7f732_row11_col1, #T_7f732_row12_col1, #T_7f732_row13_col1, #T_7f732_row14_col1, #T_7f732_row15_col1, #T_7f732_row16_col1, #T_7f732_row17_col1, #T_7f732_row18_col1, #T_7f732_row19_col1, #T_7f732_row20_col1, #T_7f732_row21_col1, #T_7f732_row22_col1, #T_7f732_row23_col1, #T_7f732_row24_col1, #T_7f732_row25_col1, #T_7f732_row26_col1, #T_7f732_row27_col1 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 15%;
  max-width: 200px;
  min-width: 100px;
  word-break: break-word;
}
#T_7f732_row0_col2, #T_7f732_row1_col2, #T_7f732_row2_col2, #T_7f732_row3_col2, #T_7f732_row4_col2, #T_7f732_row5_col2, #T_7f732_row6_col2, #T_7f732_row7_col2, #T_7f732_row8_col2, #T_7f732_row9_col2, #T_7f732_row10_col2, #T_7f732_row11_col2, #T_7f732_row12_col2, #T_7f732_row13_col2, #T_7f732_row14_col2, #T_7f732_row15_col2, #T_7f732_row16_col2, #T_7f732_row17_col2, #T_7f732_row18_col2, #T_7f732_row19_col2, #T_7f732_row20_col2, #T_7f732_row21_col2, #T_7f732_row22_col2, #T_7f732_row23_col2, #T_7f732_row24_col2, #T_7f732_row25_col2, #T_7f732_row26_col2, #T_7f732_row27_col2 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 30%;
  min-width: 100px;
}
#T_7f732_row0_col3, #T_7f732_row1_col3, #T_7f732_row2_col3, #T_7f732_row3_col3, #T_7f732_row4_col3, #T_7f732_row5_col3, #T_7f732_row6_col3, #T_7f732_row7_col3, #T_7f732_row8_col3, #T_7f732_row9_col3, #T_7f732_row10_col3, #T_7f732_row11_col3, #T_7f732_row12_col3, #T_7f732_row13_col3, #T_7f732_row14_col3, #T_7f732_row15_col3, #T_7f732_row16_col3, #T_7f732_row17_col3, #T_7f732_row18_col3, #T_7f732_row19_col3, #T_7f732_row20_col3, #T_7f732_row21_col3, #T_7f732_row22_col3, #T_7f732_row23_col3, #T_7f732_row24_col3, #T_7f732_row25_col3, #T_7f732_row26_col3, #T_7f732_row27_col3 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 25%;
  min-width: 100px;
}
#T_7f732_row0_col4, #T_7f732_row1_col4, #T_7f732_row2_col4, #T_7f732_row3_col4, #T_7f732_row4_col4, #T_7f732_row5_col4, #T_7f732_row6_col4, #T_7f732_row7_col4, #T_7f732_row8_col4, #T_7f732_row9_col4, #T_7f732_row10_col4, #T_7f732_row11_col4, #T_7f732_row12_col4, #T_7f732_row13_col4, #T_7f732_row14_col4, #T_7f732_row15_col4, #T_7f732_row16_col4, #T_7f732_row17_col4, #T_7f732_row18_col4, #T_7f732_row19_col4, #T_7f732_row20_col4, #T_7f732_row21_col4, #T_7f732_row22_col4, #T_7f732_row23_col4, #T_7f732_row24_col4, #T_7f732_row25_col4, #T_7f732_row26_col4, #T_7f732_row27_col4 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 20%;
  min-width: 150px;
}
#T_7f732_row0_col5, #T_7f732_row1_col5, #T_7f732_row2_col5, #T_7f732_row3_col5, #T_7f732_row4_col5, #T_7f732_row5_col5, #T_7f732_row6_col5, #T_7f732_row7_col5, #T_7f732_row8_col5, #T_7f732_row9_col5, #T_7f732_row10_col5, #T_7f732_row11_col5, #T_7f732_row12_col5, #T_7f732_row13_col5, #T_7f732_row14_col5, #T_7f732_row15_col5, #T_7f732_row16_col5, #T_7f732_row17_col5, #T_7f732_row18_col5, #T_7f732_row19_col5, #T_7f732_row20_col5, #T_7f732_row21_col5, #T_7f732_row22_col5, #T_7f732_row23_col5, #T_7f732_row24_col5, #T_7f732_row25_col5, #T_7f732_row26_col5, #T_7f732_row27_col5 {
  text-align: left;
  font-size: 12px;
  vertical-align: middle;
  width: 10%;
}
</style>
<table id="T_7f732">
  <caption><strong>Data Frame Summary</strong><br>df_orders<br>Dimensions: 2,163,953 x 28<br>Duplicates: 0</caption>
  <thead>
    <tr>
      <th id="T_7f732_level0_col0" class="col_heading level0 col0" >No</th>
      <th id="T_7f732_level0_col1" class="col_heading level0 col1" >Variable</th>
      <th id="T_7f732_level0_col2" class="col_heading level0 col2" >Stats / Values</th>
      <th id="T_7f732_level0_col3" class="col_heading level0 col3" >Freqs / (% of Valid)</th>
      <th id="T_7f732_level0_col4" class="col_heading level0 col4" >Graph</th>
      <th id="T_7f732_level0_col5" class="col_heading level0 col5" >Missing</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_7f732_row0_col0" class="data row0 col0" >1</td>
      <td id="T_7f732_row0_col1" class="data row0 col1" ><strong>variant_id</strong><br>[int64]</td>
      <td id="T_7f732_row0_col2" class="data row0 col2" >Mean (sd) : 34010222998353.5 (277529437979.2)<br>min < med < max:<br>33615294398596.0 < 33973246886020.0 < 34543002157188.0<br>IQR (CV) : 481409925120.0 (122.5)</td>
      <td id="T_7f732_row0_col3" class="data row0 col3" >976 distinct values</td>
      <td id="T_7f732_row0_col4" class="data row0 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABn0lEQVR4nO3bMVLCQBiG4ehY0OikoKfXgit4BsZzcgZP4B3oLSiRBjs6DSSbyX47z9NnZouXWfbP5uFyuXRQu8elFwC3ECoRnpZewH82m8173/frMc8ej8fvw+HwWXhJLKTqUPu+X+92u58xz+73+1GBUydbPxGESgShEkGoRBAqEQZP/UZE1GAwVCMiamDrJ4JQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIlT9uTT3a/Wiu1Ab0+pFd1s/EYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEsG7fq5Op9Prdrv9GPv8nJdahMrVarV6GXuhpevmvdRi6yeCUIkgVCIIlQhCJYJQiSBUIpijzqDVL0GXJNQZtPol6JKaDbXm14Hcr9lQa34dyP2aDTXV1J3gfD6/dV33VXBJVRBqZQrsBM8l11MLoVLMlN1g6EwgVIqZshsMnQkM/IkgVCIIlQhCJcKshykzQUqZNdTkmeCUH5kfWHnGU3+YOGppcui+JP9RiSBUIgiVCEIlglCJIFQi/AIo0YbUp7tkZQAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row0_col5" class="data row0 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row1_col0" class="data row1 col0" >2</td>
      <td id="T_7f732_row1_col1" class="data row1 col1" ><strong>product_type</strong><br>[object]</td>
      <td id="T_7f732_row1_col2" class="data row1 col2" >1. tinspackagedfoods<br>2. condimentsdressings<br>3. ricepastapulses<br>4. haircare<br>5. longlifemilksubstitutes<br>6. cookingingredientsoils<br>7. dishwasherdetergent<br>8. cereal<br>9. bathroomlimescalecleaner<br>10. kidssnacks<br>11. other</td>
      <td id="T_7f732_row1_col3" class="data row1 col3" >169,541 (7.8%)<br>97,681 (4.5%)<br>96,554 (4.5%)<br>86,704 (4.0%)<br>83,162 (3.8%)<br>83,064 (3.8%)<br>74,033 (3.4%)<br>68,302 (3.2%)<br>62,420 (2.9%)<br>54,406 (2.5%)<br>1,288,086 (59.5%)</td>
      <td id="T_7f732_row1_col4" class="data row1 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAD+CAYAAAAtWHdlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAADPElEQVR4nO3cMWoCURhG0RiygClSu5aswYW6hmzF3uK1qUwbGBUD824gntMPvOLC3/i5u1wuL1B4/esH8DzERkZsZMRGRmxkxEZGbGTERkZsZMRGZrff7z+WZXl/9IMxxvl0On1OfBP/1NuyLO+Hw+Hr0Q+Ox+PDYcJPzigZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkXkbY5x/84PIMcZ55oP4v3b+xYiKM0pGbGTurqssqdjS3XWVJRVbckbJiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MjcHbwYt7AlgxcyzigZsZExeCFj8ELGGSUjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiI2PwQsbghYwzSkZsZMRG5ua6yrKKrd1cV1lWsTVnlIzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNzM11lWUVW7OuIuOMkhEbGYMXMgYvZJxRMmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyBi9kDF7IOKNkxEbG4IWMwQsZZ5SM2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyFhXkbGuIuOMkhEbmavrKssqZri6rrKsYgZnlIzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNzNXBi7ELMxi8kHFGyYiNjMELGYMXMs4oGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZgxcyBi9knFEyYiMjNjKrdZVlFbOs1lWWVczijJIRGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkVmtqyyrmMW6iowzSkZsZAxeyBi8kHFGyYiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIGLyQMXgh44ySERuZ1eCFuZ55ULQavDDXMw+KnFEyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjKrwQtzPfOgyOCFjDNKRmxkxEZGbGTERkZsZMRGRmxkxEZGbGS+AYp0FslymfnKAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row1_col5" class="data row1 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row2_col0" class="data row2 col0" >3</td>
      <td id="T_7f732_row2_col1" class="data row2 col1" ><strong>order_id</strong><br>[int64]</td>
      <td id="T_7f732_row2_col2" class="data row2 col2" >Mean (sd) : 2970614701897.8 (237764426887.6)<br>min < med < max:<br>2807985930372.0 < 2900908834948.0 < 3643294515332.0<br>IQR (CV) : 48388341760.0 (12.5)</td>
      <td id="T_7f732_row2_col3" class="data row2 col3" >2,603 distinct values</td>
      <td id="T_7f732_row2_col4" class="data row2 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABSklEQVR4nO3dsU3DUBRA0YBokVykzwAswQwZ1DNkAnZwn8ItaUKLkCCRnQguOqe1nvWLK/3iF+/hfD5v4K97/O0DwDWESsLTPX++2+1eh2HYLp2f5/k4TdPhhkci6q6hDsOw3e/370vnx3FcHDn/i6ufBKGSIFQShEqCUEkQKglCJUGoJAiVBKGScPEJdc17/el0etlsNm9LZuGzi6Guea8fx/F5yRx85eonQagkCJUEoZIgVBKESoJQSRAqCUIlQagkCJUEoZIgVBKESoJQSRAqCUIlQagkCJUEoZIgVBKESoJQSRAqCUIlQagkCJUEoZIgVBKESoJQSRAqCUIlQagkCJUEoZIgVBKESsLF9T1wrTU7yeZ5Pk7TdPjuu1C5mZU7yX4M3NVPglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCR8AMSNJUk32nbBAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row2_col5" class="data row2 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row3_col0" class="data row3 col0" >4</td>
      <td id="T_7f732_row3_col1" class="data row3 col1" ><strong>user_id</strong><br>[int64]</td>
      <td id="T_7f732_row3_col2" class="data row3 col2" >Mean (sd) : 3731337850014.8 (184517975869.9)<br>min < med < max:<br>3046041190532.0 < 3806872141956.0 < 5023380701316.0<br>IQR (CV) : 341070675968.0 (20.2)</td>
      <td id="T_7f732_row3_col3" class="data row3 col3" >1,517 distinct values</td>
      <td id="T_7f732_row3_col4" class="data row3 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABc0lEQVR4nO3csW3bQBiAUTpIocYBC/fqk0IrZAYhc3KGTJAd1KdQKatRSleGA5KC+Bnv9Qdc8QEHHvHf0+12G2Drvjx6A/A/hErC10dv4F72+/3PcRxf5q4/n89/T6fT7xW3xAKfNtRxHF+Ox+Pr3PXTNM2OnPU5+kkQKglCJUGoJAiVhE/71b/U5XL5fjgcfs1Z62prfUJ9x263+zb3esvV1voc/SQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJm57rX/LG6fV6/TEMw5+Vt8SDbDrUJW+cTtP0vPZ+eBxHPwlCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJeGuoyhLZp6GwdwTb+4a6pKZp2Ew98QbRz8JQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCR8+K/fY7pswYehekyXLfgHVcM6aTI3v48AAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row3_col5" class="data row3 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row4_col0" class="data row4 col0" >5</td>
      <td id="T_7f732_row4_col1" class="data row4 col1" ><strong>created_at</strong><br>[datetime64[us]]</td>
      <td id="T_7f732_row4_col2" class="data row4 col2" >Min: 2020-10-05<br>Max: 2021-03-03<br>Duration: 148 days</td>
      <td id="T_7f732_row4_col3" class="data row4 col3" >2603 distinct values</td>
      <td id="T_7f732_row4_col4" class="data row4 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABNCAYAAACxBha+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABqklEQVR4nO3dMU7DMACG0YAYuoAydO8OQ6/AGSrO2TNwAu7QnaFj6VIu0FTCJH9S+721iuThU+xYTvpwuVw6SHicewC0Q2zEiI2Yp7kHwDJtNpv3vu/Xf73ueDx+Hw6Hz2u/iY2r+r5f73a7n79et9/vBwM1jRIjNmLERozYiBEbMWIjRmzEiI0YsREjNmLERozYiBEbMWIjxhGjypWeSzufz29d132NORaxVe4f59Kexx6LaZQYsREjNmLERozYiBEbMbY+7sSS9stKie1OLGm/rJTYwmq4Q5USW1gNd6hSHhCIERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiIERsxTR8Ln+Kf5xjWdGxT/PMcw0yjxDR9Zyt1Op1et9vtR8m1NbySV0psBVar1UvJ9Nt1dbySV2pRsVmw121RsVmw121RsZUqXUO1vH6aQxWxla6hWl4/zcHWBzGT3Nla/lIPwyaJreUv9TDMNEqM2IgRGzFiI0ZsxNx8GrWFwZhuxmYLgzGZRokRGzFiI0ZsxIiNGLERIzZixEbML9wQbhhSHEVtAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row4_col5" class="data row4 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row5_col0" class="data row5 col0" >6</td>
      <td id="T_7f732_row5_col1" class="data row5 col1" ><strong>order_date</strong><br>[datetime64[us]]</td>
      <td id="T_7f732_row5_col2" class="data row5 col2" >Min: 2020-10-05<br>Max: 2021-03-03<br>Duration: 149 days</td>
      <td id="T_7f732_row5_col3" class="data row5 col3" >149 distinct values</td>
      <td id="T_7f732_row5_col4" class="data row5 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABNCAYAAACxBha+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABsklEQVR4nO3dPW6CYADHYdp0MCZtGNzd6+AVegbSc3qGnqB3cO/gaF3sBcSEt/AH4XlWQ/IOv/B+BPDper1WkPA89gBYDrERIzZiXsYeANO03W4/6rredL3udDr9HI/Hr1u/iY2b6rreNE3z2/W6w+HQGqhplBixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiIERsxHjGaudLn0i6Xy66qqu8+xyK2mfvHc2mvfY/FNEqM2IgRGzFiI0ZsxIiNGEcfD2JK52WlxBZWGs16vd41TdM5miHOy0qJLWxKh6xp1mzEiI0YsREjNmLERozYiBEbMWIjRmzEiI0YsREjNmLERozYiBEbMWIjRmzELPpJ3SH+n4l2i45tiP9nop1plBixESM2Yha9Zit1Pp/f9/v9Z8m1U3ppOG1SsT3K7nC1Wr2VbCyqah7vf5aaVGx2h/NmzUbMpO5spUrXUEteP41hFrGVrqGWvH4ag2mUmEHubHP4lhj9GyS2JX8WinamUWLERozYiBEbMWIj5u5u1BEGfbobmyMM+mQaJUZsxIiNGLERIzZixEaM2IgRGzF/Ub1qB8KX0GQAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row5_col5" class="data row5 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row6_col0" class="data row6 col0" >7</td>
      <td id="T_7f732_row6_col1" class="data row6 col1" ><strong>user_order_seq</strong><br>[int32]</td>
      <td id="T_7f732_row6_col2" class="data row6 col2" >Mean (sd) : 3.4 (2.3)<br>min < med < max:<br>2.0 < 3.0 < 21.0<br>IQR (CV) : 2.0 (1.5)</td>
      <td id="T_7f732_row6_col3" class="data row6 col3" >19 distinct values</td>
      <td id="T_7f732_row6_col4" class="data row6 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABcUlEQVR4nO3aMYrbQBiA0SRs4SrY4DLgPo2vkDOInFNn2KsIUi5Ypbdy2hACu0hr4k+8V2uGKT5mCv2fb7fbJ3h0X/73AeA9hErC01sfnE6nH/v9/rhk83meX6Zpel6yFv70Zqj7/f44DMPrks3HcVwUOPzN00+CUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGS8HTPza/X6/fz+fxz6fp5nl+maXr+wCMRdddQd7vd12EYXpeuH8fx+JHnocvTT4JQSRAqCUIlQagkCJUEoZIgVBKESoJQSbjrL9S11swKmBPYlocOdc2sgDmBbfH0k/DQN+oaRgy3ZbOhGjHcls2GutaaG/lyuXw7HA6/lqx1k//bb7BDTM1B8o86AAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row6_col5" class="data row6 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row7_col0" class="data row7 col0" >8</td>
      <td id="T_7f732_row7_col1" class="data row7 col1" ><strong>outcome</strong><br>[float64]</td>
      <td id="T_7f732_row7_col2" class="data row7 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row7_col3" class="data row7 col3" >2,132,624 (98.6%)<br>31,329 (1.4%)</td>
      <td id="T_7f732_row7_col4" class="data row7 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAA6ElEQVR4nO3bsW2EQBRF0QGttOkExNTiGiiUGtwKOcFPHeHEbAc8JHxOBT+40iTzhuM4GiSMdx/A/zHM8/zVe5/uPoTnqqp927bvV+99Wpbl5+6DeK51XafWPKMEiY0YsREjNmLERozYiBEbMWIjRmzEiI0YsREjNmLERozYiBEbMWIjRmzEvKpqP39SwhWqam+ttcG6ihTPKDFiI+Yz5TvnVncfxHON55TPdpSreUaJERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiIGf+mfO9zbgVXMeUj5hcsKDM0/SXDEAAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row7_col5" class="data row7 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row8_col0" class="data row8 col0" >9</td>
      <td id="T_7f732_row8_col1" class="data row8 col1" ><strong>ordered_before</strong><br>[float64]</td>
      <td id="T_7f732_row8_col2" class="data row8 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row8_col3" class="data row8 col3" >2,108,804 (97.5%)<br>55,149 (2.5%)</td>
      <td id="T_7f732_row8_col4" class="data row8 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAA50lEQVR4nO3bsQ2CUBiFUTEO8AprZnEGBmUGV6Gn+FsrrEwUWrwkek5H94ovvJBwu2VZTpBwPvoA/I+u7/tba+169EH4PVU1T9N0fz1fWmvXYRgeB56JHzWO48dLzDVKjNiIERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGzKWq5vUflbCHqprfnzvrKlJco8SIjZjNlG89v4K9bKZ8Phb4FtcoMWIjRmzEiI0YsREjNmLERozYiBEbMWIjRmzEiI0YsREjNmLERozYiBEbMZsp33p+BXsx5SPmCZy4NR/zqYk0AAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row8_col5" class="data row8 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row9_col0" class="data row9 col0" >10</td>
      <td id="T_7f732_row9_col1" class="data row9 col1" ><strong>abandoned_before</strong><br>[float64]</td>
      <td id="T_7f732_row9_col2" class="data row9 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row9_col3" class="data row9 col3" >2,162,348 (99.9%)<br>1,605 (0.1%)</td>
      <td id="T_7f732_row9_col4" class="data row9 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAAxUlEQVR4nO3bsY3DMBQFQVK4Ahg4O0C1XA3qP3URjnRdLAFrpoIXLPATct73PaBw7B7Ac8zzPP/WWq/dQ/h+P2ut13Vdn91D+H7OKBmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZz8LJTL+rqDijZMRG5phz/u4ewTMcYwyxkXBGyYiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MgcY4z37hE8g698ZP4BGScOXsQ7mJwAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row9_col5" class="data row9 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row10_col0" class="data row10 col0" >11</td>
      <td id="T_7f732_row10_col1" class="data row10 col1" ><strong>active_snoozed</strong><br>[float64]</td>
      <td id="T_7f732_row10_col2" class="data row10 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row10_col3" class="data row10 col3" >2,157,797 (99.7%)<br>6,156 (0.3%)</td>
      <td id="T_7f732_row10_col4" class="data row10 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAAxUlEQVR4nO3bsY3DMBQFQVK4Ahg4O0C1XA3qP3URjnRdLAFrpoIXLPATct73PaBw7B7Ac8zzPP/WWq/dQ/h+P2ut13Vdn91D+H7OKBmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZz8LJTL+rqDijZMRG5phz/u4ewTMcYwyxkXBGyYiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MgcY4z37hE8g698ZP4BGScOXsQ7mJwAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row10_col5" class="data row10 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row11_col0" class="data row11 col0" >12</td>
      <td id="T_7f732_row11_col1" class="data row11 col1" ><strong>set_as_regular</strong><br>[float64]</td>
      <td id="T_7f732_row11_col2" class="data row11 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row11_col3" class="data row11 col3" >2,154,680 (99.6%)<br>9,273 (0.4%)</td>
      <td id="T_7f732_row11_col4" class="data row11 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAA10lEQVR4nO3bsQ2DMBRFURMxgAtqz5IZGJQZsgo9xW9TkS2eJXLOBK+40hcSXu77bpDwmj2A/7GMMd699232EJ6tqq61977t+/6dPYZnO45jc0aJERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiI8Vs4EVV1LV5XkeKMEiM2YpYxxru11s7z/MydwtOtPg5IcUaJERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiIWavqmj2C/+ApHzE/4AQe1nALPmIAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row11_col5" class="data row11 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row12_col0" class="data row12 col0" >13</td>
      <td id="T_7f732_row12_col1" class="data row12 col1" ><strong>normalised_price</strong><br>[float64]</td>
      <td id="T_7f732_row12_col2" class="data row12 col2" >Mean (sd) : 0.1 (0.1)<br>min < med < max:<br>0.0 < 0.1 < 1.0<br>IQR (CV) : 0.1 (1.0)</td>
      <td id="T_7f732_row12_col3" class="data row12 col3" >127 distinct values</td>
      <td id="T_7f732_row12_col4" class="data row12 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABgElEQVR4nO3asWrCQBzHcVscnEoEx4J7l7xB6TOEPqNjnsFXCXQUzKiTXUtBLBel+R2fz5w7bvgmB3/ydLlcFjB3z/99APgLoRJheeuB7Xb70TTNpmTzcRwPwzDsS9bCTzdDbZpm03XduWTzvu+LAoffXP1EECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEWD5y89Pp9Na27Wfp+nEcD8Mw7O94JEI9NNTVavXSdd25dH3f95t7nodcrn4iCJUIQiWCUIkgVCIIlQgPHU9NNWUOawZbl1mHOmUOawZbF1c/EYRKhFlf/VP4z6Au1YbqP4O6VBvqVCYO8yLUK6Z8kXe73XvbtkVf5OPx+Lper79K1i4W9b4k323bWNQSwuo5AAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row12_col5" class="data row12 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row13_col0" class="data row13 col0" >14</td>
      <td id="T_7f732_row13_col1" class="data row13 col1" ><strong>discount_pct</strong><br>[float64]</td>
      <td id="T_7f732_row13_col2" class="data row13 col2" >Mean (sd) : 0.2 (0.2)<br>min < med < max:<br>-0.0 < 0.1 < 1.3<br>IQR (CV) : 0.1 (1.0)</td>
      <td id="T_7f732_row13_col3" class="data row13 col3" >527 distinct values</td>
      <td id="T_7f732_row13_col4" class="data row13 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABnElEQVR4nO3dMWoCQQCG0UlIYZOwgmXAPim8Qs4gOeeewRPkDgspBbc0NqYNgWDYcXH/5b1+dIoPZ8HZmbvz+Vxg6u5vPQH4D6ES4WHMD1+v129N06yGju/7ft913e6KUyLUqKE2TbPabrdfQ8e3bTs4cubF0k8EoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEuPhef80hEqfT6bWU8jFkLPx0MdSaQyTatn0cMg5+s/QTQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEmHUa9BrHY/Hl81m8z5kbN/3+67rdleeEjcy6VAXi8VTxavag84iYJos/UQQKhGESgShEkGoRBAqEYRKBKESQahEECoRJv0Xao2afQKl2CswNbMNtWafQCn2CkyNpZ8IQiXCbJf+Wql7YWuuW5ryc7lQ/5C6F7byuqXJPpcLdQQ1v8aHw+F5uVx+Dv3uuV5C9w0Dkl78oLpjcwAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row13_col5" class="data row13 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row14_col0" class="data row14 col0" >15</td>
      <td id="T_7f732_row14_col1" class="data row14 col1" ><strong>vendor</strong><br>[object]</td>
      <td id="T_7f732_row14_col2" class="data row14 col2" >1. biona<br>2. ecover<br>3. method<br>4. organix<br>5. treeoflife<br>6. huggies<br>7. bulldog<br>8. clearspring<br>9. cooksco<br>10. biod<br>11. other</td>
      <td id="T_7f732_row14_col3" class="data row14 col3" >110,820 (5.1%)<br>85,323 (3.9%)<br>59,869 (2.8%)<br>56,312 (2.6%)<br>52,060 (2.4%)<br>33,814 (1.6%)<br>31,890 (1.5%)<br>31,236 (1.4%)<br>29,625 (1.4%)<br>28,633 (1.3%)<br>1,644,371 (76.0%)</td>
      <td id="T_7f732_row14_col4" class="data row14 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAD+CAYAAAAtWHdlAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAADRklEQVR4nO3cMW7iUBhG0YCQpnWRmrVkDSzUa8hW6Clem4ppkigS8khR7AvKnFNTvOKKv+Fjd71en6Cwv/cD+H+IjYzYyIiNjNjIiI2M2MiIjYzYyIiNzO54PL5M0/S89IExxuV8Pr+Gb+KXOkzT9Hw6nd6WPjDP82KI8B3OKBmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGROYwxLv/6geQY41I+iN9r51+MqDijZMRGZnFdZVXF2hbXVVZVrM0ZJSM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjszh4MXRhbQYvZJxRMmIjczN4MXRhKzeDF0MXtuKMkhEbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRuRm8GLqwFYMXMs4oGbGRERsZ6yoy1lVknFEyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjLWVWSsq8g4o2TERsbghYzBCxlnlIzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjMELGYMXMs4oGbGR+Ry8GLqwtf3H4OXrwgq24IySERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERG5n9+7rqj1UVW7OuIuOMkhEbGesqMtZVZJxRMmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyBi9kDF7IOKNkxEbG4IWMwQsZZ5SM2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzBCxmDFzLOKBmxkREbGesqMtZVZJxRMmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYy1lVkrKvIOKNkxEbG4IWMwQsZZ5SM2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzBCxmDFzLOKBmxkfkcvNz7IazrEQdMh4/By70fwrrmeX64LxBnlIzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNzOF98PJwv33iZx5xwGTwQsYZJSM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyfwFoCg0yhWHu9wAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row14_col5" class="data row14 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row15_col0" class="data row15 col0" >16</td>
      <td id="T_7f732_row15_col1" class="data row15 col1" ><strong>global_popularity</strong><br>[float64]</td>
      <td id="T_7f732_row15_col2" class="data row15 col2" >Mean (sd) : 0.0 (0.0)<br>min < med < max:<br>0.0 < 0.0 < 0.4<br>IQR (CV) : 0.0 (0.6)</td>
      <td id="T_7f732_row15_col3" class="data row15 col3" >5,968 distinct values</td>
      <td id="T_7f732_row15_col4" class="data row15 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABO0lEQVR4nO3VIW4CURRA0bZBoJoReHwNW+gaJl3nrIGtjEeMBEVtU0MzgZSbnKP/e3niJv/1er2+wLN7++8D4C+ESsLm1oP9fv85DMNuzfJlWU7zPB/XzMJPN0MdhmE3juNlzfJpmlYFDr/5+kkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQSNo9cfj6fPw6Hw9fa+WVZTvM8H+94ElEPDXW73b6P43hZOz9N0+6e99D1DVgmHychGFosAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row15_col5" class="data row15 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row16_col0" class="data row16 col0" >17</td>
      <td id="T_7f732_row16_col1" class="data row16 col1" ><strong>count_adults</strong><br>[float64]</td>
      <td id="T_7f732_row16_col2" class="data row16 col2" >1. 2.0<br>2. 3.0<br>3. 4.0<br>4. 1.0<br>5. 5.0</td>
      <td id="T_7f732_row16_col3" class="data row16 col3" >2,112,747 (97.6%)<br>22,616 (1.0%)<br>13,481 (0.6%)<br>11,042 (0.5%)<br>4,067 (0.2%)</td>
      <td id="T_7f732_row16_col4" class="data row16 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABzCAYAAACPdnBjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABvUlEQVR4nO3bIY7bUBhG0WcrUqnBsEpey6whC80aupWAsoBHi1LSqdrgzLU0OYeZPXCVn+Rb7vf7gMJ69AN4HWIjIzYyYiOz7Pv+vm3b29EP4euZc96u1+uPj+/Ttm1v5/P514Fv4ou6XC7//Yg5o2TERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZE5zztvj/47gGeact3+/F1M+Ks4oGbGRERuZv+uqxyUMPNv6sa4y5+OzOaNkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTWP+uqb49LGHg26yoyzigZsZERG5ll3/f3McawrOKznayqqDijZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkTlZVVKyryDijZMRGxuCFjMELGWeUjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MhYV5GxriLjjJIRG5l1WZbvRz+C17COMcRGwhklIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjLrGOPn0Y/gNVhXkXFGyYiNjNjI/AapCE3QQMnosAAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row16_col5" class="data row16 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row17_col0" class="data row17 col0" >18</td>
      <td id="T_7f732_row17_col1" class="data row17 col1" ><strong>count_children</strong><br>[float64]</td>
      <td id="T_7f732_row17_col2" class="data row17 col2" >1. 0.0<br>2. 2.0<br>3. 1.0<br>4. 3.0</td>
      <td id="T_7f732_row17_col3" class="data row17 col3" >2,080,173 (96.1%)<br>44,284 (2.0%)<br>29,140 (1.3%)<br>10,356 (0.5%)</td>
      <td id="T_7f732_row17_col4" class="data row17 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABcCAYAAAB5jMeAAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABfklEQVR4nO3dMWoCYRhF0VGEtFOknrW4BhfqGrIVe4u/TWWaJGDQJjhXSM4pp5ri4ofgw83lcpmgsH32C/B/iI2M2MhslmXZz/P8+uwX4e8YY5xPp9Pbz+e7eZ5fD4fD+xPeiT/qeDze/PByRsmIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyOzGGOd7vz+C3xhjnG8931hXUXFGyYiNjNjIXK2r7q1i4BGu1lW+lbImZ5SM2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyFxN+e5NsOARTPnIOKNkxEbme11lWcXatl/rKn+WxtqcUTJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjs/1cV71YVrE26yoyzigZsZHZLMuyn6ZpsqxibTurKirOKBmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGR2VlVUbGuIuOMkhEbmQ/sOFdv9SQccQAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row17_col5" class="data row17 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row18_col0" class="data row18 col0" >19</td>
      <td id="T_7f732_row18_col1" class="data row18 col1" ><strong>count_babies</strong><br>[float64]</td>
      <td id="T_7f732_row18_col2" class="data row18 col2" >1. 0.0<br>2. 1.0</td>
      <td id="T_7f732_row18_col3" class="data row18 col3" >2,155,136 (99.6%)<br>8,817 (0.4%)</td>
      <td id="T_7f732_row18_col4" class="data row18 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAAAuCAYAAAA/ZmtKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAA10lEQVR4nO3bsQ2DMBRFURMxgAtqz5IZGJQZsgo9xW9TkS2eJXLOBK+40hcSXu77bpDwmj2A/7GMMd699232EJ6tqq61977t+/6dPYZnO45jc0aJERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiI8Vs4EVV1LV5XkeKMEiM2YpYxxru11s7z/MydwtOtPg5IcUaJERsxYiNGbMSIjRixESM2YsRGjNiIERsxYiNGbMSIjRixESM2YsRGjNiIWavqmj2C/+ApHzE/4AQe1nALPmIAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row18_col5" class="data row18 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row19_col0" class="data row19 col0" >20</td>
      <td id="T_7f732_row19_col1" class="data row19 col1" ><strong>count_pets</strong><br>[float64]</td>
      <td id="T_7f732_row19_col2" class="data row19 col2" >1. 0.0<br>2. 1.0<br>3. 2.0<br>4. 3.0<br>5. 6.0</td>
      <td id="T_7f732_row19_col3" class="data row19 col3" >2,062,084 (95.3%)<br>70,404 (3.3%)<br>27,404 (1.3%)<br>2,362 (0.1%)<br>1,699 (0.1%)</td>
      <td id="T_7f732_row19_col4" class="data row19 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABzCAYAAACPdnBjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABwUlEQVR4nO3dMUoDURhG0ZlBsJ3CTshaXEMWmjW4lRR2KV5rFRu1SgIGcgPmnDbNKy75CfjhfDweJygs934Aj0NsZMRGRmxk5s1m87au68u9H8L/MMY47Pf791OfPa3r+rLdbj/jN/FP7Xa7s19czigZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZERG5mnMcbh0t8gwV+MMQ7nPptN+ag4o2TERkZsZE6uqy4tZOBaJ9dVfp1yC84oGbGRERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERuZk+uqSwsZuJZ1FRlnlIzYyIiNzO+6yqKKW1t+1lX+WRq35oySERsZsZERGxmxkREbGbGRERsZsZERGxmxkREbGbGRERsZsZFZvtdVzxZV3Jp1FRlnlIzYyCzzPL/e+xE8hmWaJrGRcEbJiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjcwyTdPHvR/BY7CuIuOMkhEbGesqMtZVZJxRMmIjIzYyYiMjNjJiIyM2MmIjIzYyYiMjNjJiIyM2MmIjIzYyYiNjXUXGuoqMM0pGbGTERuYLCMFU1alPf9kAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row19_col5" class="data row19 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row20_col0" class="data row20 col0" >21</td>
      <td id="T_7f732_row20_col1" class="data row20 col1" ><strong>people_ex_baby</strong><br>[float64]</td>
      <td id="T_7f732_row20_col2" class="data row20 col2" >1. 2.0<br>2. 4.0<br>3. 3.0<br>4. 5.0<br>5. 1.0</td>
      <td id="T_7f732_row20_col3" class="data row20 col3" >2,054,227 (94.9%)<br>49,237 (2.3%)<br>34,862 (1.6%)<br>22,951 (1.1%)<br>2,676 (0.1%)</td>
      <td id="T_7f732_row20_col4" class="data row20 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAJsAAABzCAYAAACPdnBjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABwUlEQVR4nO3dMUoDURhG0ZlBsJ3CTshaXEMWmjW4lRR2KV5rFRu1CRoRcoPmnDbNKy75CfjhfDweJygs134At0NsZMRGRmxk5s1m87Su68O1H8LfN8Y47Pf7568+v1vX9WG73b6Gb+Kf2u12335pOaNkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTuxhiHc3+HBD8xxjh89/lsykfFGSUjNjJiI3Oyrjq3kIHfOllX+WXKpTijZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkTtZV5xYy8FvWVWScUTJiIyM2Mp/rKqsqLm35WFf5Z2lcmjNKRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZmeV9X3VtVcWnWVWScUTJiI2PwQsbghYwzSkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkxEZGbGTERkZsZMRGRmxkrKvIWFeRcUbJiI3MMs/z47UfwW1YpmkSGwlnlIzYyIiNjNjIiI2M2MiIjYzYyIiNjNjIiI2M2MiIjYzYyIiNjNjILNM0vVz7EdwG6yoyzigZsZERG5k3txtns4LdVMwAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row20_col5" class="data row20 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row21_col0" class="data row21 col0" >22</td>
      <td id="T_7f732_row21_col1" class="data row21 col1" ><strong>days_since_purchase_variant_id</strong><br>[float64]</td>
      <td id="T_7f732_row21_col2" class="data row21 col2" >Mean (sd) : 33.2 (4.1)<br>min < med < max:<br>0.0 < 33.0 < 148.0<br>IQR (CV) : 0.0 (8.1)</td>
      <td id="T_7f732_row21_col3" class="data row21 col3" >142 distinct values</td>
      <td id="T_7f732_row21_col4" class="data row21 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABEUlEQVR4nO3VsWnDYBRG0Th4gL9wr1UygwbVDF5FfQq1qZQNjBEG6cI5/YOvuPBu+75/wdV9nz0A3iFUEu5nD3hlmqafMcbjyO22bb/ruj4/PImTXDrUMcZjnue/I7fLshwKnGvy+kkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEkQKglCJUGoJAiVBKGSIFQShEqCUEn4B0V1D97bCmYHAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row21_col5" class="data row21 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row22_col0" class="data row22 col0" >23</td>
      <td id="T_7f732_row22_col1" class="data row22 col1" ><strong>avg_days_to_buy_variant_id</strong><br>[float64]</td>
      <td id="T_7f732_row22_col2" class="data row22 col2" >Mean (sd) : 35.3 (10.6)<br>min < med < max:<br>0.0 < 34.0 < 84.0<br>IQR (CV) : 10.0 (3.3)</td>
      <td id="T_7f732_row22_col3" class="data row22 col3" >122 distinct values</td>
      <td id="T_7f732_row22_col4" class="data row22 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABtElEQVR4nO3dPWobQQCG4XFwoSZhFlQKto8LXcFnWHJOncEnyA1SCFwaVqWsRjlABA6zFrvf+nn6QQN6tT+agXm4Xq8Flu7b3BOA/yFUIjzOPYF76fv+uda6bR0/juOu67rXlrGn0+nteDy+tH42/1ptqLXW7TAM763jD4fDbhiGP41jm38g3ObWTwShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRVnto75zO5/PP/X7/q2Wsk6lvE+odbDabH60nWzuZ+rZFh9r3/XOttemLu1wuT6WU3588JWay6FBrrdsJV6bvnz0f5uNlighCJYJQiSBUIgiVCIt+6/+KpiwWlLLeBQOhLsyUxYJS1rtg4NZPBKESQahEECoRhEoEoRLhrn9PTdmmV4qtei3Wumn7rqFO2aZXiq16Lda6adutnwgfXlHtsv86pi7fjuO467rutWXsR48dfwHEQmu2yXkWkgAAAABJRU5ErkJggg=="></img></td>
      <td id="T_7f732_row22_col5" class="data row22 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row23_col0" class="data row23 col0" >24</td>
      <td id="T_7f732_row23_col1" class="data row23 col1" ><strong>std_days_to_buy_variant_id</strong><br>[float64]</td>
      <td id="T_7f732_row23_col2" class="data row23 col2" >Mean (sd) : 26.5 (7.1)<br>min < med < max:<br>1.4 < 27.7 < 58.7<br>IQR (CV) : 7.4 (3.7)</td>
      <td id="T_7f732_row23_col3" class="data row23 col3" >819 distinct values</td>
      <td id="T_7f732_row23_col4" class="data row23 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABkklEQVR4nO3dsU3DQBiG4R9EkQbkIn16KLICM0TM6RmYgB3SU6QMacICRIDPVvw5z9NbuuJVLvIv392dz+eCubu/9gLgL4RKhIdrL2CuNpvNa9d16yHPHg6Hz/1+/z7ykm6aUC/oum692+2+hjzb9/2gwLnM1k8EoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqERw9tQEjsfj83a7fRvyrAPWfibUCaxWqycHrI3L1k8EoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEWOysv+Xmvaqq0+n0UlUfIy6JBosNteXmvaqqvu8fx1wPbWz9RBAqEYRKBKESQahEECoRhEoEoRJBqESY9WSqZQxqBLossw61ZQxqBLostn4iCJUIQiWCUIkgVCIIlQiTvp7yOQhjmTRUn4P8X8shwFXLPQh41i/8b1HLIcBVyz0I2H9UIvz6i2renmWp9wf8Gqp5e5al3h/wDRs7Ygt/y2q9AAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row23_col5" class="data row23 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row24_col0" class="data row24 col0" >25</td>
      <td id="T_7f732_row24_col1" class="data row24 col1" ><strong>days_since_purchase_product_type</strong><br>[float64]</td>
      <td id="T_7f732_row24_col2" class="data row24 col2" >Mean (sd) : 31.7 (13.3)<br>min < med < max:<br>0.0 < 30.0 < 148.0<br>IQR (CV) : 0.0 (2.4)</td>
      <td id="T_7f732_row24_col3" class="data row24 col3" >141 distinct values</td>
      <td id="T_7f732_row24_col4" class="data row24 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABiUlEQVR4nO3WsW3CUBSGUSeioEn0kCgjuU8KbxAxg5UZKT0DE2QHSymRcEloyAYhMibwo3P6a93i83vv4Xg8VnDrHq+9APyFUIkwu/YCv6nrelVKWY6ZHYZh2/f9ZuKVuJKbDrWUsmzb9nvMbNd1owLnNrn6iSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQizS368rutVKWU5dv5wOLxVVfU54UqEumiopZRl27bfY+e7rnuach9ynQz1nFPRichUToZ6zqnoRGQqF736r2m/3782TfMxdn4Yhm3f95sJV+IMdxvqfD5/Pud9vF6v35umGfXk2e12L4vF4uu/Z6vqfn+wH+aqQSXEdlQmAAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row24_col5" class="data row24 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row25_col0" class="data row25 col0" >26</td>
      <td id="T_7f732_row25_col1" class="data row25 col1" ><strong>avg_days_to_buy_product_type</strong><br>[float64]</td>
      <td id="T_7f732_row25_col2" class="data row25 col2" >Mean (sd) : 30.9 (4.3)<br>min < med < max:<br>7.0 < 31.0 < 39.5<br>IQR (CV) : 6.0 (7.2)</td>
      <td id="T_7f732_row25_col3" class="data row25 col3" >26 distinct values</td>
      <td id="T_7f732_row25_col4" class="data row25 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABlElEQVR4nO3dMUoDURSG0atYpFEmYCmk1yJbcA3Bdc4aXIF7CFgKSRnTxC5tZN6Mzh/O6R+84oMbSN7Nzel0Kpi72/++APyGUIlw998XYFyr1eq167rHIWf3+/3Xdrt9H/lKoxDqlem67nGz2XwPOdv3/aDA/4LRTwShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEfxwmrPD4fC8Xq/fhp6f8oWAUDlbLBYPQ18HVE37QsDoJ4JQiSBUIviMOjMtz52rqo7H40tVfYx4pVkQ6sy0PHeuqur7/n7M+8yF0U8EoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRLKBgNC1rKy+trBQqo2lZW3lpZaXRTwShEsHon0DLRr5r3cbXSqgTaNnId63b+FoZ/UQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEqEi9/1t/zAYrfbPS2Xy88hZ6um/f93svwAcxxMnhecmeUAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row25_col5" class="data row25 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row26_col0" class="data row26 col0" >27</td>
      <td id="T_7f732_row26_col1" class="data row26 col1" ><strong>std_days_to_buy_product_type</strong><br>[float64]</td>
      <td id="T_7f732_row26_col2" class="data row26 col2" >Mean (sd) : 26.0 (3.3)<br>min < med < max:<br>2.8 < 26.1 < 35.6<br>IQR (CV) : 3.7 (8.0)</td>
      <td id="T_7f732_row26_col3" class="data row26 col3" >61 distinct values</td>
      <td id="T_7f732_row26_col4" class="data row26 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABjElEQVR4nO3cMU7DQBRF0QFRpAG5SO8eimyBNUSs02vICtiD+xQuQ5rQpUMBj038rHN6S1/KlSYa2f/hcrkUWLrHew8AvyFUIjzdewCm1bbte9M02zHPDsNw7Pv+MPFIkxDqyjRNs93v919jnu26blTg/8HRTwShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEkGoRBAqEYRKBKESQahEECoRhEoEoRJBqEQQKhGESgShEsFKn4Wp2R1VSinn8/mtlPI54UiLINSFqdkdVUopXdc9TznPUjj6iSBUIgiVCEIlglCJIFQiCJUIQiWCUIkgVCIIlQhCJYJQiSBUIgiVCEIlglCJIFQiCJUIQiWCUIngK1SuTqfT6263+xj7/DAMx77vDxOOdCVUrjabzUvlp9qj9xHc4ugnglCJIFQi+I86g5r9UWvdHVVLqDOo2R+11t1RtRz9RBAqEYRKBKESQahEECoRhEoE96g/cGm/LDdDrfnB5nzta24u7f+u5jXBW618A8svTXuVecH8AAAAAElFTkSuQmCC"></img></td>
      <td id="T_7f732_row26_col5" class="data row26 col5" >0<br>(0.0%)</td>
    </tr>
    <tr>
      <td id="T_7f732_row27_col0" class="data row27 col0" >28</td>
      <td id="T_7f732_row27_col1" class="data row27 col1" ><strong>order_size</strong><br>[float64]</td>
      <td id="T_7f732_row27_col2" class="data row27 col2" >Mean (sd) : 12.0 (6.1)<br>min < med < max:<br>5.0 < 11.0 < 78.0<br>IQR (CV) : 7.0 (2.0)</td>
      <td id="T_7f732_row27_col3" class="data row27 col3" >42 distinct values</td>
      <td id="T_7f732_row27_col4" class="data row27 col4" ><img src = "data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAKoAAABGCAYAAABc8A97AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8g+/7EAAAACXBIWXMAAA9hAAAPYQGoP6dpAAABYUlEQVR4nO3doU7DUBiA0UEQU6Rifh7TNyA8w8IzTu4ZeJV5xORQQxEIZqSFsG85R7c3FV9yW/H33pxOpwVcutv/fgD4CaGScHfugvV6/TQMw2rK4ofD4XW/379MuRe+OhvqMAyrzWbzNmXx3W43KXD4ztZPglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiXh7GETcxyPx4dxHJ+n3u9UFT78aajL5fJ+6okqi4VTVfhk6ydBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkiBUEoRKglBJECoJQiVBqCQIlQShkvCnw31zzZliNcF6XS461DlTrCZYr4utnwShkiBUEi76HXUOvxO6Llcb6tzfCW2328dxHCd9kIn8970DYCVFi+c6a9sAAAAASUVORK5CYII="></img></td>
      <td id="T_7f732_row27_col5" class="data row27 col5" >0<br>(0.0%)</td>
    </tr>
  </tbody>
</table>




# Milestone 1


```python
info_cols = ['variant_id', 'order_id', 'user_id', 'created_at', 'order_date']
label_col = ['outcome']
features_cols = [col for col in df_orders.columns if col not in info_cols + [label_col]]

categorical_cols = ['product_type', 'vendor']
binary_cols = ['ordered_before', 'abandoned_before', 'active_snoozed', 'set_as_regular']
numerical_cols = [col for col in features_cols if col not in categorical_cols + binary_cols]
```


```python
# extra code – the next 5 lines define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

df_orders.hist(bins=50, figsize=(40,25))
```




    array([[<Axes: title={'center': 'variant_id'}>,
            <Axes: title={'center': 'order_id'}>,
            <Axes: title={'center': 'user_id'}>,
            <Axes: title={'center': 'created_at'}>,
            <Axes: title={'center': 'order_date'}>],
           [<Axes: title={'center': 'user_order_seq'}>,
            <Axes: title={'center': 'outcome'}>,
            <Axes: title={'center': 'ordered_before'}>,
            <Axes: title={'center': 'abandoned_before'}>,
            <Axes: title={'center': 'active_snoozed'}>],
           [<Axes: title={'center': 'set_as_regular'}>,
            <Axes: title={'center': 'normalised_price'}>,
            <Axes: title={'center': 'discount_pct'}>,
            <Axes: title={'center': 'global_popularity'}>,
            <Axes: title={'center': 'count_adults'}>],
           [<Axes: title={'center': 'count_children'}>,
            <Axes: title={'center': 'count_babies'}>,
            <Axes: title={'center': 'count_pets'}>,
            <Axes: title={'center': 'people_ex_baby'}>,
            <Axes: title={'center': 'days_since_purchase_variant_id'}>],
           [<Axes: title={'center': 'avg_days_to_buy_variant_id'}>,
            <Axes: title={'center': 'std_days_to_buy_variant_id'}>,
            <Axes: title={'center': 'days_since_purchase_product_type'}>,
            <Axes: title={'center': 'avg_days_to_buy_product_type'}>,
            <Axes: title={'center': 'std_days_to_buy_product_type'}>],
           [<Axes: title={'center': 'order_size'}>, <Axes: >, <Axes: >,
            <Axes: >, <Axes: >]], dtype=object)




    
![png](module3_statistical_learning_files/module3_statistical_learning_8_1.png)
    



```python
# Correlation matrix

columns = numerical_cols + label_col

def plot_correlation_matrix(df: pd.DataFrame, columns: list):
    # Compute the correlation matrix
    corr = df[columns].corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots (figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette (230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,# annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
plot_correlation_matrix(df_orders, columns)
```


    
![png](module3_statistical_learning_files/module3_statistical_learning_9_0.png)
    


## Create train and test sets

Puntos a tener en cuenta
- Hacer el split por order_id - No puede haber un order_id en train y test
- Debemos tener en cuenta las fechas -  Ventas futuras de un mismo usuario se verán influenciadas por ventas pasadas (producto previamente en abandoned_cart o si un producto se ha comprado recientemente)


```python
# Before splitting the data we'll handle datetimes

### Extract components of datetimes
df_orders = df_orders.drop('created_at', axis=1) # I'll only keep order_date as they display almost the same info

df_orders['order_year'] = df_orders['order_date'].dt.year
df_orders['order_month'] = df_orders['order_date'].dt.month
df_orders['order_day_of_week'] = df_orders['order_date'].dt.dayofweek 

df_orders = df_orders.drop('order_date', axis=1)
```


```python
def split_by_order_id(df, test_size=0.1, validation_size=0.1, random_state=42):
    """
    Splits a DataFrame into train, validation, and test sets based on unique order IDs.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float, optional): Proportion of the dataset to include in the test set.
                                     Defaults to 0.2.
        validation_size (float, optional): Proportion of the dataset to include in the validation set.
                                           Defaults to 0.2.
        random_state (int, optional): Controls the randomness of the split. 
                                      Set for reproducible results. Defaults to None.

    Returns:
        tuple: A tuple containing three DataFrames: (train_df, validation_df, test_df)
    """

    # Get unique order IDs
    unique_order_ids = df['order_id'].unique()

    # Split order IDs into train and test groups 
    train_order_ids, test_order_ids = train_test_split(unique_order_ids, 
                                                       test_size=test_size, 
                                                       random_state=random_state)

    # Optionally, split train order IDs further into train and validation groups
    if validation_size > 0:
        train_order_ids, validation_order_ids = train_test_split(train_order_ids,
                                                                 test_size=validation_size / (1 - test_size),
                                                                 random_state=random_state)

    # Create split masks
    train_mask = df['order_id'].isin(train_order_ids)
    validation_mask = df['order_id'].isin(validation_order_ids)
    test_mask = df['order_id'].isin(test_order_ids)

    # Split the DataFrame based on masks
    train_df = df[train_mask]
    validation_df = df[validation_mask]
    test_df = df[test_mask]

    return train_df, validation_df, test_df

train_df_0, validation_df_0, test_df_0 = split_by_order_id(df_orders)
```

### Create a baseline model


```python
train_df = train_df_0.copy()
validation_df = validation_df_0.copy()

### Label Encoder
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

# Save the encoding to reuse it in validation and test
with open(data_path + 'label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# ------------------ Model Training ------------------
    
X_train = train_df.drop(label_col, axis=1)
y_train = train_df[label_col]

# Fit the LogisticRegression model
log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train) 

# ------------------ Classification ------------------

# Encode test data
with open(data_path + 'label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

for col in categorical_cols: 
    validation_df[col] = encoders[col].transform(validation_df[col])

X_val = validation_df.drop(label_col, axis=1)
y_val = validation_df[label_col]

# Make predictions
y_pred = log_reg.predict(X_val) 

```


```python
def model_performance_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    precision = precision_score(y_test, y_pred, labels=[1, 0])
    recall = recall_score(y_test, y_pred, labels=[1, 0])
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n {conf_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'ROC AUC: {roc_auc}')

model_performance_metrics(y_val, y_pred)
```

    Accuracy: 0.9854730694473871
    Confusion Matrix:
     [[     0   3154]
     [     0 213960]]
    Precision: 0.0
    Recall: 0.0
    F1-Score: 0.0
    ROC AUC: 0.5


Como se podía esperar, el modelo predice sólo 0s ya que las clases están totalmente desbalanceadas. El siguiente paso será aplicar un undersampling para igualar las clases

## Undersampling

En este paso voy a aplicar un undersampling para tener las clases balanceadas al 50%


```python
train_df = train_df_0.copy()
validation_df = validation_df_0.copy()

### Label Encoder
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

# Save the encoding to reuse it in validation and test
with open(data_path + 'label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# ------------------ Model Training ------------------
    
X_train = train_df.drop(label_col, axis=1)
y_train = train_df[label_col]

# Undersampling
rus = RandomUnderSampler(random_state=42)  # Set random_state for reproducibility
X_resampled, y_resampled = rus.fit_resample(X_train, y_train) 

# Fit the LogisticRegression model
log_reg = LogisticRegression() 
log_reg.fit(X_resampled, y_resampled) 

# ------------------ Classification ------------------

# Encode test data
with open(data_path + 'label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

for col in categorical_cols: 
    validation_df[col] = encoders[col].transform(validation_df[col])

X_val = validation_df.drop(label_col, axis=1)
y_val = validation_df[label_col]

# Make predictions
y_pred = log_reg.predict(X_val) 
```


```python
model_performance_metrics(y_val, y_pred)
```

    Accuracy: 0.6231703160551599
    Confusion Matrix:
     [[  1334   1820]
     [ 79995 133965]]
    Precision: 0.01640251324865669
    Recall: 0.4229549778059607
    F1-Score: 0.0315803179337855
    ROC AUC: 0.524538341398774


Aplicar undersampling ha permitido que el modelo empiece a distinguir 1s, sin embargo seguimos teniendo muchos FPs ya que nuestro modelo está estimando casi al 50% la probabilidad de 1 o 0, como se puede ver en el ROC AUC.

He probado otras técnicas de undersampling aparte de random y no han dado mucho mejor resultado así que lo siguiente que probaré será cambiar el tipo de encoding

## Frequency encoding


```python
def apply_frequency_encoding(df, columns, outcome_condition='outcome'):
    """
    Applies frequency encoding to specified columns in a DataFrame. 
    Frequency is calculated only when a particular condition is met.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of columns to apply frequency encoding to.
        outcome_condition (str, optional): Column name containing the condition for 
                                           calculating frequency. Defaults to 'outcome'.

    Returns:
        pd.DataFrame: The DataFrame with frequency-encoded columns.
    """

    for col in columns:
        # Filter for rows where the outcome condition is True
        temp_df = df[df[outcome_condition] == 1] 

        # Calculate value frequencies
        frequencies = temp_df[col].value_counts(normalize=True) 

        # Create a mapping dictionary
        freq_map = frequencies.to_dict()

        # Apply frequency encoding to the entire column
        df[f'{col}_freq_enc'] = df[col].map(freq_map)

    return df

apply_frequency_encoding(train_df, categorical_cols, outcome_condition='outcome')

```


```python
train_df = train_df_0.copy()
validation_df = validation_df_0.copy()

### Frequency Encoder
encoders = {}
for col in categorical_cols:
    le = ce.CountEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    encoders[col] = le

# Save the encoding to reuse it in validation and test
with open(data_path + 'label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# ------------------ Model Training ------------------
    
X_train = train_df.drop(label_col, axis=1)
y_train = train_df[label_col]

# Undersampling
rus = RandomUnderSampler(random_state=42)  # Set random_state for reproducibility
X_resampled, y_resampled = rus.fit_resample(X_train, y_train) 

# Fit the LogisticRegression model
log_reg = LogisticRegression() 
log_reg.fit(X_resampled, y_resampled) 

# ------------------ Classification ------------------

# Encode test data
with open(data_path + 'label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

for col in categorical_cols: 
    validation_df[col] = encoders[col].transform(validation_df[col])

X_val = validation_df.drop(label_col, axis=1)
y_val = validation_df[label_col]

# Make predictions
y_pred = log_reg.predict(X_val) 
```


```python
model_performance_metrics(y_val, y_pred)
```

    Accuracy: 0.6231703160551599
    Confusion Matrix:
     [[  1334   1820]
     [ 79995 133965]]
    Precision: 0.01640251324865669
    Recall: 0.4229549778059607
    F1-Score: 0.0315803179337855
    ROC AUC: 0.524538341398774



```python

```
