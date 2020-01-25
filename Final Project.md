
## Introduction

#### Business problem

According to Bloomberg News, the London Housing Market is in a rut. It is now facing a number of different headwinds, including the prospect of higher taxes and a warning from the Bank of England that U.K. home values could fall as much as 30 percent in the event of a disorderly exit from the European Union. More specifically, four overlooked cracks suggest that the London market may be in worse shape than many realize: hidden price falls, record-low sales, homebuilder exodus and tax hikes addressing overseas buyers of homes in England and Wales.

In this scenario, it is urgent to adopt machine learning tools in order to assist homebuyers clientele in London to make wise and effective decisions. As a result, the business problem we are currently posing is: how could we provide support to homebuyers clientele in to purchase a suitable real estate in London in this uncertain economic and financial scenario?

To solve this business problem, we are going to cluster London neighborhoods in order to recommend venues and the current average price of real estate where homebuyers can make a real estate investment. We will recommend profitable venues according to amenities and essential facilities surrounding such venues i.e. elementary schools, high schools, hospitals & grocery stores.

## Data

Data on London properties and the relative price paid data were extracted from the HM Land Registry (http://landregistry.data.gov.uk/). The following fields comprise the address data included in Price Paid Data: Postcode; PAON Primary Addressable Object Name. Typically the house number or name; SAON Secondary Addressable Object Name. If there is a sub-building, for example, the building is divided into flats, there will be a SAON; Street; Locality; Town/City; District; County.

To explore and target recommended locations across different venues according to the presence of amenities and essential facilities, we will access data through FourSquare API interface and arrange them as a dataframe for visualization. By merging data on London properties and the relative price paid data from the HM Land Registry and data on amenities and essential facilities surrounding such properties from FourSquare API interface, we will be able to recommend profitable real estate investments.

## Methodology section

The Methodology section will describe the main components of our analysis and predication system. The Methodology section comprises four stages:

1. Collect Inspection Data
2. Explore and Understand Data
3. Data preparation and preprocessing 
4. Modeling

#### Collect Inspection Data

After importing the necessary libraries, we download the data from the HM Land Registry website as follows:


```python
import os # Operating System
import numpy as np
import pandas as pd
import datetime as dt # Datetime
import json # library to handle JSON files

!conda install -c conda-forge geopy --yes
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

!conda install -c conda-forge folium=0.5.0 --yes
import folium #import folium # map rendering library

print('Libraries imported.')

```

    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/conda/envs/Python36
    
      added / updated specs: 
        - geopy
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        geopy-1.20.0               |             py_0          57 KB  conda-forge
        certifi-2019.11.28         |           py36_0         149 KB  conda-forge
        openssl-1.1.1d             |       h516909a_0         2.1 MB  conda-forge
        ca-certificates-2019.11.28 |       hecc5488_0         145 KB  conda-forge
        geographiclib-1.50         |             py_0          34 KB  conda-forge
        ------------------------------------------------------------
                                               Total:         2.5 MB
    
    The following NEW packages will be INSTALLED:
    
        geographiclib:   1.50-py_0         conda-forge
        geopy:           1.20.0-py_0       conda-forge
    
    The following packages will be UPDATED:
    
        ca-certificates: 2019.11.27-0                  --> 2019.11.28-hecc5488_0 conda-forge
        certifi:         2019.11.28-py36_0             --> 2019.11.28-py36_0     conda-forge
    
    The following packages will be DOWNGRADED:
    
        openssl:         1.1.1d-h7b6447c_3             --> 1.1.1d-h516909a_0     conda-forge
    
    
    Downloading and Extracting Packages
    geopy-1.20.0         | 57 KB     | ##################################### | 100% 
    certifi-2019.11.28   | 149 KB    | ##################################### | 100% 
    openssl-1.1.1d       | 2.1 MB    | ##################################### | 100% 
    ca-certificates-2019 | 145 KB    | ##################################### | 100% 
    geographiclib-1.50   | 34 KB     | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/conda/envs/Python36
    
      added / updated specs: 
        - folium=0.5.0
    
    
    The following packages will be downloaded:
    
        package                    |            build
        ---------------------------|-----------------
        branca-0.3.1               |             py_0          25 KB  conda-forge
        altair-4.0.1               |             py_0         575 KB  conda-forge
        folium-0.5.0               |             py_0          45 KB  conda-forge
        vincent-0.4.4              |             py_1          28 KB  conda-forge
        ------------------------------------------------------------
                                               Total:         673 KB
    
    The following NEW packages will be INSTALLED:
    
        altair:  4.0.1-py_0 conda-forge
        branca:  0.3.1-py_0 conda-forge
        folium:  0.5.0-py_0 conda-forge
        vincent: 0.4.4-py_1 conda-forge
    
    
    Downloading and Extracting Packages
    branca-0.3.1         | 25 KB     | ##################################### | 100% 
    altair-4.0.1         | 575 KB    | ##################################### | 100% 
    folium-0.5.0         | 45 KB     | ##################################### | 100% 
    vincent-0.4.4        | 28 KB     | ##################################### | 100% 
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    Libraries imported.



```python
#Read the data for examination (Source: http://landregistry.data.gov.uk/)
df_ppd = pd.read_csv("http://prod2.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2018.csv")
```

#### Explore and Understand Data

We read the dataset that we collected from the HM Land Registry website into a pandas' data frame and display the first five rows of it as follows:


```python
df_ppd.head()
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
      <th>{666758D7-43A9-3363-E053-6B04A8C0D74E}</th>
      <th>405000</th>
      <th>2018-01-25 00:00</th>
      <th>WR15 8LH</th>
      <th>D</th>
      <th>N</th>
      <th>F</th>
      <th>RAMBLERS WAY</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>BORASTON</th>
      <th>TENBURY WELLS</th>
      <th>SHROPSHIRE</th>
      <th>SHROPSHIRE.1</th>
      <th>A</th>
      <th>A.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{666758D7-43AA-3363-E053-6B04A8C0D74E}</td>
      <td>315000</td>
      <td>2018-01-23 00:00</td>
      <td>SY7 8QA</td>
      <td>D</td>
      <td>N</td>
      <td>F</td>
      <td>MONT CENISE</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CLUN</td>
      <td>CRAVEN ARMS</td>
      <td>SHROPSHIRE</td>
      <td>SHROPSHIRE</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{666758D7-43AD-3363-E053-6B04A8C0D74E}</td>
      <td>165000</td>
      <td>2018-01-19 00:00</td>
      <td>SY1 2BF</td>
      <td>T</td>
      <td>Y</td>
      <td>F</td>
      <td>42</td>
      <td>NaN</td>
      <td>PENSON WAY</td>
      <td>NaN</td>
      <td>SHREWSBURY</td>
      <td>SHROPSHIRE</td>
      <td>SHROPSHIRE</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{666758D7-43B0-3363-E053-6B04A8C0D74E}</td>
      <td>370000</td>
      <td>2018-01-22 00:00</td>
      <td>SY8 4DF</td>
      <td>D</td>
      <td>N</td>
      <td>F</td>
      <td>WILLOW HEY</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ASHFORD CARBONEL</td>
      <td>LUDLOW</td>
      <td>SHROPSHIRE</td>
      <td>SHROPSHIRE</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{666758D7-43B3-3363-E053-6B04A8C0D74E}</td>
      <td>320000</td>
      <td>2018-01-19 00:00</td>
      <td>TF10 7ET</td>
      <td>D</td>
      <td>N</td>
      <td>F</td>
      <td>3</td>
      <td>NaN</td>
      <td>PRINCESS GARDENS</td>
      <td>NaN</td>
      <td>NEWPORT</td>
      <td>WREKIN</td>
      <td>WREKIN</td>
      <td>A</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{666758D7-43B4-3363-E053-6B04A8C0D74E}</td>
      <td>180000</td>
      <td>2018-01-31 00:00</td>
      <td>SY3 0NQ</td>
      <td>S</td>
      <td>N</td>
      <td>F</td>
      <td>79</td>
      <td>NaN</td>
      <td>LYTHWOOD ROAD</td>
      <td>BAYSTON HILL</td>
      <td>SHREWSBURY</td>
      <td>SHROPSHIRE</td>
      <td>SHROPSHIRE</td>
      <td>A</td>
      <td>A</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_ppd.shape
```




    (1027301, 16)



Our dataset consists of over 700000 rows and 16 columns. We will now prepare and preprocess data accordingly.

#### Data preparation and preprocessing

At this stage, we prepare our dataset for the modeling process, opting for the most suitable machine learning algorithm for our scope. Accordingly, we perform the following steps:

1.Rename the column names  
2.Format the date column  
3.Sort data by date of sale  
4.Select data only for the city of London  
5.Make a list of street names in London  
6.Calculate the street-wise average price of the property   
7.Read the street-wise coordinates into a data frame, eliminating recurring word London from individual names  
8.Join the data to find the coordinates of locations which fit into client's budget  
9.Plot recommended locations on London map along with current market prices  


```python
# Assign meaningful column names
df_ppd.columns = ['TUID', 'Price', 'Date_Transfer', 'Postcode', 'Prop_Type', 'Old_New', 'Duration', 'PAON', \
                  'SAON', 'Street', 'Locality', 'Town_City', 'District', 'County', 'PPD_Cat_Type', 'Record_Status']
```


```python
# Format the date column
df_ppd['Date_Transfer'] = df_ppd['Date_Transfer'].apply(pd.to_datetime)

# Delete all obsolete transactions which were done before 2016
df_ppd.drop(df_ppd[df_ppd.Date_Transfer.dt.year < 2016].index, inplace=True)

# Sort by Date of Sale
df_ppd.sort_values(by=['Date_Transfer'],ascending=[False],inplace=True)
```


```python
df_ppd_london = df_ppd.query("Town_City == 'LONDON'")

# Make a list of street names in LONDON
streets = df_ppd_london['Street'].unique().tolist()
```


```python
df_grp_price = df_ppd_london.groupby(['Street'])['Price'].mean().reset_index()

# Give meaningful names to the columns
df_grp_price.columns = ['Street', 'Avg_Price']
```


```python
#Input your Budget's Upper Limit and Lower Limit - Find the locations df_grp_price which fits your budget
df_affordable = df_grp_price.query("(Avg_Price >= 2200000) & (Avg_Price <= 2500000)")
```


```python
# Display the dataframe
df_affordable
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
      <th>Street</th>
      <th>Avg_Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2.450000e+06</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2.435000e+06</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2.250000e+06</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2.400000e+06</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2.375000e+06</td>
    </tr>
    <tr>
      <th>981</th>
      <td>BEAUCLERC ROAD</td>
      <td>2.480000e+06</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>BELVEDERE DRIVE</td>
      <td>2.340000e+06</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>BICKENHALL STREET</td>
      <td>2.208500e+06</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>2.217000e+06</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>BRAMPTON GROVE</td>
      <td>2.456875e+06</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>BRIARDALE GARDENS</td>
      <td>2.397132e+06</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>BROOKWAY</td>
      <td>2.400000e+06</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>BURBAGE ROAD</td>
      <td>2.445000e+06</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>BURY WALK</td>
      <td>2.492500e+06</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>CALLCOTT STREET</td>
      <td>2.375000e+06</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>2.379653e+06</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>CAMPION ROAD</td>
      <td>2.461000e+06</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>CANNING PLACE</td>
      <td>2.425000e+06</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>CARLISLE ROAD</td>
      <td>2.200000e+06</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>CARLTON GARDENS</td>
      <td>2.483500e+06</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>CARLYLE COURT</td>
      <td>2.300000e+06</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>CHALCOT SQUARE</td>
      <td>2.286679e+06</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>CHARLES LANE</td>
      <td>2.414000e+06</td>
    </tr>
    <tr>
      <th>2561</th>
      <td>CHELSEA CRESCENT</td>
      <td>2.495000e+06</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>2.450000e+06</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>CHEYNE COURT</td>
      <td>2.250000e+06</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>CHEYNE ROW</td>
      <td>2.410000e+06</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>CHISWICK MALL</td>
      <td>2.287500e+06</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>CITY ROAD</td>
      <td>2.468340e+06</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>CLARENDON STREET</td>
      <td>2.250000e+06</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10924</th>
      <td>RUSSELL GARDENS MEWS</td>
      <td>2.300000e+06</td>
    </tr>
    <tr>
      <th>11171</th>
      <td>SETTLES STREET</td>
      <td>2.487500e+06</td>
    </tr>
    <tr>
      <th>11246</th>
      <td>SHELDON AVENUE</td>
      <td>2.349542e+06</td>
    </tr>
    <tr>
      <th>11480</th>
      <td>SOUTH END ROW</td>
      <td>2.470000e+06</td>
    </tr>
    <tr>
      <th>11558</th>
      <td>SOUTHWOOD LAWN ROAD</td>
      <td>2.350000e+06</td>
    </tr>
    <tr>
      <th>11561</th>
      <td>SOVEREIGN PARK</td>
      <td>2.500000e+06</td>
    </tr>
    <tr>
      <th>11778</th>
      <td>ST MARGARETS CRESCENT</td>
      <td>2.216500e+06</td>
    </tr>
    <tr>
      <th>11814</th>
      <td>ST OSWALDS PLACE</td>
      <td>2.250000e+06</td>
    </tr>
    <tr>
      <th>11834</th>
      <td>ST PETERS SQUARE</td>
      <td>2.468730e+06</td>
    </tr>
    <tr>
      <th>11871</th>
      <td>STAFFORD TERRACE</td>
      <td>2.355000e+06</td>
    </tr>
    <tr>
      <th>12210</th>
      <td>SUTHERLAND PLACE</td>
      <td>2.456000e+06</td>
    </tr>
    <tr>
      <th>12272</th>
      <td>SYDNEY STREET</td>
      <td>2.240833e+06</td>
    </tr>
    <tr>
      <th>12414</th>
      <td>THAMES BANK</td>
      <td>2.400000e+06</td>
    </tr>
    <tr>
      <th>12476</th>
      <td>THE HEXAGON</td>
      <td>2.335000e+06</td>
    </tr>
    <tr>
      <th>12755</th>
      <td>TREDEGAR SQUARE</td>
      <td>2.436666e+06</td>
    </tr>
    <tr>
      <th>12817</th>
      <td>TRINITY STREET</td>
      <td>2.317500e+06</td>
    </tr>
    <tr>
      <th>12967</th>
      <td>UPPER HAMPSTEAD WALK</td>
      <td>2.500000e+06</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>WALPOLE GARDENS</td>
      <td>2.303500e+06</td>
    </tr>
    <tr>
      <th>13231</th>
      <td>WALPOLE STREET</td>
      <td>2.242500e+06</td>
    </tr>
    <tr>
      <th>13305</th>
      <td>WARWICK SQUARE</td>
      <td>2.432273e+06</td>
    </tr>
    <tr>
      <th>13403</th>
      <td>WELBECK WAY</td>
      <td>2.267000e+06</td>
    </tr>
    <tr>
      <th>13415</th>
      <td>WELLESLEY TERRACE</td>
      <td>2.410000e+06</td>
    </tr>
    <tr>
      <th>13426</th>
      <td>WELLINGTON STREET</td>
      <td>2.293155e+06</td>
    </tr>
    <tr>
      <th>13558</th>
      <td>WESTMORELAND PLACE</td>
      <td>2.300000e+06</td>
    </tr>
    <tr>
      <th>13665</th>
      <td>WHITFIELD STREET</td>
      <td>2.451000e+06</td>
    </tr>
    <tr>
      <th>13710</th>
      <td>WILFRED STREET</td>
      <td>2.410538e+06</td>
    </tr>
    <tr>
      <th>13736</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>2.425000e+06</td>
    </tr>
    <tr>
      <th>13756</th>
      <td>WILSON STREET</td>
      <td>2.257500e+06</td>
    </tr>
    <tr>
      <th>13784</th>
      <td>WINCHENDON ROAD</td>
      <td>2.350000e+06</td>
    </tr>
    <tr>
      <th>13821</th>
      <td>WINGATE ROAD</td>
      <td>2.206400e+06</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 2 columns</p>
</div>




```python
import pandas as pd
import numpy as np
import datetime as DT
import hmac
from geopy.geocoders import Nominatim
from geopy.distance import vincenty
# import k-means from clustering stage
from sklearn.cluster import KMeans
```


```python
for index, item in df_affordable.iterrows():
    print(f"index: {index}")
    print(f"item: {item}")
    print(f"item.Street only: {item.Street}")
```

    index: 196
    item: Street       ALBION SQUARE
    Avg_Price         2.45e+06
    Name: 196, dtype: object
    item.Street only: ALBION SQUARE
    index: 391
    item: Street       ANHALT ROAD
    Avg_Price      2.435e+06
    Name: 391, dtype: object
    item.Street only: ANHALT ROAD
    index: 406
    item: Street       ANSDELL TERRACE
    Avg_Price           2.25e+06
    Name: 406, dtype: object
    item.Street only: ANSDELL TERRACE
    index: 422
    item: Street       APPLEGARTH ROAD
    Avg_Price            2.4e+06
    Name: 422, dtype: object
    item.Street only: APPLEGARTH ROAD
    index: 855
    item: Street       BARONSMEAD ROAD
    Avg_Price          2.375e+06
    Name: 855, dtype: object
    item.Street only: BARONSMEAD ROAD
    index: 981
    item: Street       BEAUCLERC ROAD
    Avg_Price          2.48e+06
    Name: 981, dtype: object
    item.Street only: BEAUCLERC ROAD
    index: 1102
    item: Street       BELVEDERE DRIVE
    Avg_Price           2.34e+06
    Name: 1102, dtype: object
    item.Street only: BELVEDERE DRIVE
    index: 1215
    item: Street       BICKENHALL STREET
    Avg_Price           2.2085e+06
    Name: 1215, dtype: object
    item.Street only: BICKENHALL STREET
    index: 1253
    item: Street       BIRCHLANDS AVENUE
    Avg_Price            2.217e+06
    Name: 1253, dtype: object
    item.Street only: BIRCHLANDS AVENUE
    index: 1553
    item: Street       BRAMPTON GROVE
    Avg_Price       2.45688e+06
    Name: 1553, dtype: object
    item.Street only: BRAMPTON GROVE
    index: 1632
    item: Street       BRIARDALE GARDENS
    Avg_Price          2.39713e+06
    Name: 1632, dtype: object
    item.Street only: BRIARDALE GARDENS
    index: 1797
    item: Street       BROOKWAY
    Avg_Price     2.4e+06
    Name: 1797, dtype: object
    item.Street only: BROOKWAY
    index: 1914
    item: Street       BURBAGE ROAD
    Avg_Price       2.445e+06
    Name: 1914, dtype: object
    item.Street only: BURBAGE ROAD
    index: 1980
    item: Street        BURY WALK
    Avg_Price    2.4925e+06
    Name: 1980, dtype: object
    item.Street only: BURY WALK
    index: 2068
    item: Street       CALLCOTT STREET
    Avg_Price          2.375e+06
    Name: 2068, dtype: object
    item.Street only: CALLCOTT STREET
    index: 2129
    item: Street       CAMPDEN HILL ROAD
    Avg_Price          2.37965e+06
    Name: 2129, dtype: object
    item.Street only: CAMPDEN HILL ROAD
    index: 2136
    item: Street       CAMPION ROAD
    Avg_Price       2.461e+06
    Name: 2136, dtype: object
    item.Street only: CAMPION ROAD
    index: 2158
    item: Street       CANNING PLACE
    Avg_Price        2.425e+06
    Name: 2158, dtype: object
    item.Street only: CANNING PLACE
    index: 2225
    item: Street       CARLISLE ROAD
    Avg_Price          2.2e+06
    Name: 2225, dtype: object
    item.Street only: CARLISLE ROAD
    index: 2230
    item: Street       CARLTON GARDENS
    Avg_Price         2.4835e+06
    Name: 2230, dtype: object
    item.Street only: CARLTON GARDENS
    index: 2242
    item: Street       CARLYLE COURT
    Avg_Price          2.3e+06
    Name: 2242, dtype: object
    item.Street only: CARLYLE COURT
    index: 2405
    item: Street       CHALCOT SQUARE
    Avg_Price       2.28668e+06
    Name: 2405, dtype: object
    item.Street only: CHALCOT SQUARE
    index: 2483
    item: Street       CHARLES LANE
    Avg_Price       2.414e+06
    Name: 2483, dtype: object
    item.Street only: CHARLES LANE
    index: 2561
    item: Street       CHELSEA CRESCENT
    Avg_Price           2.495e+06
    Name: 2561, dtype: object
    item.Street only: CHELSEA CRESCENT
    index: 2605
    item: Street       CHESTER CLOSE NORTH
    Avg_Price               2.45e+06
    Name: 2605, dtype: object
    item.Street only: CHESTER CLOSE NORTH
    index: 2637
    item: Street       CHEYNE COURT
    Avg_Price        2.25e+06
    Name: 2637, dtype: object
    item.Street only: CHEYNE COURT
    index: 2640
    item: Street       CHEYNE ROW
    Avg_Price      2.41e+06
    Name: 2640, dtype: object
    item.Street only: CHEYNE ROW
    index: 2685
    item: Street       CHISWICK MALL
    Avg_Price       2.2875e+06
    Name: 2685, dtype: object
    item.Street only: CHISWICK MALL
    index: 2760
    item: Street         CITY ROAD
    Avg_Price    2.46834e+06
    Name: 2760, dtype: object
    item.Street only: CITY ROAD
    index: 2807
    item: Street       CLARENDON STREET
    Avg_Price            2.25e+06
    Name: 2807, dtype: object
    item.Street only: CLARENDON STREET
    index: 2899
    item: Street       CLONCURRY STREET
    Avg_Price         2.38833e+06
    Name: 2899, dtype: object
    item.Street only: CLONCURRY STREET
    index: 2943
    item: Street       COLBECK MEWS
    Avg_Price      2.3675e+06
    Name: 2943, dtype: object
    item.Street only: COLBECK MEWS
    index: 2994
    item: Street       COLLEGE CRESCENT
    Avg_Price            2.44e+06
    Name: 2994, dtype: object
    item.Street only: COLLEGE CRESCENT
    index: 3201
    item: Street       CORNWALL TERRACE MEWS
    Avg_Price                 2.35e+06
    Name: 3201, dtype: object
    item.Street only: CORNWALL TERRACE MEWS
    index: 3254
    item: Street       COURT LANE GARDENS
    Avg_Price              2.36e+06
    Name: 3254, dtype: object
    item.Street only: COURT LANE GARDENS
    index: 3376
    item: Street       CRESCENT GROVE
    Avg_Price         2.298e+06
    Name: 3376, dtype: object
    item.Street only: CRESCENT GROVE
    index: 3582
    item: Street       DALEBURY ROAD
    Avg_Price          2.4e+06
    Name: 3582, dtype: object
    item.Street only: DALEBURY ROAD
    index: 3847
    item: Street       DEWHURST ROAD
    Avg_Price        2.425e+06
    Name: 3847, dtype: object
    item.Street only: DEWHURST ROAD
    index: 3928
    item: Street       DORIA ROAD
    Avg_Price    2.3625e+06
    Name: 3928, dtype: object
    item.Street only: DORIA ROAD
    index: 3979
    item: Street       DOWNSHIRE HILL
    Avg_Price         2.225e+06
    Name: 3979, dtype: object
    item.Street only: DOWNSHIRE HILL
    index: 4034
    item: Street       DUCHESS WALK
    Avg_Price      2.4775e+06
    Name: 4034, dtype: object
    item.Street only: DUCHESS WALK
    index: 4231
    item: Street       ECCLESTON SQUARE MEWS
    Avg_Price               2.3355e+06
    Name: 4231, dtype: object
    item.Street only: ECCLESTON SQUARE MEWS
    index: 4284
    item: Street       EGBERT STREET
    Avg_Price        2.265e+06
    Name: 4284, dtype: object
    item.Street only: EGBERT STREET
    index: 4288
    item: Street       EGERTON PLACE
    Avg_Price          2.2e+06
    Name: 4288, dtype: object
    item.Street only: EGERTON PLACE
    index: 4373
    item: Street       ELM PARK ROAD
    Avg_Price      2.32042e+06
    Name: 4373, dtype: object
    item.Street only: ELM PARK ROAD
    index: 4891
    item: Street       FLORAL STREET
    Avg_Price      2.22722e+06
    Name: 4891, dtype: object
    item.Street only: FLORAL STREET
    index: 5012
    item: Street       FRANK DIXON WAY
    Avg_Price         2.2125e+06
    Name: 5012, dtype: object
    item.Street only: FRANK DIXON WAY
    index: 5094
    item: Street       FULTON MEWS
    Avg_Price      2.299e+06
    Name: 5094, dtype: object
    item.Street only: FULTON MEWS
    index: 5237
    item: Street       GERARD ROAD
    Avg_Price     2.2585e+06
    Name: 5237, dtype: object
    item.Street only: GERARD ROAD
    index: 5240
    item: Street       GERRARD ROAD
    Avg_Price      2.2425e+06
    Name: 5240, dtype: object
    item.Street only: GERRARD ROAD
    index: 5285
    item: Street       GIRDLERS ROAD
    Avg_Price      2.44167e+06
    Name: 5285, dtype: object
    item.Street only: GIRDLERS ROAD
    index: 5378
    item: Street       GLOUCESTER CRESCENT
    Avg_Price            2.35083e+06
    Name: 5378, dtype: object
    item.Street only: GLOUCESTER CRESCENT
    index: 5446
    item: Street       GORDON PLACE
    Avg_Price       2.477e+06
    Name: 5446, dtype: object
    item.Street only: GORDON PLACE
    index: 5482
    item: Street       GRAFTON SQUARE
    Avg_Price          2.27e+06
    Name: 5482, dtype: object
    item.Street only: GRAFTON SQUARE
    index: 5489
    item: Street       GRAHAM TERRACE
    Avg_Price         2.325e+06
    Name: 5489, dtype: object
    item.Street only: GRAHAM TERRACE
    index: 5947
    item: Street       HARMAN DRIVE
    Avg_Price      2.2625e+06
    Name: 5947, dtype: object
    item.Street only: HARMAN DRIVE
    index: 5971
    item: Street       HARRIS STREET
    Avg_Price      2.47177e+06
    Name: 5971, dtype: object
    item.Street only: HARRIS STREET
    index: 6034
    item: Street       HAVANNAH STREET
    Avg_Price        2.21731e+06
    Name: 6034, dtype: object
    item.Street only: HAVANNAH STREET
    index: 6106
    item: Street       HAZLEWELL ROAD
    Avg_Price           2.5e+06
    Name: 6106, dtype: object
    item.Street only: HAZLEWELL ROAD
    index: 6222
    item: Street       HEREFORD MEWS
    Avg_Price         2.31e+06
    Name: 6222, dtype: object
    item.Street only: HEREFORD MEWS
    index: 6242
    item: Street       HERONDALE AVENUE
    Avg_Price           2.475e+06
    Name: 6242, dtype: object
    item.Street only: HERONDALE AVENUE
    index: 6339
    item: Street       HIGHGATE HIGH STREET
    Avg_Price               2.211e+06
    Name: 6339, dtype: object
    item.Street only: HIGHGATE HIGH STREET
    index: 6354
    item: Street       HIGHWOOD HILL
    Avg_Price       2.2525e+06
    Name: 6354, dtype: object
    item.Street only: HIGHWOOD HILL
    index: 6390
    item: Street       HILLGATE PLACE
    Avg_Price           2.2e+06
    Name: 6390, dtype: object
    item.Street only: HILLGATE PLACE
    index: 6501
    item: Street       HOLLYCROFT AVENUE
    Avg_Price          2.36138e+06
    Name: 6501, dtype: object
    item.Street only: HOLLYCROFT AVENUE
    index: 6505
    item: Street       HOLLYWOOD MEWS
    Avg_Price          2.35e+06
    Name: 6505, dtype: object
    item.Street only: HOLLYWOOD MEWS
    index: 6550
    item: Street       HONEYWELL ROAD
    Avg_Price       2.27833e+06
    Name: 6550, dtype: object
    item.Street only: HONEYWELL ROAD
    index: 6606
    item: Street       HORTENSIA ROAD
    Avg_Price       2.27592e+06
    Name: 6606, dtype: object
    item.Street only: HORTENSIA ROAD
    index: 6635
    item: Street       HOXTON SQUARE
    Avg_Price      2.23429e+06
    Name: 6635, dtype: object
    item.Street only: HOXTON SQUARE
    index: 6661
    item: Street       HUNTER ROAD
    Avg_Price        2.3e+06
    Name: 6661, dtype: object
    item.Street only: HUNTER ROAD
    index: 6820
    item: Street       JACKSONS LANE
    Avg_Price       2.3625e+06
    Name: 6820, dtype: object
    item.Street only: JACKSONS LANE
    index: 6880
    item: Street       JOHN STREET
    Avg_Price      2.235e+06
    Name: 6880, dtype: object
    item.Street only: JOHN STREET
    index: 7184
    item: Street       KINNERTON STREET
    Avg_Price          2.4856e+06
    Name: 7184, dtype: object
    item.Street only: KINNERTON STREET
    index: 7218
    item: Street       KNARESBOROUGH PLACE
    Avg_Price              2.325e+06
    Name: 7218, dtype: object
    item.Street only: KNARESBOROUGH PLACE
    index: 7238
    item: Street       KNOX STREET
    Avg_Price       2.25e+06
    Name: 7238, dtype: object
    item.Street only: KNOX STREET
    index: 7258
    item: Street       LADBROKE GROVE
    Avg_Price        2.4833e+06
    Name: 7258, dtype: object
    item.Street only: LADBROKE GROVE
    index: 7327
    item: Street       LANCASTER MEWS
    Avg_Price        2.3125e+06
    Name: 7327, dtype: object
    item.Street only: LANCASTER MEWS
    index: 7397
    item: Street       LANSDOWNE ROAD
    Avg_Price       2.36488e+06
    Name: 7397, dtype: object
    item.Street only: LANSDOWNE ROAD
    index: 7425
    item: Street       LATIMER INDUSTRIAL ESTATE
    Avg_Price                      2.4e+06
    Name: 7425, dtype: object
    item.Street only: LATIMER INDUSTRIAL ESTATE
    index: 7478
    item: Street       LAXTON PLACE
    Avg_Price         2.5e+06
    Name: 7478, dtype: object
    item.Street only: LAXTON PLACE
    index: 7680
    item: Street       LINCOLN AVENUE
    Avg_Price        2.2035e+06
    Name: 7680, dtype: object
    item.Street only: LINCOLN AVENUE
    index: 7709
    item: Street       LINGFIELD ROAD
    Avg_Price       2.24875e+06
    Name: 7709, dtype: object
    item.Street only: LINGFIELD ROAD
    index: 7738
    item: Street       LISSON STREET
    Avg_Price       2.4625e+06
    Name: 7738, dtype: object
    item.Street only: LISSON STREET
    index: 7772
    item: Street       LIVERPOOL GROVE
    Avg_Price          2.288e+06
    Name: 7772, dtype: object
    item.Street only: LIVERPOOL GROVE
    index: 7864
    item: Street       LONGWOOD DRIVE
    Avg_Price         2.375e+06
    Name: 7864, dtype: object
    item.Street only: LONGWOOD DRIVE
    index: 7870
    item: Street       LONSDALE SQUARE
    Avg_Price         2.3575e+06
    Name: 7870, dtype: object
    item.Street only: LONSDALE SQUARE
    index: 8408
    item: Street       MAZE HILL
    Avg_Price     2.25e+06
    Name: 8408, dtype: object
    item.Street only: MAZE HILL
    index: 8543
    item: Street       MIDDLESEX PASSAGE
    Avg_Price             2.28e+06
    Name: 8543, dtype: object
    item.Street only: MIDDLESEX PASSAGE
    index: 8706
    item: Street       MONTPELIER AVENUE
    Avg_Price              2.5e+06
    Name: 8706, dtype: object
    item.Street only: MONTPELIER AVENUE
    index: 8716
    item: Street       MONTPELIER WALK
    Avg_Price           2.32e+06
    Name: 8716, dtype: object
    item.Street only: MONTPELIER WALK
    index: 8864
    item: Street       MULTON ROAD
    Avg_Price        2.3e+06
    Name: 8864, dtype: object
    item.Street only: MULTON ROAD
    index: 8869
    item: Street       MUNDEN STREET
    Avg_Price         2.25e+06
    Name: 8869, dtype: object
    item.Street only: MUNDEN STREET
    index: 9143
    item: Street       NORFOLK CRESCENT
    Avg_Price         2.22333e+06
    Name: 9143, dtype: object
    item.Street only: NORFOLK CRESCENT
    index: 9166
    item: Street       NORTH CIRCULAR ROAD
    Avg_Price            2.39314e+06
    Name: 9166, dtype: object
    item.Street only: NORTH CIRCULAR ROAD
    index: 9242
    item: Street       NOTTINGHAM STREET
    Avg_Price           2.2275e+06
    Name: 9242, dtype: object
    item.Street only: NOTTINGHAM STREET
    index: 9317
    item: Street       OAKLEY STREET
    Avg_Price      2.38129e+06
    Name: 9317, dtype: object
    item.Street only: OAKLEY STREET
    index: 9328
    item: Street       OAKWOOD COURT
    Avg_Price      2.34875e+06
    Name: 9328, dtype: object
    item.Street only: OAKWOOD COURT
    index: 9336
    item: Street       OBSERVATORY GARDENS
    Avg_Price               2.42e+06
    Name: 9336, dtype: object
    item.Street only: OBSERVATORY GARDENS
    index: 9371
    item: Street       OLD COURT PLACE
    Avg_Price          2.395e+06
    Name: 9371, dtype: object
    item.Street only: OLD COURT PLACE
    index: 9432
    item: Street       ONSLOW MEWS WEST
    Avg_Price             2.3e+06
    Name: 9432, dtype: object
    item.Street only: ONSLOW MEWS WEST
    index: 9564
    item: Street       PALACE PLACE
    Avg_Price         2.3e+06
    Name: 9564, dtype: object
    item.Street only: PALACE PLACE
    index: 9593
    item: Street       PANTON STREET
    Avg_Price        2.475e+06
    Name: 9593, dtype: object
    item.Street only: PANTON STREET
    index: 9612
    item: Street       PARK CRESCENT
    Avg_Price         2.44e+06
    Name: 9612, dtype: object
    item.Street only: PARK CRESCENT
    index: 9621
    item: Street        PARK LANE
    Avg_Price    2.2415e+06
    Name: 9621, dtype: object
    item.Street only: PARK LANE
    index: 9642
    item: Street       PARKE ROAD
    Avg_Price    2.2625e+06
    Name: 9642, dtype: object
    item.Street only: PARKE ROAD
    index: 9645
    item: Street       PARKFIELDS
    Avg_Price       2.2e+06
    Name: 9645, dtype: object
    item.Street only: PARKFIELDS
    index: 9682
    item: Street       PARTHENIA ROAD
    Avg_Price       2.20057e+06
    Name: 9682, dtype: object
    item.Street only: PARTHENIA ROAD
    index: 9714
    item: Street       PAVILION ROAD
    Avg_Price          2.2e+06
    Name: 9714, dtype: object
    item.Street only: PAVILION ROAD
    index: 9775
    item: Street       PEMBRIDGE MEWS
    Avg_Price         2.251e+06
    Name: 9775, dtype: object
    item.Street only: PEMBRIDGE MEWS
    index: 9777
    item: Street       PEMBRIDGE ROAD
    Avg_Price           2.4e+06
    Name: 9777, dtype: object
    item.Street only: PEMBRIDGE ROAD
    index: 9786
    item: Street       PEMBROKE STUDIOS
    Avg_Price            2.45e+06
    Name: 9786, dtype: object
    item.Street only: PEMBROKE STUDIOS
    index: 9791
    item: Street       PENCOMBE MEWS
    Avg_Price          2.2e+06
    Name: 9791, dtype: object
    item.Street only: PENCOMBE MEWS
    index: 9871
    item: Street       PETERSHAM PLACE
    Avg_Price            2.3e+06
    Name: 9871, dtype: object
    item.Street only: PETERSHAM PLACE
    index: 9886
    item: Street       PHILLIMORE GARDENS
    Avg_Price           2.48467e+06
    Name: 9886, dtype: object
    item.Street only: PHILLIMORE GARDENS
    index: 9894
    item: Street       PHYSIC PLACE
    Avg_Price         2.5e+06
    Name: 9894, dtype: object
    item.Street only: PHYSIC PLACE
    index: 9932
    item: Street       PITFIELD STREET
    Avg_Price        2.48333e+06
    Name: 9932, dtype: object
    item.Street only: PITFIELD STREET
    index: 10142
    item: Street       PRINCES GATE
    Avg_Price     2.40333e+06
    Name: 10142, dtype: object
    item.Street only: PRINCES GATE
    index: 10170
    item: Street       PRIORY ROAD
    Avg_Price    2.24826e+06
    Name: 10170, dtype: object
    item.Street only: PRIORY ROAD
    index: 10181
    item: Street       PROTHERO GARDENS
    Avg_Price          2.2025e+06
    Name: 10181, dtype: object
    item.Street only: PROTHERO GARDENS
    index: 10217
    item: Street       PUTNEY HIGH STREET
    Avg_Price            2.3486e+06
    Name: 10217, dtype: object
    item.Street only: PUTNEY HIGH STREET
    index: 10234
    item: Street       QUARRENDON STREET
    Avg_Price          2.43775e+06
    Name: 10234, dtype: object
    item.Street only: QUARRENDON STREET
    index: 10267
    item: Street       QUEENS GATE TERRACE
    Avg_Price            2.39944e+06
    Name: 10267, dtype: object
    item.Street only: QUEENS GATE TERRACE
    index: 10321
    item: Street       RADSTOCK STREET
    Avg_Price         2.2175e+06
    Name: 10321, dtype: object
    item.Street only: RADSTOCK STREET
    index: 10366
    item: Street       RANELAGH AVENUE
    Avg_Price            2.3e+06
    Name: 10366, dtype: object
    item.Street only: RANELAGH AVENUE
    index: 10473
    item: Street       REDCLIFFE ROAD
    Avg_Price       2.44875e+06
    Name: 10473, dtype: object
    item.Street only: REDCLIFFE ROAD
    index: 10498
    item: Street       REEVES MEWS
    Avg_Price       2.45e+06
    Name: 10498, dtype: object
    item.Street only: REEVES MEWS
    index: 10555
    item: Street       RHEIDOL MEWS
    Avg_Price        2.31e+06
    Name: 10555, dtype: object
    item.Street only: RHEIDOL MEWS
    index: 10610
    item: Street       RINGWOOD AVENUE
    Avg_Price          2.275e+06
    Name: 10610, dtype: object
    item.Street only: RINGWOOD AVENUE
    index: 10690
    item: Street       RODERICK ROAD
    Avg_Price          2.4e+06
    Name: 10690, dtype: object
    item.Street only: RODERICK ROAD
    index: 10747
    item: Street       ROPEMAKERS FIELDS
    Avg_Price              2.5e+06
    Name: 10747, dtype: object
    item.Street only: ROPEMAKERS FIELDS
    index: 10861
    item: Street       ROYAL CRESCENT
    Avg_Price       2.34833e+06
    Name: 10861, dtype: object
    item.Street only: ROYAL CRESCENT
    index: 10867
    item: Street       ROYAL HILL
    Avg_Price    2.2525e+06
    Name: 10867, dtype: object
    item.Street only: ROYAL HILL
    index: 10924
    item: Street       RUSSELL GARDENS MEWS
    Avg_Price                 2.3e+06
    Name: 10924, dtype: object
    item.Street only: RUSSELL GARDENS MEWS
    index: 11171
    item: Street       SETTLES STREET
    Avg_Price        2.4875e+06
    Name: 11171, dtype: object
    item.Street only: SETTLES STREET
    index: 11246
    item: Street       SHELDON AVENUE
    Avg_Price       2.34954e+06
    Name: 11246, dtype: object
    item.Street only: SHELDON AVENUE
    index: 11480
    item: Street       SOUTH END ROW
    Avg_Price         2.47e+06
    Name: 11480, dtype: object
    item.Street only: SOUTH END ROW
    index: 11558
    item: Street       SOUTHWOOD LAWN ROAD
    Avg_Price               2.35e+06
    Name: 11558, dtype: object
    item.Street only: SOUTHWOOD LAWN ROAD
    index: 11561
    item: Street       SOVEREIGN PARK
    Avg_Price           2.5e+06
    Name: 11561, dtype: object
    item.Street only: SOVEREIGN PARK
    index: 11778
    item: Street       ST MARGARETS CRESCENT
    Avg_Price               2.2165e+06
    Name: 11778, dtype: object
    item.Street only: ST MARGARETS CRESCENT
    index: 11814
    item: Street       ST OSWALDS PLACE
    Avg_Price            2.25e+06
    Name: 11814, dtype: object
    item.Street only: ST OSWALDS PLACE
    index: 11834
    item: Street       ST PETERS SQUARE
    Avg_Price         2.46873e+06
    Name: 11834, dtype: object
    item.Street only: ST PETERS SQUARE
    index: 11871
    item: Street       STAFFORD TERRACE
    Avg_Price           2.355e+06
    Name: 11871, dtype: object
    item.Street only: STAFFORD TERRACE
    index: 12210
    item: Street       SUTHERLAND PLACE
    Avg_Price           2.456e+06
    Name: 12210, dtype: object
    item.Street only: SUTHERLAND PLACE
    index: 12272
    item: Street       SYDNEY STREET
    Avg_Price      2.24083e+06
    Name: 12272, dtype: object
    item.Street only: SYDNEY STREET
    index: 12414
    item: Street       THAMES BANK
    Avg_Price        2.4e+06
    Name: 12414, dtype: object
    item.Street only: THAMES BANK
    index: 12476
    item: Street       THE HEXAGON
    Avg_Price      2.335e+06
    Name: 12476, dtype: object
    item.Street only: THE HEXAGON
    index: 12755
    item: Street       TREDEGAR SQUARE
    Avg_Price        2.43667e+06
    Name: 12755, dtype: object
    item.Street only: TREDEGAR SQUARE
    index: 12817
    item: Street       TRINITY STREET
    Avg_Price        2.3175e+06
    Name: 12817, dtype: object
    item.Street only: TRINITY STREET
    index: 12967
    item: Street       UPPER HAMPSTEAD WALK
    Avg_Price                 2.5e+06
    Name: 12967, dtype: object
    item.Street only: UPPER HAMPSTEAD WALK
    index: 13229
    item: Street       WALPOLE GARDENS
    Avg_Price         2.3035e+06
    Name: 13229, dtype: object
    item.Street only: WALPOLE GARDENS
    index: 13231
    item: Street       WALPOLE STREET
    Avg_Price        2.2425e+06
    Name: 13231, dtype: object
    item.Street only: WALPOLE STREET
    index: 13305
    item: Street       WARWICK SQUARE
    Avg_Price       2.43227e+06
    Name: 13305, dtype: object
    item.Street only: WARWICK SQUARE
    index: 13403
    item: Street       WELBECK WAY
    Avg_Price      2.267e+06
    Name: 13403, dtype: object
    item.Street only: WELBECK WAY
    index: 13415
    item: Street       WELLESLEY TERRACE
    Avg_Price             2.41e+06
    Name: 13415, dtype: object
    item.Street only: WELLESLEY TERRACE
    index: 13426
    item: Street       WELLINGTON STREET
    Avg_Price          2.29316e+06
    Name: 13426, dtype: object
    item.Street only: WELLINGTON STREET
    index: 13558
    item: Street       WESTMORELAND PLACE
    Avg_Price               2.3e+06
    Name: 13558, dtype: object
    item.Street only: WESTMORELAND PLACE
    index: 13665
    item: Street       WHITFIELD STREET
    Avg_Price           2.451e+06
    Name: 13665, dtype: object
    item.Street only: WHITFIELD STREET
    index: 13710
    item: Street       WILFRED STREET
    Avg_Price       2.41054e+06
    Name: 13710, dtype: object
    item.Street only: WILFRED STREET
    index: 13736
    item: Street       WILLOW BRIDGE ROAD
    Avg_Price             2.425e+06
    Name: 13736, dtype: object
    item.Street only: WILLOW BRIDGE ROAD
    index: 13756
    item: Street       WILSON STREET
    Avg_Price       2.2575e+06
    Name: 13756, dtype: object
    item.Street only: WILSON STREET
    index: 13784
    item: Street       WINCHENDON ROAD
    Avg_Price           2.35e+06
    Name: 13784, dtype: object
    item.Street only: WINCHENDON ROAD
    index: 13821
    item: Street       WINGATE ROAD
    Avg_Price      2.2064e+06
    Name: 13821, dtype: object
    item.Street only: WINGATE ROAD



```python
geolocator = Nominatim()
```

    /opt/conda/envs/Python36/lib/python3.6/site-packages/ipykernel/__main__.py:1: DeprecationWarning: Using Nominatim with the default "geopy/1.20.0" `user_agent` is strongly discouraged, as it violates Nominatim's ToS https://operations.osmfoundation.org/policies/nominatim/ and may possibly cause 403 and 429 HTTP errors. Please specify a custom `user_agent` with `Nominatim(user_agent="my-application")` or by overriding the default `user_agent`: `geopy.geocoders.options.default_user_agent = "my-application"`. In geopy 2.0 this will become an exception.
      if __name__ == '__main__':



```python
df_affordable['city_coord'] = df_affordable['Street'].apply(geolocator.geocode).apply(lambda x: (x.latitude, x.longitude))
```

    /opt/conda/envs/Python36/lib/python3.6/site-packages/ipykernel/__main__.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':



```python
df_affordable
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
      <th>Street</th>
      <th>Avg_Price</th>
      <th>city_coord</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2.450000e+06</td>
      <td>(-41.27375755, 173.28939323910353)</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2.435000e+06</td>
      <td>(51.4803164, -0.1668011)</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2.250000e+06</td>
      <td>(51.4998899, -0.1891027)</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2.400000e+06</td>
      <td>(53.7486539, -0.3266704)</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2.375000e+06</td>
      <td>(51.4773147, -0.239457)</td>
    </tr>
    <tr>
      <th>981</th>
      <td>BEAUCLERC ROAD</td>
      <td>2.480000e+06</td>
      <td>(51.4995771, -0.2290331)</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>BELVEDERE DRIVE</td>
      <td>2.340000e+06</td>
      <td>(52.4142089, 1.7244152)</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>BICKENHALL STREET</td>
      <td>2.208500e+06</td>
      <td>(51.5212014, -0.1589082)</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>2.217000e+06</td>
      <td>(51.4483941, -0.1604676)</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>BRAMPTON GROVE</td>
      <td>2.456875e+06</td>
      <td>(51.5899607, -0.3185249)</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>BRIARDALE GARDENS</td>
      <td>2.397132e+06</td>
      <td>(51.5601748, -0.1954305)</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>BROOKWAY</td>
      <td>2.400000e+06</td>
      <td>(45.432184899999996, -122.80281166115779)</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>BURBAGE ROAD</td>
      <td>2.445000e+06</td>
      <td>(51.4482603, -0.0885073)</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>BURY WALK</td>
      <td>2.492500e+06</td>
      <td>(52.1455294, -0.4235933)</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>CALLCOTT STREET</td>
      <td>2.375000e+06</td>
      <td>(51.5083499, -0.1983276)</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>2.379653e+06</td>
      <td>(51.50141, -0.1951157)</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>CAMPION ROAD</td>
      <td>2.461000e+06</td>
      <td>(52.6813749, 0.9654713)</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>CANNING PLACE</td>
      <td>2.425000e+06</td>
      <td>(51.4995696, -0.1842477)</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>CARLISLE ROAD</td>
      <td>2.200000e+06</td>
      <td>(-36.7091715, 174.7282805)</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>CARLTON GARDENS</td>
      <td>2.483500e+06</td>
      <td>(-37.80194335, 144.9719701710172)</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>CARLYLE COURT</td>
      <td>2.300000e+06</td>
      <td>(32.972700950000004, -97.17339170977195)</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>CHALCOT SQUARE</td>
      <td>2.286679e+06</td>
      <td>(51.5411955, -0.1558168)</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>CHARLES LANE</td>
      <td>2.414000e+06</td>
      <td>(51.533837, -0.170298)</td>
    </tr>
    <tr>
      <th>2561</th>
      <td>CHELSEA CRESCENT</td>
      <td>2.495000e+06</td>
      <td>(34.522443, -85.443891)</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>2.450000e+06</td>
      <td>(51.5292054, -0.1450813)</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>CHEYNE COURT</td>
      <td>2.250000e+06</td>
      <td>(51.599677, 0.5256231)</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>CHEYNE ROW</td>
      <td>2.410000e+06</td>
      <td>(51.4837173, -0.169603)</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>CHISWICK MALL</td>
      <td>2.287500e+06</td>
      <td>(51.4871849, -0.2480168)</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>CITY ROAD</td>
      <td>2.468340e+06</td>
      <td>(51.5296972, -0.0977626)</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>CLARENDON STREET</td>
      <td>2.250000e+06</td>
      <td>(51.36516, 1.1085692)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10924</th>
      <td>RUSSELL GARDENS MEWS</td>
      <td>2.300000e+06</td>
      <td>(51.4998689, -0.2118904)</td>
    </tr>
    <tr>
      <th>11171</th>
      <td>SETTLES STREET</td>
      <td>2.487500e+06</td>
      <td>(51.5153237, -0.0643266)</td>
    </tr>
    <tr>
      <th>11246</th>
      <td>SHELDON AVENUE</td>
      <td>2.349542e+06</td>
      <td>(51.592613, 0.0731449)</td>
    </tr>
    <tr>
      <th>11480</th>
      <td>SOUTH END ROW</td>
      <td>2.470000e+06</td>
      <td>(51.4987463, -0.1890787)</td>
    </tr>
    <tr>
      <th>11558</th>
      <td>SOUTHWOOD LAWN ROAD</td>
      <td>2.350000e+06</td>
      <td>(51.5746267, -0.146238)</td>
    </tr>
    <tr>
      <th>11561</th>
      <td>SOVEREIGN PARK</td>
      <td>2.500000e+06</td>
      <td>(52.6880941, -2.724567251143845)</td>
    </tr>
    <tr>
      <th>11778</th>
      <td>ST MARGARETS CRESCENT</td>
      <td>2.216500e+06</td>
      <td>(17.0975135, -88.6161153)</td>
    </tr>
    <tr>
      <th>11814</th>
      <td>ST OSWALDS PLACE</td>
      <td>2.250000e+06</td>
      <td>(51.4872071, -0.1185341)</td>
    </tr>
    <tr>
      <th>11834</th>
      <td>ST PETERS SQUARE</td>
      <td>2.468730e+06</td>
      <td>(41.9022353, 12.457357310298008)</td>
    </tr>
    <tr>
      <th>11871</th>
      <td>STAFFORD TERRACE</td>
      <td>2.355000e+06</td>
      <td>(51.5009379, -0.1960492)</td>
    </tr>
    <tr>
      <th>12210</th>
      <td>SUTHERLAND PLACE</td>
      <td>2.456000e+06</td>
      <td>(51.5166166, -0.1972762)</td>
    </tr>
    <tr>
      <th>12272</th>
      <td>SYDNEY STREET</td>
      <td>2.240833e+06</td>
      <td>(51.807082, 1.0239602)</td>
    </tr>
    <tr>
      <th>12414</th>
      <td>THAMES BANK</td>
      <td>2.400000e+06</td>
      <td>(52.0856689, -0.2432764)</td>
    </tr>
    <tr>
      <th>12476</th>
      <td>THE HEXAGON</td>
      <td>2.335000e+06</td>
      <td>(51.45388235, -0.9778342181052991)</td>
    </tr>
    <tr>
      <th>12755</th>
      <td>TREDEGAR SQUARE</td>
      <td>2.436666e+06</td>
      <td>(51.5270935, -0.03218423325966063)</td>
    </tr>
    <tr>
      <th>12817</th>
      <td>TRINITY STREET</td>
      <td>2.317500e+06</td>
      <td>(52.6239349, 1.2821379)</td>
    </tr>
    <tr>
      <th>12967</th>
      <td>UPPER HAMPSTEAD WALK</td>
      <td>2.500000e+06</td>
      <td>(51.558467, -0.1774529)</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>WALPOLE GARDENS</td>
      <td>2.303500e+06</td>
      <td>(51.4397435, -0.3412634)</td>
    </tr>
    <tr>
      <th>13231</th>
      <td>WALPOLE STREET</td>
      <td>2.242500e+06</td>
      <td>(52.6261811, 1.2859311)</td>
    </tr>
    <tr>
      <th>13305</th>
      <td>WARWICK SQUARE</td>
      <td>2.432273e+06</td>
      <td>(33.7385632, -117.84650567746482)</td>
    </tr>
    <tr>
      <th>13403</th>
      <td>WELBECK WAY</td>
      <td>2.267000e+06</td>
      <td>(52.5580053, -0.262309)</td>
    </tr>
    <tr>
      <th>13415</th>
      <td>WELLESLEY TERRACE</td>
      <td>2.410000e+06</td>
      <td>(51.5292142, -0.0929327)</td>
    </tr>
    <tr>
      <th>13426</th>
      <td>WELLINGTON STREET</td>
      <td>2.293155e+06</td>
      <td>(45.4234495, -75.6980579)</td>
    </tr>
    <tr>
      <th>13558</th>
      <td>WESTMORELAND PLACE</td>
      <td>2.300000e+06</td>
      <td>(40.7402242, -111.8463235)</td>
    </tr>
    <tr>
      <th>13665</th>
      <td>WHITFIELD STREET</td>
      <td>2.451000e+06</td>
      <td>(51.5229437, -0.1379311)</td>
    </tr>
    <tr>
      <th>13710</th>
      <td>WILFRED STREET</td>
      <td>2.410538e+06</td>
      <td>(51.4988397, -0.1392949)</td>
    </tr>
    <tr>
      <th>13736</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>2.425000e+06</td>
      <td>(51.5431088, -0.0955079)</td>
    </tr>
    <tr>
      <th>13756</th>
      <td>WILSON STREET</td>
      <td>2.257500e+06</td>
      <td>(30.5977973, -81.595757)</td>
    </tr>
    <tr>
      <th>13784</th>
      <td>WINCHENDON ROAD</td>
      <td>2.350000e+06</td>
      <td>(51.4329074, -0.3484547)</td>
    </tr>
    <tr>
      <th>13821</th>
      <td>WINGATE ROAD</td>
      <td>2.206400e+06</td>
      <td>(51.092557, 1.1794554)</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 3 columns</p>
</div>




```python
df_affordable[['Latitude', 'Longitude']] = df_affordable['city_coord'].apply(pd.Series)
```

    /opt/conda/envs/Python36/lib/python3.6/site-packages/pandas/core/frame.py:3391: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self[k1] = value[k2]



```python
df_affordable
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
      <th>Street</th>
      <th>Avg_Price</th>
      <th>city_coord</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2.450000e+06</td>
      <td>(-41.27375755, 173.28939323910353)</td>
      <td>-41.273758</td>
      <td>173.289393</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2.435000e+06</td>
      <td>(51.4803164, -0.1668011)</td>
      <td>51.480316</td>
      <td>-0.166801</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2.250000e+06</td>
      <td>(51.4998899, -0.1891027)</td>
      <td>51.499890</td>
      <td>-0.189103</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2.400000e+06</td>
      <td>(53.7486539, -0.3266704)</td>
      <td>53.748654</td>
      <td>-0.326670</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2.375000e+06</td>
      <td>(51.4773147, -0.239457)</td>
      <td>51.477315</td>
      <td>-0.239457</td>
    </tr>
    <tr>
      <th>981</th>
      <td>BEAUCLERC ROAD</td>
      <td>2.480000e+06</td>
      <td>(51.4995771, -0.2290331)</td>
      <td>51.499577</td>
      <td>-0.229033</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>BELVEDERE DRIVE</td>
      <td>2.340000e+06</td>
      <td>(52.4142089, 1.7244152)</td>
      <td>52.414209</td>
      <td>1.724415</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>BICKENHALL STREET</td>
      <td>2.208500e+06</td>
      <td>(51.5212014, -0.1589082)</td>
      <td>51.521201</td>
      <td>-0.158908</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>2.217000e+06</td>
      <td>(51.4483941, -0.1604676)</td>
      <td>51.448394</td>
      <td>-0.160468</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>BRAMPTON GROVE</td>
      <td>2.456875e+06</td>
      <td>(51.5899607, -0.3185249)</td>
      <td>51.589961</td>
      <td>-0.318525</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>BRIARDALE GARDENS</td>
      <td>2.397132e+06</td>
      <td>(51.5601748, -0.1954305)</td>
      <td>51.560175</td>
      <td>-0.195431</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>BROOKWAY</td>
      <td>2.400000e+06</td>
      <td>(45.432184899999996, -122.80281166115779)</td>
      <td>45.432185</td>
      <td>-122.802812</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>BURBAGE ROAD</td>
      <td>2.445000e+06</td>
      <td>(51.4482603, -0.0885073)</td>
      <td>51.448260</td>
      <td>-0.088507</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>BURY WALK</td>
      <td>2.492500e+06</td>
      <td>(52.1455294, -0.4235933)</td>
      <td>52.145529</td>
      <td>-0.423593</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>CALLCOTT STREET</td>
      <td>2.375000e+06</td>
      <td>(51.5083499, -0.1983276)</td>
      <td>51.508350</td>
      <td>-0.198328</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>2.379653e+06</td>
      <td>(51.50141, -0.1951157)</td>
      <td>51.501410</td>
      <td>-0.195116</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>CAMPION ROAD</td>
      <td>2.461000e+06</td>
      <td>(52.6813749, 0.9654713)</td>
      <td>52.681375</td>
      <td>0.965471</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>CANNING PLACE</td>
      <td>2.425000e+06</td>
      <td>(51.4995696, -0.1842477)</td>
      <td>51.499570</td>
      <td>-0.184248</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>CARLISLE ROAD</td>
      <td>2.200000e+06</td>
      <td>(-36.7091715, 174.7282805)</td>
      <td>-36.709171</td>
      <td>174.728281</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>CARLTON GARDENS</td>
      <td>2.483500e+06</td>
      <td>(-37.80194335, 144.9719701710172)</td>
      <td>-37.801943</td>
      <td>144.971970</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>CARLYLE COURT</td>
      <td>2.300000e+06</td>
      <td>(32.972700950000004, -97.17339170977195)</td>
      <td>32.972701</td>
      <td>-97.173392</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>CHALCOT SQUARE</td>
      <td>2.286679e+06</td>
      <td>(51.5411955, -0.1558168)</td>
      <td>51.541196</td>
      <td>-0.155817</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>CHARLES LANE</td>
      <td>2.414000e+06</td>
      <td>(51.533837, -0.170298)</td>
      <td>51.533837</td>
      <td>-0.170298</td>
    </tr>
    <tr>
      <th>2561</th>
      <td>CHELSEA CRESCENT</td>
      <td>2.495000e+06</td>
      <td>(34.522443, -85.443891)</td>
      <td>34.522443</td>
      <td>-85.443891</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>2.450000e+06</td>
      <td>(51.5292054, -0.1450813)</td>
      <td>51.529205</td>
      <td>-0.145081</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>CHEYNE COURT</td>
      <td>2.250000e+06</td>
      <td>(51.599677, 0.5256231)</td>
      <td>51.599677</td>
      <td>0.525623</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>CHEYNE ROW</td>
      <td>2.410000e+06</td>
      <td>(51.4837173, -0.169603)</td>
      <td>51.483717</td>
      <td>-0.169603</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>CHISWICK MALL</td>
      <td>2.287500e+06</td>
      <td>(51.4871849, -0.2480168)</td>
      <td>51.487185</td>
      <td>-0.248017</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>CITY ROAD</td>
      <td>2.468340e+06</td>
      <td>(51.5296972, -0.0977626)</td>
      <td>51.529697</td>
      <td>-0.097763</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>CLARENDON STREET</td>
      <td>2.250000e+06</td>
      <td>(51.36516, 1.1085692)</td>
      <td>51.365160</td>
      <td>1.108569</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10924</th>
      <td>RUSSELL GARDENS MEWS</td>
      <td>2.300000e+06</td>
      <td>(51.4998689, -0.2118904)</td>
      <td>51.499869</td>
      <td>-0.211890</td>
    </tr>
    <tr>
      <th>11171</th>
      <td>SETTLES STREET</td>
      <td>2.487500e+06</td>
      <td>(51.5153237, -0.0643266)</td>
      <td>51.515324</td>
      <td>-0.064327</td>
    </tr>
    <tr>
      <th>11246</th>
      <td>SHELDON AVENUE</td>
      <td>2.349542e+06</td>
      <td>(51.592613, 0.0731449)</td>
      <td>51.592613</td>
      <td>0.073145</td>
    </tr>
    <tr>
      <th>11480</th>
      <td>SOUTH END ROW</td>
      <td>2.470000e+06</td>
      <td>(51.4987463, -0.1890787)</td>
      <td>51.498746</td>
      <td>-0.189079</td>
    </tr>
    <tr>
      <th>11558</th>
      <td>SOUTHWOOD LAWN ROAD</td>
      <td>2.350000e+06</td>
      <td>(51.5746267, -0.146238)</td>
      <td>51.574627</td>
      <td>-0.146238</td>
    </tr>
    <tr>
      <th>11561</th>
      <td>SOVEREIGN PARK</td>
      <td>2.500000e+06</td>
      <td>(52.6880941, -2.724567251143845)</td>
      <td>52.688094</td>
      <td>-2.724567</td>
    </tr>
    <tr>
      <th>11778</th>
      <td>ST MARGARETS CRESCENT</td>
      <td>2.216500e+06</td>
      <td>(17.0975135, -88.6161153)</td>
      <td>17.097514</td>
      <td>-88.616115</td>
    </tr>
    <tr>
      <th>11814</th>
      <td>ST OSWALDS PLACE</td>
      <td>2.250000e+06</td>
      <td>(51.4872071, -0.1185341)</td>
      <td>51.487207</td>
      <td>-0.118534</td>
    </tr>
    <tr>
      <th>11834</th>
      <td>ST PETERS SQUARE</td>
      <td>2.468730e+06</td>
      <td>(41.9022353, 12.457357310298008)</td>
      <td>41.902235</td>
      <td>12.457357</td>
    </tr>
    <tr>
      <th>11871</th>
      <td>STAFFORD TERRACE</td>
      <td>2.355000e+06</td>
      <td>(51.5009379, -0.1960492)</td>
      <td>51.500938</td>
      <td>-0.196049</td>
    </tr>
    <tr>
      <th>12210</th>
      <td>SUTHERLAND PLACE</td>
      <td>2.456000e+06</td>
      <td>(51.5166166, -0.1972762)</td>
      <td>51.516617</td>
      <td>-0.197276</td>
    </tr>
    <tr>
      <th>12272</th>
      <td>SYDNEY STREET</td>
      <td>2.240833e+06</td>
      <td>(51.807082, 1.0239602)</td>
      <td>51.807082</td>
      <td>1.023960</td>
    </tr>
    <tr>
      <th>12414</th>
      <td>THAMES BANK</td>
      <td>2.400000e+06</td>
      <td>(52.0856689, -0.2432764)</td>
      <td>52.085669</td>
      <td>-0.243276</td>
    </tr>
    <tr>
      <th>12476</th>
      <td>THE HEXAGON</td>
      <td>2.335000e+06</td>
      <td>(51.45388235, -0.9778342181052991)</td>
      <td>51.453882</td>
      <td>-0.977834</td>
    </tr>
    <tr>
      <th>12755</th>
      <td>TREDEGAR SQUARE</td>
      <td>2.436666e+06</td>
      <td>(51.5270935, -0.03218423325966063)</td>
      <td>51.527093</td>
      <td>-0.032184</td>
    </tr>
    <tr>
      <th>12817</th>
      <td>TRINITY STREET</td>
      <td>2.317500e+06</td>
      <td>(52.6239349, 1.2821379)</td>
      <td>52.623935</td>
      <td>1.282138</td>
    </tr>
    <tr>
      <th>12967</th>
      <td>UPPER HAMPSTEAD WALK</td>
      <td>2.500000e+06</td>
      <td>(51.558467, -0.1774529)</td>
      <td>51.558467</td>
      <td>-0.177453</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>WALPOLE GARDENS</td>
      <td>2.303500e+06</td>
      <td>(51.4397435, -0.3412634)</td>
      <td>51.439743</td>
      <td>-0.341263</td>
    </tr>
    <tr>
      <th>13231</th>
      <td>WALPOLE STREET</td>
      <td>2.242500e+06</td>
      <td>(52.6261811, 1.2859311)</td>
      <td>52.626181</td>
      <td>1.285931</td>
    </tr>
    <tr>
      <th>13305</th>
      <td>WARWICK SQUARE</td>
      <td>2.432273e+06</td>
      <td>(33.7385632, -117.84650567746482)</td>
      <td>33.738563</td>
      <td>-117.846506</td>
    </tr>
    <tr>
      <th>13403</th>
      <td>WELBECK WAY</td>
      <td>2.267000e+06</td>
      <td>(52.5580053, -0.262309)</td>
      <td>52.558005</td>
      <td>-0.262309</td>
    </tr>
    <tr>
      <th>13415</th>
      <td>WELLESLEY TERRACE</td>
      <td>2.410000e+06</td>
      <td>(51.5292142, -0.0929327)</td>
      <td>51.529214</td>
      <td>-0.092933</td>
    </tr>
    <tr>
      <th>13426</th>
      <td>WELLINGTON STREET</td>
      <td>2.293155e+06</td>
      <td>(45.4234495, -75.6980579)</td>
      <td>45.423449</td>
      <td>-75.698058</td>
    </tr>
    <tr>
      <th>13558</th>
      <td>WESTMORELAND PLACE</td>
      <td>2.300000e+06</td>
      <td>(40.7402242, -111.8463235)</td>
      <td>40.740224</td>
      <td>-111.846323</td>
    </tr>
    <tr>
      <th>13665</th>
      <td>WHITFIELD STREET</td>
      <td>2.451000e+06</td>
      <td>(51.5229437, -0.1379311)</td>
      <td>51.522944</td>
      <td>-0.137931</td>
    </tr>
    <tr>
      <th>13710</th>
      <td>WILFRED STREET</td>
      <td>2.410538e+06</td>
      <td>(51.4988397, -0.1392949)</td>
      <td>51.498840</td>
      <td>-0.139295</td>
    </tr>
    <tr>
      <th>13736</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>2.425000e+06</td>
      <td>(51.5431088, -0.0955079)</td>
      <td>51.543109</td>
      <td>-0.095508</td>
    </tr>
    <tr>
      <th>13756</th>
      <td>WILSON STREET</td>
      <td>2.257500e+06</td>
      <td>(30.5977973, -81.595757)</td>
      <td>30.597797</td>
      <td>-81.595757</td>
    </tr>
    <tr>
      <th>13784</th>
      <td>WINCHENDON ROAD</td>
      <td>2.350000e+06</td>
      <td>(51.4329074, -0.3484547)</td>
      <td>51.432907</td>
      <td>-0.348455</td>
    </tr>
    <tr>
      <th>13821</th>
      <td>WINGATE ROAD</td>
      <td>2.206400e+06</td>
      <td>(51.092557, 1.1794554)</td>
      <td>51.092557</td>
      <td>1.179455</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 5 columns</p>
</div>




```python
df1 = df_affordable.drop(columns=['city_coord'])
```


```python
df1
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
      <th>Street</th>
      <th>Avg_Price</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2.450000e+06</td>
      <td>-41.273758</td>
      <td>173.289393</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2.435000e+06</td>
      <td>51.480316</td>
      <td>-0.166801</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2.250000e+06</td>
      <td>51.499890</td>
      <td>-0.189103</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2.400000e+06</td>
      <td>53.748654</td>
      <td>-0.326670</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2.375000e+06</td>
      <td>51.477315</td>
      <td>-0.239457</td>
    </tr>
    <tr>
      <th>981</th>
      <td>BEAUCLERC ROAD</td>
      <td>2.480000e+06</td>
      <td>51.499577</td>
      <td>-0.229033</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>BELVEDERE DRIVE</td>
      <td>2.340000e+06</td>
      <td>52.414209</td>
      <td>1.724415</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>BICKENHALL STREET</td>
      <td>2.208500e+06</td>
      <td>51.521201</td>
      <td>-0.158908</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>2.217000e+06</td>
      <td>51.448394</td>
      <td>-0.160468</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>BRAMPTON GROVE</td>
      <td>2.456875e+06</td>
      <td>51.589961</td>
      <td>-0.318525</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>BRIARDALE GARDENS</td>
      <td>2.397132e+06</td>
      <td>51.560175</td>
      <td>-0.195431</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>BROOKWAY</td>
      <td>2.400000e+06</td>
      <td>45.432185</td>
      <td>-122.802812</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>BURBAGE ROAD</td>
      <td>2.445000e+06</td>
      <td>51.448260</td>
      <td>-0.088507</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>BURY WALK</td>
      <td>2.492500e+06</td>
      <td>52.145529</td>
      <td>-0.423593</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>CALLCOTT STREET</td>
      <td>2.375000e+06</td>
      <td>51.508350</td>
      <td>-0.198328</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>2.379653e+06</td>
      <td>51.501410</td>
      <td>-0.195116</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>CAMPION ROAD</td>
      <td>2.461000e+06</td>
      <td>52.681375</td>
      <td>0.965471</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>CANNING PLACE</td>
      <td>2.425000e+06</td>
      <td>51.499570</td>
      <td>-0.184248</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>CARLISLE ROAD</td>
      <td>2.200000e+06</td>
      <td>-36.709171</td>
      <td>174.728281</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>CARLTON GARDENS</td>
      <td>2.483500e+06</td>
      <td>-37.801943</td>
      <td>144.971970</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>CARLYLE COURT</td>
      <td>2.300000e+06</td>
      <td>32.972701</td>
      <td>-97.173392</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>CHALCOT SQUARE</td>
      <td>2.286679e+06</td>
      <td>51.541196</td>
      <td>-0.155817</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>CHARLES LANE</td>
      <td>2.414000e+06</td>
      <td>51.533837</td>
      <td>-0.170298</td>
    </tr>
    <tr>
      <th>2561</th>
      <td>CHELSEA CRESCENT</td>
      <td>2.495000e+06</td>
      <td>34.522443</td>
      <td>-85.443891</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>2.450000e+06</td>
      <td>51.529205</td>
      <td>-0.145081</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>CHEYNE COURT</td>
      <td>2.250000e+06</td>
      <td>51.599677</td>
      <td>0.525623</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>CHEYNE ROW</td>
      <td>2.410000e+06</td>
      <td>51.483717</td>
      <td>-0.169603</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>CHISWICK MALL</td>
      <td>2.287500e+06</td>
      <td>51.487185</td>
      <td>-0.248017</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>CITY ROAD</td>
      <td>2.468340e+06</td>
      <td>51.529697</td>
      <td>-0.097763</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>CLARENDON STREET</td>
      <td>2.250000e+06</td>
      <td>51.365160</td>
      <td>1.108569</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>10924</th>
      <td>RUSSELL GARDENS MEWS</td>
      <td>2.300000e+06</td>
      <td>51.499869</td>
      <td>-0.211890</td>
    </tr>
    <tr>
      <th>11171</th>
      <td>SETTLES STREET</td>
      <td>2.487500e+06</td>
      <td>51.515324</td>
      <td>-0.064327</td>
    </tr>
    <tr>
      <th>11246</th>
      <td>SHELDON AVENUE</td>
      <td>2.349542e+06</td>
      <td>51.592613</td>
      <td>0.073145</td>
    </tr>
    <tr>
      <th>11480</th>
      <td>SOUTH END ROW</td>
      <td>2.470000e+06</td>
      <td>51.498746</td>
      <td>-0.189079</td>
    </tr>
    <tr>
      <th>11558</th>
      <td>SOUTHWOOD LAWN ROAD</td>
      <td>2.350000e+06</td>
      <td>51.574627</td>
      <td>-0.146238</td>
    </tr>
    <tr>
      <th>11561</th>
      <td>SOVEREIGN PARK</td>
      <td>2.500000e+06</td>
      <td>52.688094</td>
      <td>-2.724567</td>
    </tr>
    <tr>
      <th>11778</th>
      <td>ST MARGARETS CRESCENT</td>
      <td>2.216500e+06</td>
      <td>17.097514</td>
      <td>-88.616115</td>
    </tr>
    <tr>
      <th>11814</th>
      <td>ST OSWALDS PLACE</td>
      <td>2.250000e+06</td>
      <td>51.487207</td>
      <td>-0.118534</td>
    </tr>
    <tr>
      <th>11834</th>
      <td>ST PETERS SQUARE</td>
      <td>2.468730e+06</td>
      <td>41.902235</td>
      <td>12.457357</td>
    </tr>
    <tr>
      <th>11871</th>
      <td>STAFFORD TERRACE</td>
      <td>2.355000e+06</td>
      <td>51.500938</td>
      <td>-0.196049</td>
    </tr>
    <tr>
      <th>12210</th>
      <td>SUTHERLAND PLACE</td>
      <td>2.456000e+06</td>
      <td>51.516617</td>
      <td>-0.197276</td>
    </tr>
    <tr>
      <th>12272</th>
      <td>SYDNEY STREET</td>
      <td>2.240833e+06</td>
      <td>51.807082</td>
      <td>1.023960</td>
    </tr>
    <tr>
      <th>12414</th>
      <td>THAMES BANK</td>
      <td>2.400000e+06</td>
      <td>52.085669</td>
      <td>-0.243276</td>
    </tr>
    <tr>
      <th>12476</th>
      <td>THE HEXAGON</td>
      <td>2.335000e+06</td>
      <td>51.453882</td>
      <td>-0.977834</td>
    </tr>
    <tr>
      <th>12755</th>
      <td>TREDEGAR SQUARE</td>
      <td>2.436666e+06</td>
      <td>51.527093</td>
      <td>-0.032184</td>
    </tr>
    <tr>
      <th>12817</th>
      <td>TRINITY STREET</td>
      <td>2.317500e+06</td>
      <td>52.623935</td>
      <td>1.282138</td>
    </tr>
    <tr>
      <th>12967</th>
      <td>UPPER HAMPSTEAD WALK</td>
      <td>2.500000e+06</td>
      <td>51.558467</td>
      <td>-0.177453</td>
    </tr>
    <tr>
      <th>13229</th>
      <td>WALPOLE GARDENS</td>
      <td>2.303500e+06</td>
      <td>51.439743</td>
      <td>-0.341263</td>
    </tr>
    <tr>
      <th>13231</th>
      <td>WALPOLE STREET</td>
      <td>2.242500e+06</td>
      <td>52.626181</td>
      <td>1.285931</td>
    </tr>
    <tr>
      <th>13305</th>
      <td>WARWICK SQUARE</td>
      <td>2.432273e+06</td>
      <td>33.738563</td>
      <td>-117.846506</td>
    </tr>
    <tr>
      <th>13403</th>
      <td>WELBECK WAY</td>
      <td>2.267000e+06</td>
      <td>52.558005</td>
      <td>-0.262309</td>
    </tr>
    <tr>
      <th>13415</th>
      <td>WELLESLEY TERRACE</td>
      <td>2.410000e+06</td>
      <td>51.529214</td>
      <td>-0.092933</td>
    </tr>
    <tr>
      <th>13426</th>
      <td>WELLINGTON STREET</td>
      <td>2.293155e+06</td>
      <td>45.423449</td>
      <td>-75.698058</td>
    </tr>
    <tr>
      <th>13558</th>
      <td>WESTMORELAND PLACE</td>
      <td>2.300000e+06</td>
      <td>40.740224</td>
      <td>-111.846323</td>
    </tr>
    <tr>
      <th>13665</th>
      <td>WHITFIELD STREET</td>
      <td>2.451000e+06</td>
      <td>51.522944</td>
      <td>-0.137931</td>
    </tr>
    <tr>
      <th>13710</th>
      <td>WILFRED STREET</td>
      <td>2.410538e+06</td>
      <td>51.498840</td>
      <td>-0.139295</td>
    </tr>
    <tr>
      <th>13736</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>2.425000e+06</td>
      <td>51.543109</td>
      <td>-0.095508</td>
    </tr>
    <tr>
      <th>13756</th>
      <td>WILSON STREET</td>
      <td>2.257500e+06</td>
      <td>30.597797</td>
      <td>-81.595757</td>
    </tr>
    <tr>
      <th>13784</th>
      <td>WINCHENDON ROAD</td>
      <td>2.350000e+06</td>
      <td>51.432907</td>
      <td>-0.348455</td>
    </tr>
    <tr>
      <th>13821</th>
      <td>WINGATE ROAD</td>
      <td>2.206400e+06</td>
      <td>51.092557</td>
      <td>1.179455</td>
    </tr>
  </tbody>
</table>
<p>162 rows × 4 columns</p>
</div>




```python
address = 'London, UK'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of London City are {}, {}.'.format(latitude, longitude))
```

    /opt/conda/envs/Python36/lib/python3.6/site-packages/ipykernel/__main__.py:3: DeprecationWarning: Using Nominatim with the default "geopy/1.20.0" `user_agent` is strongly discouraged, as it violates Nominatim's ToS https://operations.osmfoundation.org/policies/nominatim/ and may possibly cause 403 and 429 HTTP errors. Please specify a custom `user_agent` with `Nominatim(user_agent="my-application")` or by overriding the default `user_agent`: `geopy.geocoders.options.default_user_agent = "my-application"`. In geopy 2.0 this will become an exception.
      app.launch_new_instance()


    The geograpical coordinate of London City are 51.5073219, -0.1276474.



```python
# create map of London using latitude and longitude values
map_london = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, price, street in zip(df1['Latitude'], df1['Longitude'], df1['Avg_Price'], df1['Street']):
    label = '{}, {}'.format(street, price)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_london)  
    
map_london
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTEuNTA3MzIxOSwtMC4xMjc2NDc0XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl9iNzAxZGYxZGIxMDQ0NWNkOTA5NzUzMjVmYjVkNjlhMCA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjMzMDljZTE0M2JhNDgwYTg2ZDNhZGRhYzI5MDhhMjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstNDEuMjczNzU3NTUsMTczLjI4OTM5MzIzOTEwMzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg2N2NjNzJiNjIwZTQ0Nzk5MTM5OGVkNmRmYmFmMjY2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyODZlMTJiNzhlZDRlODQ5MzIxYmNhNzE0NDFkZDQwID0gJCgnPGRpdiBpZD0iaHRtbF9hMjg2ZTEyYjc4ZWQ0ZTg0OTMyMWJjYTcxNDQxZGQ0MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QUxCSU9OIFNRVUFSRSwgMjQ1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84NjdjYzcyYjYyMGU0NDc5OTEzOThlZDZkZmJhZjI2Ni5zZXRDb250ZW50KGh0bWxfYTI4NmUxMmI3OGVkNGU4NDkzMjFiY2E3MTQ0MWRkNDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjMzMDljZTE0M2JhNDgwYTg2ZDNhZGRhYzI5MDhhMjMuYmluZFBvcHVwKHBvcHVwXzg2N2NjNzJiNjIwZTQ0Nzk5MTM5OGVkNmRmYmFmMjY2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M2MzQ5MDMxZGY3NTQwN2NiYjI3ZDc2NTU1ODdkODI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgwMzE2NCwtMC4xNjY4MDExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk0NWU2MTQ5YzZjNDRjODFhMjY5MTQ0YzVjMzU1MWMwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UxMzMxZmI1MDUwYzRlODY5MmEyYjA1ODlkZjgyNmVlID0gJCgnPGRpdiBpZD0iaHRtbF9lMTMzMWZiNTA1MGM0ZTg2OTJhMmIwNTg5ZGY4MjZlZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QU5IQUxUIFJPQUQsIDI0MzUwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTQ1ZTYxNDljNmM0NGM4MWEyNjkxNDRjNWMzNTUxYzAuc2V0Q29udGVudChodG1sX2UxMzMxZmI1MDUwYzRlODY5MmEyYjA1ODlkZjgyNmVlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M2MzQ5MDMxZGY3NTQwN2NiYjI3ZDc2NTU1ODdkODI1LmJpbmRQb3B1cChwb3B1cF85NDVlNjE0OWM2YzQ0YzgxYTI2OTE0NGM1YzM1NTFjMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMTBkMDY0MTk4YTU0YjBiYjlkYTM1ZGRjOWNlNzBjZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5OTg4OTksLTAuMTg5MTAyN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZTkxODVjMzcxNjM0ODdlODg3MDZmMGFjMTZmNGRkZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYmE3MjIzNGM3Y2E0ZmY4YTQ0NDQ1MjUxZTVmMWRmMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMWJhNzIyMzRjN2NhNGZmOGE0NDQ0NTI1MWU1ZjFkZjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFOU0RFTEwgVEVSUkFDRSwgMjI1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZTkxODVjMzcxNjM0ODdlODg3MDZmMGFjMTZmNGRkZS5zZXRDb250ZW50KGh0bWxfMWJhNzIyMzRjN2NhNGZmOGE0NDQ0NTI1MWU1ZjFkZjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTEwZDA2NDE5OGE1NGIwYmI5ZGEzNWRkYzljZTcwY2QuYmluZFBvcHVwKHBvcHVwX2FlOTE4NWMzNzE2MzQ4N2U4ODcwNmYwYWMxNmY0ZGRlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhiNjc4NDY2MzVmNzRlZGU4NDVmNDk4ZmY5NmZmMTAxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuNzQ4NjUzOSwtMC4zMjY2NzA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NiYTUxOTlhYjQ2MTRiMWI5MWM0ZDI4YWNlYmQyZDA1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZjOTJkNDU4NWFlNzRjYTk4OTZmMTdmODhlNTJkNmU0ID0gJCgnPGRpdiBpZD0iaHRtbF82YzkyZDQ1ODVhZTc0Y2E5ODk2ZjE3Zjg4ZTUyZDZlNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QVBQTEVHQVJUSCBST0FELCAyNDAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NiYTUxOTlhYjQ2MTRiMWI5MWM0ZDI4YWNlYmQyZDA1LnNldENvbnRlbnQoaHRtbF82YzkyZDQ1ODVhZTc0Y2E5ODk2ZjE3Zjg4ZTUyZDZlNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjY3ODQ2NjM1Zjc0ZWRlODQ1ZjQ5OGZmOTZmZjEwMS5iaW5kUG9wdXAocG9wdXBfY2JhNTE5OWFiNDYxNGIxYjkxYzRkMjhhY2ViZDJkMDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjVjZjJjZmZmY2I3NGFmN2FhYjU3ZjgyMGUyMzk5MTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NzczMTQ3LC0wLjIzOTQ1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNmY5NmU5ZWFkNjA0NTMxOTY5MjRiYTQwMWYxZmVlYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NjZkYzQzZTU4ZjY0ZmJmYWFhYzk5NmVjMTdhOThlOCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDY2ZGM0M2U1OGY2NGZiZmFhYWM5OTZlYzE3YTk4ZTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJBUk9OU01FQUQgUk9BRCwgMjM3NTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNmY5NmU5ZWFkNjA0NTMxOTY5MjRiYTQwMWYxZmVlYy5zZXRDb250ZW50KGh0bWxfNDY2ZGM0M2U1OGY2NGZiZmFhYWM5OTZlYzE3YTk4ZTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjVjZjJjZmZmY2I3NGFmN2FhYjU3ZjgyMGUyMzk5MTAuYmluZFBvcHVwKHBvcHVwXzI2Zjk2ZTllYWQ2MDQ1MzE5NjkyNGJhNDAxZjFmZWVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3NmY1YTEwY2E3NTQ1NWI4NjdhZjg5MTI1NTMyY2ZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk5NTc3MSwtMC4yMjkwMzMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzEwODEyODJjZGM3YTQ0ZjY4YjI5MjRjNWVhZGM4ODZjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJiMWVlODkxYjdiMzRlMzNhYzZlNzk0NjQ3MTg0ZDY2ID0gJCgnPGRpdiBpZD0iaHRtbF8yYjFlZTg5MWI3YjM0ZTMzYWM2ZTc5NDY0NzE4NGQ2NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QkVBVUNMRVJDIFJPQUQsIDI0ODAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTA4MTI4MmNkYzdhNDRmNjhiMjkyNGM1ZWFkYzg4NmMuc2V0Q29udGVudChodG1sXzJiMWVlODkxYjdiMzRlMzNhYzZlNzk0NjQ3MTg0ZDY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ3NmY1YTEwY2E3NTQ1NWI4NjdhZjg5MTI1NTMyY2ZiLmJpbmRQb3B1cChwb3B1cF8xMDgxMjgyY2RjN2E0NGY2OGIyOTI0YzVlYWRjODg2Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYjNhYmQ4MjQ5YWM0MmVjOTBhYWU1MDQzY2FmYTc1OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjQxNDIwODksMS43MjQ0MTUyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyNDE3YjM2ODVhODRkZWViMDk5ZDBkZmJjNjc4NDUwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyMWYxY2FmOWU1NjRlOWM4NDc3YTM5OGZkNzVmMjE4ID0gJCgnPGRpdiBpZD0iaHRtbF9hMjFmMWNhZjllNTY0ZTljODQ3N2EzOThmZDc1ZjIxOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QkVMVkVERVJFIERSSVZFLCAyMzQwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYyNDE3YjM2ODVhODRkZWViMDk5ZDBkZmJjNjc4NDUwLnNldENvbnRlbnQoaHRtbF9hMjFmMWNhZjllNTY0ZTljODQ3N2EzOThmZDc1ZjIxOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYjNhYmQ4MjQ5YWM0MmVjOTBhYWU1MDQzY2FmYTc1OC5iaW5kUG9wdXAocG9wdXBfNjI0MTdiMzY4NWE4NGRlZWIwOTlkMGRmYmM2Nzg0NTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODIwODBkNzFiMmEwNDRhZjljYjNmYzY5Zjk2ODY5MzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjEyMDE0LC0wLjE1ODkwODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTJkMzljZjQzODJkNDg5YThkZWRhZGE3NDM0ZDA2ZmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjgzMTkwNjIyZGY5NDE1ZmFlMTAwMjMwZGZkZDU2ZTcgPSAkKCc8ZGl2IGlkPSJodG1sX2I4MzE5MDYyMmRmOTQxNWZhZTEwMDIzMGRmZGQ1NmU3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CSUNLRU5IQUxMIFNUUkVFVCwgMjIwODUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMmQzOWNmNDM4MmQ0ODlhOGRlZGFkYTc0MzRkMDZmZC5zZXRDb250ZW50KGh0bWxfYjgzMTkwNjIyZGY5NDE1ZmFlMTAwMjMwZGZkZDU2ZTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODIwODBkNzFiMmEwNDRhZjljYjNmYzY5Zjk2ODY5MzQuYmluZFBvcHVwKHBvcHVwX2EyZDM5Y2Y0MzgyZDQ4OWE4ZGVkYWRhNzQzNGQwNmZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JkOTVmMThhMmMxZjRhMzU5Y2ZhNWNiN2U3ZmRkNmNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDQ4Mzk0MSwtMC4xNjA0Njc2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MwYzViNjIzYTJjNDQwNjViNjY2NDc0M2FlNzIyMDE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJlMDkzMDQ4OGYxZTRiNDFiZmNhYTM4ODQwYTdiYjAxID0gJCgnPGRpdiBpZD0iaHRtbF8yZTA5MzA0ODhmMWU0YjQxYmZjYWEzODg0MGE3YmIwMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QklSQ0hMQU5EUyBBVkVOVUUsIDIyMTcwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzBjNWI2MjNhMmM0NDA2NWI2NjY0NzQzYWU3MjIwMTQuc2V0Q29udGVudChodG1sXzJlMDkzMDQ4OGYxZTRiNDFiZmNhYTM4ODQwYTdiYjAxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkOTVmMThhMmMxZjRhMzU5Y2ZhNWNiN2U3ZmRkNmNmLmJpbmRQb3B1cChwb3B1cF9jMGM1YjYyM2EyYzQ0MDY1YjY2NjQ3NDNhZTcyMjAxNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMmI5NmRlZTY2M2Y0NmFiOTM4OWUwMTllZGM2ZGU0OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU4OTk2MDcsLTAuMzE4NTI0OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80M2Q3MTlhMWIxNGE0ZmI1YThiNmE1MWUxMGZmNWQ5NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NTdkYTJhZDJlNmY0YjFkOGI4Njk4ZjY2NTllMzFjOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjU3ZGEyYWQyZTZmNGIxZDhiODY5OGY2NjU5ZTMxYzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJSQU1QVE9OIEdST1ZFLCAyNDU2ODc1LjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQzZDcxOWExYjE0YTRmYjVhOGI2YTUxZTEwZmY1ZDk1LnNldENvbnRlbnQoaHRtbF82NTdkYTJhZDJlNmY0YjFkOGI4Njk4ZjY2NTllMzFjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMmI5NmRlZTY2M2Y0NmFiOTM4OWUwMTllZGM2ZGU0OS5iaW5kUG9wdXAocG9wdXBfNDNkNzE5YTFiMTRhNGZiNWE4YjZhNTFlMTBmZjVkOTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWEzYTg3MDVjYzMyNDVhMzg4OWY2NDVlZWY0ZmJhMzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41NjAxNzQ4LC0wLjE5NTQzMDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjEzM2FhNjVjNTczNDc5NDg5Y2VmMTc3ZjdmZjRhMDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTg0NmJkYThkODk1NDA1OGJhZWE4YjMwODE2MmE4MzIgPSAkKCc8ZGl2IGlkPSJodG1sXzE4NDZiZGE4ZDg5NTQwNThiYWVhOGIzMDgxNjJhODMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CUklBUkRBTEUgR0FSREVOUywgMjM5NzEzMi4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMTMzYWE2NWM1NzM0Nzk0ODljZWYxNzdmN2ZmNGEwNi5zZXRDb250ZW50KGh0bWxfMTg0NmJkYThkODk1NDA1OGJhZWE4YjMwODE2MmE4MzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWEzYTg3MDVjYzMyNDVhMzg4OWY2NDVlZWY0ZmJhMzUuYmluZFBvcHVwKHBvcHVwX2IxMzNhYTY1YzU3MzQ3OTQ4OWNlZjE3N2Y3ZmY0YTA2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE0M2VjM2E1ZDUxNjQwYjM4NzI0ZDc0MjE2YWY0ZjE0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNDMyMTg0ODk5OTk5OTk2LC0xMjIuODAyODExNjYxMTU3NzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWFiOWZiZDQ3ZmYyNGNiOTkyMjRkMmJiYjFiMjFhYzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjM0YjM5NzY5NDg5NGJmNWFlNzQwZmRmOTMzNmJkNDIgPSAkKCc8ZGl2IGlkPSJodG1sXzIzNGIzOTc2OTQ4OTRiZjVhZTc0MGZkZjkzMzZiZDQyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CUk9PS1dBWSwgMjQwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYWI5ZmJkNDdmZjI0Y2I5OTIyNGQyYmJiMWIyMWFjMS5zZXRDb250ZW50KGh0bWxfMjM0YjM5NzY5NDg5NGJmNWFlNzQwZmRmOTMzNmJkNDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTQzZWMzYTVkNTE2NDBiMzg3MjRkNzQyMTZhZjRmMTQuYmluZFBvcHVwKHBvcHVwX2VhYjlmYmQ0N2ZmMjRjYjk5MjI0ZDJiYmIxYjIxYWMxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QxMWNiMTE4OGZkYTQwZGJiZmVlMGUxMmUxNTIwNmRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDQ4MjYwMywtMC4wODg1MDczXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhiY2VmNGNiYzVhMzRjNDc5OGVmNWNlZTFlMDU0ZTk5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U1M2Q1MzUzYzE3NDQyZjBiOWY4MTc3NGY0ZWNhZGJmID0gJCgnPGRpdiBpZD0iaHRtbF9lNTNkNTM1M2MxNzQ0MmYwYjlmODE3NzRmNGVjYWRiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QlVSQkFHRSBST0FELCAyNDQ1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhiY2VmNGNiYzVhMzRjNDc5OGVmNWNlZTFlMDU0ZTk5LnNldENvbnRlbnQoaHRtbF9lNTNkNTM1M2MxNzQ0MmYwYjlmODE3NzRmNGVjYWRiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMTFjYjExODhmZGE0MGRiYmZlZTBlMTJlMTUyMDZkYi5iaW5kUG9wdXAocG9wdXBfOGJjZWY0Y2JjNWEzNGM0Nzk4ZWY1Y2VlMWUwNTRlOTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzc3NjEzMzE1Mzc0NGZhNTlkYzc3ZWYxYTljZGVkYjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi4xNDU1Mjk0LC0wLjQyMzU5MzNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjJiMjc5Yjk2MDlmNDBiNzg1ZjY0NGY0NGM2MjU3MzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzE3YTkxMzIyMDRlNDZhMjk0ZTAxYjEzOGIyMThjOTIgPSAkKCc8ZGl2IGlkPSJodG1sX2MxN2E5MTMyMjA0ZTQ2YTI5NGUwMWIxMzhiMjE4YzkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CVVJZIFdBTEssIDI0OTI1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjJiMjc5Yjk2MDlmNDBiNzg1ZjY0NGY0NGM2MjU3MzQuc2V0Q29udGVudChodG1sX2MxN2E5MTMyMjA0ZTQ2YTI5NGUwMWIxMzhiMjE4YzkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM3NzYxMzMxNTM3NDRmYTU5ZGM3N2VmMWE5Y2RlZGIyLmJpbmRQb3B1cChwb3B1cF9mMmIyNzliOTYwOWY0MGI3ODVmNjQ0ZjQ0YzYyNTczNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMDI4MDE1ZjFhMjE0ZjdhYThkYTllNTU4MmMxNzhmZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUwODM0OTksLTAuMTk4MzI3Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MDc4NWQwYzFmN2E0MTgzODNmN2U5MjM0YjkxYjJiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNDQ3NDMwMjVlOTk0NmFlOGJkOGY0M2JkYjJkOWZjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMDQ0NzQzMDI1ZTk5NDZhZThiZDhmNDNiZGIyZDlmY2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBTExDT1RUIFNUUkVFVCwgMjM3NTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MDc4NWQwYzFmN2E0MTgzODNmN2U5MjM0YjkxYjJiYy5zZXRDb250ZW50KGh0bWxfMDQ0NzQzMDI1ZTk5NDZhZThiZDhmNDNiZGIyZDlmY2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjAyODAxNWYxYTIxNGY3YWE4ZGE5ZTU1ODJjMTc4ZmQuYmluZFBvcHVwKHBvcHVwXzUwNzg1ZDBjMWY3YTQxODM4M2Y3ZTkyMzRiOTFiMmJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3NGJiYTRkYjUwMDQwZjRiYzAyN2UyNDQ3ZGIxMDIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAxNDEsLTAuMTk1MTE1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZjI1OTBkNTA3OWI0MjIzYTYyNzQ4NjMwOWJlNDcxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80ZGVlZjY4ZThiYjI0MDNkOWI1OTU5NzVlNTE0YmJhNCA9ICQoJzxkaXYgaWQ9Imh0bWxfNGRlZWY2OGU4YmIyNDAzZDliNTk1OTc1ZTUxNGJiYTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBTVBERU4gSElMTCBST0FELCAyMzc5NjUyLjc8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NmMjU5MGQ1MDc5YjQyMjNhNjI3NDg2MzA5YmU0NzFmLnNldENvbnRlbnQoaHRtbF80ZGVlZjY4ZThiYjI0MDNkOWI1OTU5NzVlNTE0YmJhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NzRiYmE0ZGI1MDA0MGY0YmMwMjdlMjQ0N2RiMTAyMS5iaW5kUG9wdXAocG9wdXBfY2YyNTkwZDUwNzliNDIyM2E2Mjc0ODYzMDliZTQ3MWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGQzNmNiYmE3OTkzNGE1ZThlNTUzZDhkZDg5OTZkZmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi42ODEzNzQ5LDAuOTY1NDcxM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYmY2MTFhZTQ2MDY0MGJkYWZkYWFhNGRhNGNlMWYzNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wOTVlYjg5NWY5MDY0ZmQ3YTgyM2I2NWYwOTQ2NTAzMiA9ICQoJzxkaXYgaWQ9Imh0bWxfMDk1ZWI4OTVmOTA2NGZkN2E4MjNiNjVmMDk0NjUwMzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBTVBJT04gUk9BRCwgMjQ2MTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYmY2MTFhZTQ2MDY0MGJkYWZkYWFhNGRhNGNlMWYzNS5zZXRDb250ZW50KGh0bWxfMDk1ZWI4OTVmOTA2NGZkN2E4MjNiNjVmMDk0NjUwMzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGQzNmNiYmE3OTkzNGE1ZThlNTUzZDhkZDg5OTZkZmMuYmluZFBvcHVwKHBvcHVwX2NiZjYxMWFlNDYwNjQwYmRhZmRhYWE0ZGE0Y2UxZjM1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMzMDZjOThhOTJiMTQ1ZGE5OGY2NTgyNmExYWFlOTMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk5NTY5NiwtMC4xODQyNDc3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FmNTNmMGI1NjRmZDQ2NWJhMDMyYmUzYTZhMzUwOTE5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NjMDMzMjI5MTQ5NDRlYjhhNzI0ZTFhYzgzYjJkYzAwID0gJCgnPGRpdiBpZD0iaHRtbF9jYzAzMzIyOTE0OTQ0ZWI4YTcyNGUxYWM4M2IyZGMwMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0FOTklORyBQTEFDRSwgMjQyNTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZjUzZjBiNTY0ZmQ0NjViYTAzMmJlM2E2YTM1MDkxOS5zZXRDb250ZW50KGh0bWxfY2MwMzMyMjkxNDk0NGViOGE3MjRlMWFjODNiMmRjMDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzMwNmM5OGE5MmIxNDVkYTk4ZjY1ODI2YTFhYWU5MzAuYmluZFBvcHVwKHBvcHVwX2FmNTNmMGI1NjRmZDQ2NWJhMDMyYmUzYTZhMzUwOTE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RmOWI3YmUzNjM5ODRjM2JhYTUzNWU0ZmU3Y2Y3YzZmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTM2LjcwOTE3MTUsMTc0LjcyODI4MDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGQyNTdiNjQxYWNjNGRiYTkyMGVlNTU4ODlkNGIwNWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDIxZjlmNWU4Yzk0NDE2ODkyYmJmOGI5NzZlMGMzZmUgPSAkKCc8ZGl2IGlkPSJodG1sXzQyMWY5ZjVlOGM5NDQxNjg5MmJiZjhiOTc2ZTBjM2ZlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DQVJMSVNMRSBST0FELCAyMjAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkMjU3YjY0MWFjYzRkYmE5MjBlZTU1ODg5ZDRiMDVmLnNldENvbnRlbnQoaHRtbF80MjFmOWY1ZThjOTQ0MTY4OTJiYmY4Yjk3NmUwYzNmZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZjliN2JlMzYzOTg0YzNiYWE1MzVlNGZlN2NmN2M2Zi5iaW5kUG9wdXAocG9wdXBfZGQyNTdiNjQxYWNjNGRiYTkyMGVlNTU4ODlkNGIwNWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTA5ZjNmZTg3ZTBkNDVhNGIyMzE0YmRmM2Y3M2QxNDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstMzcuODAxOTQzMzUsMTQ0Ljk3MTk3MDE3MTAxNzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGRmMzMzNmQzN2EzNDM2NzliM2YxZDZkNjIyYjA1MTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDgzMTk3YmEyOGIxNDAwZjg4MjczYjdhZTRiYjU4NjIgPSAkKCc8ZGl2IGlkPSJodG1sXzA4MzE5N2JhMjhiMTQwMGY4ODI3M2I3YWU0YmI1ODYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DQVJMVE9OIEdBUkRFTlMsIDI0ODM1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGRmMzMzNmQzN2EzNDM2NzliM2YxZDZkNjIyYjA1MTcuc2V0Q29udGVudChodG1sXzA4MzE5N2JhMjhiMTQwMGY4ODI3M2I3YWU0YmI1ODYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUwOWYzZmU4N2UwZDQ1YTRiMjMxNGJkZjNmNzNkMTQ1LmJpbmRQb3B1cChwb3B1cF84ZGYzMzM2ZDM3YTM0MzY3OWIzZjFkNmQ2MjJiMDUxNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZTFmZmY0OGE5ZWY0OTJiYmMwNmY3MDJkZTA4NjA0OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzMyLjk3MjcwMDk1MDAwMDAwNCwtOTcuMTczMzkxNzA5NzcxOTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzE5NTQ1NjQ1ZTA5NDM3Nzg4ZTEwMGU3MTNlMjg4YmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2ZmYjMzODMzNDlhNDc0NWJhZWUxNTFjZmE0MTJjOWYgPSAkKCc8ZGl2IGlkPSJodG1sX2NmZmIzMzgzMzQ5YTQ3NDViYWVlMTUxY2ZhNDEyYzlmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DQVJMWUxFIENPVVJULCAyMzAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxOTU0NTY0NWUwOTQzNzc4OGUxMDBlNzEzZTI4OGJjLnNldENvbnRlbnQoaHRtbF9jZmZiMzM4MzM0OWE0NzQ1YmFlZTE1MWNmYTQxMmM5Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZTFmZmY0OGE5ZWY0OTJiYmMwNmY3MDJkZTA4NjA0OC5iaW5kUG9wdXAocG9wdXBfMzE5NTQ1NjQ1ZTA5NDM3Nzg4ZTEwMGU3MTNlMjg4YmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWNkNTUxZGZjNDcyNDg2ZjljYTg4ZGMzOWJhODg1YjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41NDExOTU1LC0wLjE1NTgxNjhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjQwMGJhZmY4ZTFhNDFiY2FjMzEwN2MxMDljZDg3ZTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjQwNWYzODdkOWZjNDE3MmJiYzI4ZTI0ZDRmMzA3ZGIgPSAkKCc8ZGl2IGlkPSJodG1sXzI0MDVmMzg3ZDlmYzQxNzJiYmMyOGUyNGQ0ZjMwN2RiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEFMQ09UIFNRVUFSRSwgMjI4NjY3OC41NzE0Mjg1NzE0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NDAwYmFmZjhlMWE0MWJjYWMzMTA3YzEwOWNkODdlMC5zZXRDb250ZW50KGh0bWxfMjQwNWYzODdkOWZjNDE3MmJiYzI4ZTI0ZDRmMzA3ZGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWNkNTUxZGZjNDcyNDg2ZjljYTg4ZGMzOWJhODg1YjUuYmluZFBvcHVwKHBvcHVwXzY0MDBiYWZmOGUxYTQxYmNhYzMxMDdjMTA5Y2Q4N2UwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk3NWI4OTA4M2EyNjQzMWU5NGFiYTA5ZTdiYTdjODMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTMzODM3LC0wLjE3MDI5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNjFlZjE3ZDZkMzM0MTMzYjJhNWM0MzZhNDUwNTdjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xM2IwMmRiN2U5ZTk0NWJiODNlMWNkYjAzYWJmYWJmMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTNiMDJkYjdlOWU5NDViYjgzZTFjZGIwM2FiZmFiZjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNIQVJMRVMgTEFORSwgMjQxNDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lNjFlZjE3ZDZkMzM0MTMzYjJhNWM0MzZhNDUwNTdjZC5zZXRDb250ZW50KGh0bWxfMTNiMDJkYjdlOWU5NDViYjgzZTFjZGIwM2FiZmFiZjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTc1Yjg5MDgzYTI2NDMxZTk0YWJhMDllN2JhN2M4MzAuYmluZFBvcHVwKHBvcHVwX2U2MWVmMTdkNmQzMzQxMzNiMmE1YzQzNmE0NTA1N2NkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhYzhjYWJkYzU3NDRjNzI4MmI5OTk1M2Q5MTM0YWJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzQuNTIyNDQzLC04NS40NDM4OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODYwNjcxYjE5Mjc4NDY0MWFlYTRmN2YzYjcyMDU4NmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTc0ZDY4YjBmOTc2NDNhZjg2MDUwNzI5MzhhMzNkZDggPSAkKCc8ZGl2IGlkPSJodG1sX2E3NGQ2OGIwZjk3NjQzYWY4NjA1MDcyOTM4YTMzZGQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEVMU0VBIENSRVNDRU5ULCAyNDk1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg2MDY3MWIxOTI3ODQ2NDFhZWE0ZjdmM2I3MjA1ODZhLnNldENvbnRlbnQoaHRtbF9hNzRkNjhiMGY5NzY0M2FmODYwNTA3MjkzOGEzM2RkOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YWM4Y2FiZGM1NzQ0YzcyODJiOTk5NTNkOTEzNGFiYS5iaW5kUG9wdXAocG9wdXBfODYwNjcxYjE5Mjc4NDY0MWFlYTRmN2YzYjcyMDU4NmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDYwOTg4N2VlOWE5NGU5OTk2ZGFkY2NiNjJiNThkNjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjkyMDU0LC0wLjE0NTA4MTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDg2YWNkODUxZWY3NDU1NDk2OTg1OGJjNWNjMDg5ZTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjIwNDUzZDc0OGJlNDM5MDg0MjA3NTg0ZjUzZjcxMTIgPSAkKCc8ZGl2IGlkPSJodG1sX2IyMDQ1M2Q3NDhiZTQzOTA4NDIwNzU4NGY1M2Y3MTEyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEVTVEVSIENMT1NFIE5PUlRILCAyNDUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q4NmFjZDg1MWVmNzQ1NTQ5Njk4NThiYzVjYzA4OWUwLnNldENvbnRlbnQoaHRtbF9iMjA0NTNkNzQ4YmU0MzkwODQyMDc1ODRmNTNmNzExMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NjA5ODg3ZWU5YTk0ZTk5OTZkYWRjY2I2MmI1OGQ2Ni5iaW5kUG9wdXAocG9wdXBfZDg2YWNkODUxZWY3NDU1NDk2OTg1OGJjNWNjMDg5ZTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWQ2OTI2YjE3ZDlkNDczNGFmYTFiNWZiMjliYjZmOWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41OTk2NzcsMC41MjU2MjMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg0MGE5NDU5YWVjMDQyOGI5OTMwYjgwZDE3YjI3NWRiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhkNWE3MGQxY2RkYjRlMzg4NmI0NDI1MmUzNTUyZTRkID0gJCgnPGRpdiBpZD0iaHRtbF84ZDVhNzBkMWNkZGI0ZTM4ODZiNDQyNTJlMzU1MmU0ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0hFWU5FIENPVVJULCAyMjUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg0MGE5NDU5YWVjMDQyOGI5OTMwYjgwZDE3YjI3NWRiLnNldENvbnRlbnQoaHRtbF84ZDVhNzBkMWNkZGI0ZTM4ODZiNDQyNTJlMzU1MmU0ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZDY5MjZiMTdkOWQ0NzM0YWZhMWI1ZmIyOWJiNmY5Yy5iaW5kUG9wdXAocG9wdXBfODQwYTk0NTlhZWMwNDI4Yjk5MzBiODBkMTdiMjc1ZGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTEyODJjYTMxYTNkNDY5MTlhMjU5NzM0ZDRjMjI5M2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40ODM3MTczLC0wLjE2OTYwM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNDQwOGYyY2RkMmM0NjU5OTE1OWNmZjViNTViNTFkMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZGNjODZmYzkxZTI0ODRlOGJmNGNlOTFjYzliZWNkMyA9ICQoJzxkaXYgaWQ9Imh0bWxfYWRjYzg2ZmM5MWUyNDg0ZThiZjRjZTkxY2M5YmVjZDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNIRVlORSBST1csIDI0MTAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTQ0MDhmMmNkZDJjNDY1OTkxNTljZmY1YjU1YjUxZDMuc2V0Q29udGVudChodG1sX2FkY2M4NmZjOTFlMjQ4NGU4YmY0Y2U5MWNjOWJlY2QzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUxMjgyY2EzMWEzZDQ2OTE5YTI1OTczNGQ0YzIyOTNhLmJpbmRQb3B1cChwb3B1cF9lNDQwOGYyY2RkMmM0NjU5OTE1OWNmZjViNTViNTFkMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MDI4MzVhMzc4NjI0YmZhOTcyMjUwM2Q0YzM5Zjk4NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NzE4NDksLTAuMjQ4MDE2OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZDY3OWI4NDI4MTQ0NGNiYjJkNDQxMjg2MzgxMGU4YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MWZhNWEwNjk0NDI0NmFlOTQyNjdlZjdhODNiNDUzMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTFmYTVhMDY5NDQyNDZhZTk0MjY3ZWY3YTgzYjQ1MzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNISVNXSUNLIE1BTEwsIDIyODc1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2Q2NzliODQyODE0NDRjYmIyZDQ0MTI4NjM4MTBlOGEuc2V0Q29udGVudChodG1sXzUxZmE1YTA2OTQ0MjQ2YWU5NDI2N2VmN2E4M2I0NTMwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwMjgzNWEzNzg2MjRiZmE5NzIyNTAzZDRjMzlmOTg3LmJpbmRQb3B1cChwb3B1cF8zZDY3OWI4NDI4MTQ0NGNiYjJkNDQxMjg2MzgxMGU4YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZDY3N2RiOTI4YTE0MWE5YmY2YWYyY2VmNmE0MDJlNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUyOTY5NzIsLTAuMDk3NzYyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNzI4YmEyYjk0YmY0ZmY5YTExMzUxN2M4NzZmZGYyYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYTU2YTdkNDUzMzY0ZWJiOGE5MDhkMWJlYjg4NzkzNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmE1NmE3ZDQ1MzM2NGViYjhhOTA4ZDFiZWI4ODc5MzUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNJVFkgUk9BRCwgMjQ2ODM0MC4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNzI4YmEyYjk0YmY0ZmY5YTExMzUxN2M4NzZmZGYyYi5zZXRDb250ZW50KGh0bWxfMmE1NmE3ZDQ1MzM2NGViYjhhOTA4ZDFiZWI4ODc5MzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2Q2NzdkYjkyOGExNDFhOWJmNmFmMmNlZjZhNDAyZTUuYmluZFBvcHVwKHBvcHVwXzI3MjhiYTJiOTRiZjRmZjlhMTEzNTE3Yzg3NmZkZjJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdjOTdlOTkzNzE3MjQxNDM5ZWU1Nzk4NzUwMzQ1YTIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzY1MTYsMS4xMDg1NjkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I1YWVlMDc2NjBkMjQ4NmFiMzFiM2IzZDQ2NTdhNGZkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAyMTdjOWYwZGQ1YjQ4NDNhODNmMmRkY2RiMDY3MWQzID0gJCgnPGRpdiBpZD0iaHRtbF8wMjE3YzlmMGRkNWI0ODQzYTgzZjJkZGNkYjA2NzFkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0xBUkVORE9OIFNUUkVFVCwgMjI1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNWFlZTA3NjYwZDI0ODZhYjMxYjNiM2Q0NjU3YTRmZC5zZXRDb250ZW50KGh0bWxfMDIxN2M5ZjBkZDViNDg0M2E4M2YyZGRjZGIwNjcxZDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2M5N2U5OTM3MTcyNDE0MzllZTU3OTg3NTAzNDVhMjIuYmluZFBvcHVwKHBvcHVwX2I1YWVlMDc2NjBkMjQ4NmFiMzFiM2IzZDQ2NTdhNGZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNmMzA1MWUzM2JiMDRmNTFhYmJiNDE0YjIxMDk3OWRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDczNzYzMiwtMC4yMTYyNDQyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZmZjllZGU2NWJiNzRlOWRiYmYwZTAwNTlmNDExYWM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI5ZjViYTgwY2MxYjRhNjE5ZDA3OTRkZGY4MDFmNDIxID0gJCgnPGRpdiBpZD0iaHRtbF8yOWY1YmE4MGNjMWI0YTYxOWQwNzk0ZGRmODAxZjQyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0xPTkNVUlJZIFNUUkVFVCwgMjM4ODMzMy4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZmY5ZWRlNjViYjc0ZTlkYmJmMGUwMDU5ZjQxMWFjOC5zZXRDb250ZW50KGh0bWxfMjlmNWJhODBjYzFiNGE2MTlkMDc5NGRkZjgwMWY0MjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2YzMDUxZTMzYmIwNGY1MWFiYmI0MTRiMjEwOTc5ZGEuYmluZFBvcHVwKHBvcHVwXzZmZjllZGU2NWJiNzRlOWRiYmYwZTAwNTlmNDExYWM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAwNmViNmI3ODdjNDQ2ZDhiZjAwZDBjNDE2M2QxOGYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDkyOTQ5NywtMC4xODU5MjE1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U3MjIyNGVkZjgzZjQ4MDM4YzE4NWFkOTg1MTk5NDVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FjMDJhZDNlMGYyNjQxMTE4ZjIyY2QyNzgyZmExYTE5ID0gJCgnPGRpdiBpZD0iaHRtbF9hYzAyYWQzZTBmMjY0MTExOGYyMmNkMjc4MmZhMWExOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q09MQkVDSyBNRVdTLCAyMzY3NTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U3MjIyNGVkZjgzZjQ4MDM4YzE4NWFkOTg1MTk5NDVlLnNldENvbnRlbnQoaHRtbF9hYzAyYWQzZTBmMjY0MTExOGYyMmNkMjc4MmZhMWExOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMDZlYjZiNzg3YzQ0NmQ4YmYwMGQwYzQxNjNkMThmMS5iaW5kUG9wdXAocG9wdXBfZTcyMjI0ZWRmODNmNDgwMzhjMTg1YWQ5ODUxOTk0NWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2FiMWQ5NjI4MmI2NDFmMDk0YjM2YzYzNWIwYTAzZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0Mi44MTg2MjE5LC03My45MjU4ODEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlmMmI4YTdmODUzZDQ0OTg5MzU0YjJkYmJhZWVmOWIwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJjYzU0MGUyYzMyYTQxMDRhNDllZWQxZjAzZWJlMjVjID0gJCgnPGRpdiBpZD0iaHRtbF8yY2M1NDBlMmMzMmE0MTA0YTQ5ZWVkMWYwM2ViZTI1YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q09MTEVHRSBDUkVTQ0VOVCwgMjQ0MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZjJiOGE3Zjg1M2Q0NDk4OTM1NGIyZGJiYWVlZjliMC5zZXRDb250ZW50KGh0bWxfMmNjNTQwZTJjMzJhNDEwNGE0OWVlZDFmMDNlYmUyNWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2FiMWQ5NjI4MmI2NDFmMDk0YjM2YzYzNWIwYTAzZDIuYmluZFBvcHVwKHBvcHVwXzlmMmI4YTdmODUzZDQ0OTg5MzU0YjJkYmJhZWVmOWIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBiMTM0NDJiODhhNzQ5OTFhMjJjY2MwNTQ5NWE2ZjRlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI0MDY1NywtMC4xNTc2NjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjE3ZjVkNTAxMDk1NDNiMWFlZjFjYzI4MGNlZjY4MTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjA2YzMyOWU5MTBhNDgzMThkYmNkOTMwYzdmNmRkMzggPSAkKCc8ZGl2IGlkPSJodG1sXzIwNmMzMjllOTEwYTQ4MzE4ZGJjZDkzMGM3ZjZkZDM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DT1JOV0FMTCBURVJSQUNFIE1FV1MsIDIzNTAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjE3ZjVkNTAxMDk1NDNiMWFlZjFjYzI4MGNlZjY4MTkuc2V0Q29udGVudChodG1sXzIwNmMzMjllOTEwYTQ4MzE4ZGJjZDkzMGM3ZjZkZDM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBiMTM0NDJiODhhNzQ5OTFhMjJjY2MwNTQ5NWE2ZjRlLmJpbmRQb3B1cChwb3B1cF8yMTdmNWQ1MDEwOTU0M2IxYWVmMWNjMjgwY2VmNjgxOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ZTM4MjNiMDcxODU0ZTkwYmEwMTVjZDQ5YmU0MzljMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0ODUwMDIsLTAuMDgwMTAxNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NDcyNzU5YmI0MWM0ZDczYjU5OTk4MzNiNjVkNzQ4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZTUyNjQ1MzA2OTU0ODNjYjY4ZmFhMTg1MTE5YzdjNCA9ICQoJzxkaXYgaWQ9Imh0bWxfY2U1MjY0NTMwNjk1NDgzY2I2OGZhYTE4NTExOWM3YzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNPVVJUIExBTkUgR0FSREVOUywgMjM2MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NDcyNzU5YmI0MWM0ZDczYjU5OTk4MzNiNjVkNzQ4Mi5zZXRDb250ZW50KGh0bWxfY2U1MjY0NTMwNjk1NDgzY2I2OGZhYTE4NTExOWM3YzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmUzODIzYjA3MTg1NGU5MGJhMDE1Y2Q0OWJlNDM5YzAuYmluZFBvcHVwKHBvcHVwXzQ0NzI3NTliYjQxYzRkNzNiNTk5OTgzM2I2NWQ3NDgyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UyMTdlZjdkYTE4ZTQxODg5YzRlZDViNWFmMWVkMmVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDYuMTE1NTg1NiwtNjAuNzI1MzMwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MGVkZjQwZTYzN2U0MjlkYjYwODcyZTJlMjMxMTZmMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMmVmNGM1N2U5OTY0YjNlOGUxNjViN2QyODRmNGEyOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjJlZjRjNTdlOTk2NGIzZThlMTY1YjdkMjg0ZjRhMjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNSRVNDRU5UIEdST1ZFLCAyMjk4MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYwZWRmNDBlNjM3ZTQyOWRiNjA4NzJlMmUyMzExNmYzLnNldENvbnRlbnQoaHRtbF8yMmVmNGM1N2U5OTY0YjNlOGUxNjViN2QyODRmNGEyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMjE3ZWY3ZGExOGU0MTg4OWM0ZWQ1YjVhZjFlZDJlYS5iaW5kUG9wdXAocG9wdXBfNjBlZGY0MGU2MzdlNDI5ZGI2MDg3MmUyZTIzMTE2ZjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTA1OGZjNDM0MmJiNGI4NGIxNmM0MTJiN2M2NDMxMzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MzgyNjgzLC0wLjE2NzY2OTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODg2NTQ2NGIyMTk1NDQwNmE3ODc1ZDk2OTAxOTY4NjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTM3ZGI4MzA0ZGNiNGZkYTk4ZDY2ZTAwNzhhMzI1ZWUgPSAkKCc8ZGl2IGlkPSJodG1sXzEzN2RiODMwNGRjYjRmZGE5OGQ2NmUwMDc4YTMyNWVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EQUxFQlVSWSBST0FELCAyNDAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4NjU0NjRiMjE5NTQ0MDZhNzg3NWQ5NjkwMTk2ODY0LnNldENvbnRlbnQoaHRtbF8xMzdkYjgzMDRkY2I0ZmRhOThkNjZlMDA3OGEzMjVlZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMDU4ZmM0MzQyYmI0Yjg0YjE2YzQxMmI3YzY0MzEzMS5iaW5kUG9wdXAocG9wdXBfODg2NTQ2NGIyMTk1NDQwNmE3ODc1ZDk2OTAxOTY4NjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjE3YmUxZGQ5YTI5NDY1YzliNmY4NTJkNmRiYzhmNzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTg3Mzc1LC0wLjIyMDY2NjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTNhM2FhM2I1MDczNGYxNjliNmVjZDQ2Y2FjOTQ2ODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTQ5NTZhYmFjNTEwNDYxOTllNzMxNDhjNDZkMGJkNDAgPSAkKCc8ZGl2IGlkPSJodG1sX2E0OTU2YWJhYzUxMDQ2MTk5ZTczMTQ4YzQ2ZDBiZDQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ERVdIVVJTVCBST0FELCAyNDI1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzYTNhYTNiNTA3MzRmMTY5YjZlY2Q0NmNhYzk0Njg1LnNldENvbnRlbnQoaHRtbF9hNDk1NmFiYWM1MTA0NjE5OWU3MzE0OGM0NmQwYmQ0MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMTdiZTFkZDlhMjk0NjVjOWI2Zjg1MmQ2ZGJjOGY3MC5iaW5kUG9wdXAocG9wdXBfNTNhM2FhM2I1MDczNGYxNjliNmVjZDQ2Y2FjOTQ2ODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODYwMDIyMWMyM2M0NDIzZWE3YmVkNzNhYmRkYTI0MjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NzMxMTU3LC0wLjIwMTc0OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZTNiOGY0NmMyNmU0NzMyYjExYzFmNGQ5NzFkZmI5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMmFhNDI3ZmM4MDM0YzQ2YmQ3OTU3NGU2NjY4N2Q3NyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzJhYTQyN2ZjODAzNGM0NmJkNzk1NzRlNjY2ODdkNzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRPUklBIFJPQUQsIDIzNjI1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWUzYjhmNDZjMjZlNDczMmIxMWMxZjRkOTcxZGZiOTEuc2V0Q29udGVudChodG1sXzMyYWE0MjdmYzgwMzRjNDZiZDc5NTc0ZTY2Njg3ZDc3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg2MDAyMjFjMjNjNDQyM2VhN2JlZDczYWJkZGEyNDIyLmJpbmRQb3B1cChwb3B1cF85ZTNiOGY0NmMyNmU0NzMyYjExYzFmNGQ5NzFkZmI5MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMjY1Y2EwNTMxYzA0MTFjYWFiMTk5NjA3NzY4ZTgxZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU1NTY2MiwtMC4xNzAyOTM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNhMmNkNjNkZDdkMDQ5MWQ5N2UxNWNjMWQ0NGE5MjA5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2YzgxYTVmOWRhMzQ5MTM5ODBiM2VhMzNjZDk2MzczID0gJCgnPGRpdiBpZD0iaHRtbF9iNmM4MWE1ZjlkYTM0OTEzOTgwYjNlYTMzY2Q5NjM3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RE9XTlNISVJFIEhJTEwsIDIyMjUwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2EyY2Q2M2RkN2QwNDkxZDk3ZTE1Y2MxZDQ0YTkyMDkuc2V0Q29udGVudChodG1sX2I2YzgxYTVmOWRhMzQ5MTM5ODBiM2VhMzNjZDk2MzczKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEyNjVjYTA1MzFjMDQxMWNhYWIxOTk2MDc3NjhlODFmLmJpbmRQb3B1cChwb3B1cF8zYTJjZDYzZGQ3ZDA0OTFkOTdlMTVjYzFkNDRhOTIwOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lODZjYTFkOTBlODg0MjdhOTYxMzhjZjM0MjAyNTNkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUwMzgwMSwtMC4wNzY5MjEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYzOTljMzIyOGY1MTQ3YzU5NmYzOGE3ZjAxYTE4NjczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJhNzY4N2ZiYThkOTQ4OGI5MjI0YjM0NjMyOWQxZDllID0gJCgnPGRpdiBpZD0iaHRtbF8yYTc2ODdmYmE4ZDk0ODhiOTIyNGIzNDYzMjlkMWQ5ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RFVDSEVTUyBXQUxLLCAyNDc3NTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYzOTljMzIyOGY1MTQ3YzU5NmYzOGE3ZjAxYTE4NjczLnNldENvbnRlbnQoaHRtbF8yYTc2ODdmYmE4ZDk0ODhiOTIyNGIzNDYzMjlkMWQ5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lODZjYTFkOTBlODg0MjdhOTYxMzhjZjM0MjAyNTNkMS5iaW5kUG9wdXAocG9wdXBfNjM5OWMzMjI4ZjUxNDdjNTk2ZjM4YTdmMDFhMTg2NzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOThiMmM5MTJlNjYzNGRiMDhhYzlmMzc4MjgzMzY3NGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTE3ODQ5LC0wLjE0MjI1OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjhiNThmNmE1Y2ZjNGQ2Nzg2MDA3OTBiYTYzNjIwNDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDg0M2M0ZGE0MDc0NGM2NjlhYWE4YzZmZmYxZTFiMDAgPSAkKCc8ZGl2IGlkPSJodG1sX2Q4NDNjNGRhNDA3NDRjNjY5YWFhOGM2ZmZmMWUxYjAwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FQ0NMRVNUT04gU1FVQVJFIE1FV1MsIDIzMzU0OTkuNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjhiNThmNmE1Y2ZjNGQ2Nzg2MDA3OTBiYTYzNjIwNDkuc2V0Q29udGVudChodG1sX2Q4NDNjNGRhNDA3NDRjNjY5YWFhOGM2ZmZmMWUxYjAwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk4YjJjOTEyZTY2MzRkYjA4YWM5ZjM3ODI4MzM2NzRjLmJpbmRQb3B1cChwb3B1cF8yOGI1OGY2YTVjZmM0ZDY3ODYwMDc5MGJhNjM2MjA0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MWY0MzIwZDIwMTY0NjUxODAzZmYxNTk0N2VmYjlhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUzLjUwNzIxOCwtMi4xOTA2MDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M4NGE3ZjQ0N2NhODQxMGNiNGVmMzc0ZTJkN2UwYmJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y5MmE2MTMyNjc3YjRkYzNhMzk1NTI2NDVkNTQwM2EzID0gJCgnPGRpdiBpZD0iaHRtbF9mOTJhNjEzMjY3N2I0ZGMzYTM5NTUyNjQ1ZDU0MDNhMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RUdCRVJUIFNUUkVFVCwgMjI2NTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jODRhN2Y0NDdjYTg0MTBjYjRlZjM3NGUyZDdlMGJiZS5zZXRDb250ZW50KGh0bWxfZjkyYTYxMzI2NzdiNGRjM2EzOTU1MjY0NWQ1NDAzYTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTFmNDMyMGQyMDE2NDY1MTgwM2ZmMTU5NDdlZmI5YTguYmluZFBvcHVwKHBvcHVwX2M4NGE3ZjQ0N2NhODQxMGNiNGVmMzc0ZTJkN2UwYmJlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IzNjA1ODUyNGNkMzQ1MTc5NDYwOGRmOTZlZDJiZTQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk2Njg2NywtMC4xNjY5NDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVlMzhlNzU0NzczNTRhMTA5OGRjNWYyMWNjMDMxZmNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5NzQ4Yzc0NTVjZDQ3ZjA5MjRiNGFiODMxY2VhYWUxID0gJCgnPGRpdiBpZD0iaHRtbF9iOTc0OGM3NDU1Y2Q0N2YwOTI0YjRhYjgzMWNlYWFlMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RUdFUlRPTiBQTEFDRSwgMjIwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZTM4ZTc1NDc3MzU0YTEwOThkYzVmMjFjYzAzMWZjZi5zZXRDb250ZW50KGh0bWxfYjk3NDhjNzQ1NWNkNDdmMDkyNGI0YWI4MzFjZWFhZTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjM2MDU4NTI0Y2QzNDUxNzk0NjA4ZGY5NmVkMmJlNDguYmluZFBvcHVwKHBvcHVwXzVlMzhlNzU0NzczNTRhMTA5OGRjNWYyMWNjMDMxZmNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk4NzQwYzhhNjAxMDQ2NWU4Nzg1OTIwYzlhMzQxYzVmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNjMzOTg3MywtMC4wOTI1MjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTZjZjA3MDkwMzliNDIzZGFiZWMyZTUwNWZmYjFiMTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzUzZjYxNGM3YmVkNGRiNWExNzVhNjZiYmM3ODg1NjMgPSAkKCc8ZGl2IGlkPSJodG1sXzM1M2Y2MTRjN2JlZDRkYjVhMTc1YTY2YmJjNzg4NTYzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FTE0gUEFSSyBST0FELCAyMzIwNDI1LjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE2Y2YwNzA5MDM5YjQyM2RhYmVjMmU1MDVmZmIxYjEzLnNldENvbnRlbnQoaHRtbF8zNTNmNjE0YzdiZWQ0ZGI1YTE3NWE2NmJiYzc4ODU2Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ODc0MGM4YTYwMTA0NjVlODc4NTkyMGM5YTM0MWM1Zi5iaW5kUG9wdXAocG9wdXBfMTZjZjA3MDkwMzliNDIzZGFiZWMyZTUwNWZmYjFiMTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGJiMDJlODk5NzcyNGJhMGJhZGEwMWExYTlmMzA2ZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTMyMTA1LC0wLjEyMjk3MzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjk5MDA0MzA5YzE4NGJmZmEzZWE1MTk3MzBhYzhmMDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTIxOGYxZDgzYWJhNDc2NDg4NjEzM2ZjNTk1YWMxOTIgPSAkKCc8ZGl2IGlkPSJodG1sX2EyMThmMWQ4M2FiYTQ3NjQ4ODYxMzNmYzU5NWFjMTkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GTE9SQUwgU1RSRUVULCAyMjI3MjIyLjIyMjIyMjIyMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjk5MDA0MzA5YzE4NGJmZmEzZWE1MTk3MzBhYzhmMDEuc2V0Q29udGVudChodG1sX2EyMThmMWQ4M2FiYTQ3NjQ4ODYxMzNmYzU5NWFjMTkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRiYjAyZTg5OTc3MjRiYTBiYWRhMDFhMWE5ZjMwNmQzLmJpbmRQb3B1cChwb3B1cF82OTkwMDQzMDljMTg0YmZmYTNlYTUxOTczMGFjOGYwMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNDE2YzE4MDJjZGQ0YjM4OGZiNTE3YmQ5NTc5MTI3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0Mjc5NDksLTAuMDgwMzk3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YjQzOWM4YjkzZjc0NjY3YTA4MzllZDU0YzViYTZlNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYzUyOGU5ZjcxOGY0MzgyOWIxNDMzODQ1NDI4Y2Q2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfY2M1MjhlOWY3MThmNDM4MjliMTQzMzg0NTQyOGNkNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZSQU5LIERJWE9OIFdBWSwgMjIxMjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YjQzOWM4YjkzZjc0NjY3YTA4MzllZDU0YzViYTZlNi5zZXRDb250ZW50KGh0bWxfY2M1MjhlOWY3MThmNDM4MjliMTQzMzg0NTQyOGNkNjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjQxNmMxODAyY2RkNGIzODhmYjUxN2JkOTU3OTEyNzAuYmluZFBvcHVwKHBvcHVwXzZiNDM5YzhiOTNmNzQ2NjdhMDgzOWVkNTRjNWJhNmU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFhNGZiNThhNjkwNzQ0ZmZhOTBiMzIwZmE2Njk4ZDY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTEyNTU4MiwtMC4xODQ2Mjg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UwMjViYzhjMjUxZDRlZDQ5YTUwNmJiMDE3MmFhOTg2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQyN2RjYzlkYWMzMDRlNzdiZjQyZmE5NDgwNmMyODFhID0gJCgnPGRpdiBpZD0iaHRtbF80MjdkY2M5ZGFjMzA0ZTc3YmY0MmZhOTQ4MDZjMjgxYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RlVMVE9OIE1FV1MsIDIyOTkwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTAyNWJjOGMyNTFkNGVkNDlhNTA2YmIwMTcyYWE5ODYuc2V0Q29udGVudChodG1sXzQyN2RjYzlkYWMzMDRlNzdiZjQyZmE5NDgwNmMyODFhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFhNGZiNThhNjkwNzQ0ZmZhOTBiMzIwZmE2Njk4ZDY0LmJpbmRQb3B1cChwb3B1cF9lMDI1YmM4YzI1MWQ0ZWQ0OWE1MDZiYjAxNzJhYTk4Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NzBmNzRkNmIxNTc0ZjBiYmM3NjFkZTkyN2Y1NGFhMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjIwOTY5MzEsMC4xNTg4NzUyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I4MWEyNTkxNjdiMTQ5NWU5N2VlZTM5YzY1Njc5NDgyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YyM2ExYzY3ZTAyNTQ2NWRhNGQwMmViOTQzYzEyMTc2ID0gJCgnPGRpdiBpZD0iaHRtbF9mMjNhMWM2N2UwMjU0NjVkYTRkMDJlYjk0M2MxMjE3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R0VSQVJEIFJPQUQsIDIyNTg1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjgxYTI1OTE2N2IxNDk1ZTk3ZWVlMzljNjU2Nzk0ODIuc2V0Q29udGVudChodG1sX2YyM2ExYzY3ZTAyNTQ2NWRhNGQwMmViOTQzYzEyMTc2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ3MGY3NGQ2YjE1NzRmMGJiYzc2MWRlOTI3ZjU0YWEwLmJpbmRQb3B1cChwb3B1cF9iODFhMjU5MTY3YjE0OTVlOTdlZWUzOWM2NTY3OTQ4Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNDdlOWQ4NGQ2MWQ0OWExODg1MGU2NjMyMTlmZWRjYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUzMzg2NTYsLTAuMTAxMjQ3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81Mjk3ZDNiMDNjMzQ0MzcxOTA4NWRiOWY0MjNiZDNlYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZGY5MmEwNzA5Y2Q0MDRhOWQ1NWFiNTIwZTU0YmQ3YyA9ICQoJzxkaXYgaWQ9Imh0bWxfZmRmOTJhMDcwOWNkNDA0YTlkNTVhYjUyMGU1NGJkN2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdFUlJBUkQgUk9BRCwgMjI0MjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81Mjk3ZDNiMDNjMzQ0MzcxOTA4NWRiOWY0MjNiZDNlYy5zZXRDb250ZW50KGh0bWxfZmRmOTJhMDcwOWNkNDA0YTlkNTVhYjUyMGU1NGJkN2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTQ3ZTlkODRkNjFkNDlhMTg4NTBlNjYzMjE5ZmVkY2EuYmluZFBvcHVwKHBvcHVwXzUyOTdkM2IwM2MzNDQzNzE5MDg1ZGI5ZjQyM2JkM2VjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ5OGY5OTMxODk1ZjQxODJhYjM4Yzk0NzdhYmYzYzY5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk2Njk2OCwtMC4yMTU1MzA2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU1ZWFlMjU0N2Y0NDQ1YmU5N2Q5N2VhYTcwZWRiNmVkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzczYWU0N2U5Nzc3YzRhMWU4ZDc3ZmMyZWRlZTA0ODg4ID0gJCgnPGRpdiBpZD0iaHRtbF83M2FlNDdlOTc3N2M0YTFlOGQ3N2ZjMmVkZWUwNDg4OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R0lSRExFUlMgUk9BRCwgMjQ0MTY2Ni42NjY2NjY2NjY1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NWVhZTI1NDdmNDQ0NWJlOTdkOTdlYWE3MGVkYjZlZC5zZXRDb250ZW50KGh0bWxfNzNhZTQ3ZTk3NzdjNGExZThkNzdmYzJlZGVlMDQ4ODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDk4Zjk5MzE4OTVmNDE4MmFiMzhjOTQ3N2FiZjNjNjkuYmluZFBvcHVwKHBvcHVwXzU1ZWFlMjU0N2Y0NDQ1YmU5N2Q5N2VhYTcwZWRiNmVkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VhNDljNmM4NDg3ZjQxZmNiNzIxY2ZlN2QzOTI3NjkxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDkuNjMzODIzNSwtMTE4LjQwNjY4NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzA3MmU0ZGMwODhiNDc1NWFlYjhjYTA5NzlhYjUwNWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzg0NmJkOTI3OGVmNGIyM2I3MzE4Y2RlZTVmMmU4MWMgPSAkKCc8ZGl2IGlkPSJodG1sXzc4NDZiZDkyNzhlZjRiMjNiNzMxOGNkZWU1ZjJlODFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HTE9VQ0VTVEVSIENSRVNDRU5ULCAyMzUwODMzLjMzMzMzMzMzMzU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcwNzJlNGRjMDg4YjQ3NTVhZWI4Y2EwOTc5YWI1MDVhLnNldENvbnRlbnQoaHRtbF83ODQ2YmQ5Mjc4ZWY0YjIzYjczMThjZGVlNWYyZTgxYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYTQ5YzZjODQ4N2Y0MWZjYjcyMWNmZTdkMzkyNzY5MS5iaW5kUG9wdXAocG9wdXBfNzA3MmU0ZGMwODhiNDc1NWFlYjhjYTA5NzlhYjUwNWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjdiYmQ3ZmFiMWUwNGJkNDg2MTcxOGJlN2FjODhhZDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstNDEuMTY2NDk3OSwxNDYuMzQ2MzU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4NDdkYmY2M2FlODQ1ZjU4ZjZlMDNhMTZhYWYyYjg3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QxODMxYWJkYWJjNTQyOGNiYjUxYTJkY2IyNmMwN2FjID0gJCgnPGRpdiBpZD0iaHRtbF9kMTgzMWFiZGFiYzU0MjhjYmI1MWEyZGNiMjZjMDdhYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R09SRE9OIFBMQUNFLCAyNDc3MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U4NDdkYmY2M2FlODQ1ZjU4ZjZlMDNhMTZhYWYyYjg3LnNldENvbnRlbnQoaHRtbF9kMTgzMWFiZGFiYzU0MjhjYmI1MWEyZGNiMjZjMDdhYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iN2JiZDdmYWIxZTA0YmQ0ODYxNzE4YmU3YWM4OGFkNS5iaW5kUG9wdXAocG9wdXBfZTg0N2RiZjYzYWU4NDVmNThmNmUwM2ExNmFhZjJiODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzhmYjJmMWRlZmU5NDg0Nzg5MTc5ODU0ZjZjYTRkYTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NjM5NjU5LC0wLjEzOTA4NDNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzM0ODA4MjNjNDIwNGY3ZTg0MjgxN2U0ZWExOGNjNzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTY2ZTNmM2JlNDlhNGQ4MWIwMGNlOWRiM2FhMmUyMTggPSAkKCc8ZGl2IGlkPSJodG1sXzE2NmUzZjNiZTQ5YTRkODFiMDBjZTlkYjNhYTJlMjE4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HUkFGVE9OIFNRVUFSRSwgMjI3MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMzQ4MDgyM2M0MjA0ZjdlODQyODE3ZTRlYTE4Y2M3Ni5zZXRDb250ZW50KGh0bWxfMTY2ZTNmM2JlNDlhNGQ4MWIwMGNlOWRiM2FhMmUyMTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzhmYjJmMWRlZmU5NDg0Nzg5MTc5ODU0ZjZjYTRkYTEuYmluZFBvcHVwKHBvcHVwXzMzNDgwODIzYzQyMDRmN2U4NDI4MTdlNGVhMThjYzc2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UwYjVmNTFmYTA1NjQxZmI5ZTcwZjIyOWUzNTA1YmVkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDkxNTQ3NCwtMC4xNTQyNzUxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzljYTRkYTUwZTNiZjQxYzZhODQyNGVlOGMyOWFjNmZkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3NGVlYjZjYzFhMTQwNjViZDEwYmJiZGNiZmQxY2Q1ID0gJCgnPGRpdiBpZD0iaHRtbF8wNzRlZWI2Y2MxYTE0MDY1YmQxMGJiYmRjYmZkMWNkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R1JBSEFNIFRFUlJBQ0UsIDIzMjUwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWNhNGRhNTBlM2JmNDFjNmE4NDI0ZWU4YzI5YWM2ZmQuc2V0Q29udGVudChodG1sXzA3NGVlYjZjYzFhMTQwNjViZDEwYmJiZGNiZmQxY2Q1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UwYjVmNTFmYTA1NjQxZmI5ZTcwZjIyOWUzNTA1YmVkLmJpbmRQb3B1cChwb3B1cF85Y2E0ZGE1MGUzYmY0MWM2YTg0MjRlZThjMjlhYzZmZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84Y2I2M2ZmNjYwZGI0MzRkODE0MjkyYjEyMDY5Yjk5NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU1ODczNzksLTAuMjA2MzA2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZmQzZTRmYzU5NDk0Nzg5YTBjOTQ3NzViZWRkYzI1NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNGY5Nzk1NDNlMzk0NTdlOGU3YWFjN2EyN2QzYjgxNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzRmOTc5NTQzZTM5NDU3ZThlN2FhYzdhMjdkM2I4MTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhBUk1BTiBEUklWRSwgMjI2MjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZmQzZTRmYzU5NDk0Nzg5YTBjOTQ3NzViZWRkYzI1Ny5zZXRDb250ZW50KGh0bWxfYzRmOTc5NTQzZTM5NDU3ZThlN2FhYzdhMjdkM2I4MTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGNiNjNmZjY2MGRiNDM0ZDgxNDI5MmIxMjA2OWI5OTYuYmluZFBvcHVwKHBvcHVwXzBmZDNlNGZjNTk0OTQ3ODlhMGM5NDc3NWJlZGRjMjU3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Y4ZGZlYTYwNWRhMzQzODdhZjcyNjYxYjY4NjhiZmY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTMzLjgyMzk3NjUsMTUxLjAxMDUxMjJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjM4ZWZkYTM4ODQxNDA0MTk3NjE5MDcwN2RiZDY0M2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjA0ZjE5OTNhZmYxNDY1Mzk1NjBiMThhNTIyNmVkMzggPSAkKCc8ZGl2IGlkPSJodG1sXzIwNGYxOTkzYWZmMTQ2NTM5NTYwYjE4YTUyMjZlZDM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IQVJSSVMgU1RSRUVULCAyNDcxNzY5LjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIzOGVmZGEzODg0MTQwNDE5NzYxOTA3MDdkYmQ2NDNiLnNldENvbnRlbnQoaHRtbF8yMDRmMTk5M2FmZjE0NjUzOTU2MGIxOGE1MjI2ZWQzOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOGRmZWE2MDVkYTM0Mzg3YWY3MjY2MWI2ODY4YmZmOC5iaW5kUG9wdXAocG9wdXBfMjM4ZWZkYTM4ODQxNDA0MTk3NjE5MDcwN2RiZDY0M2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzUwMDE3OGMzNjJiNGE1MWIwMjE5OGRlYzdmOTJhZjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTk2MzI2LC0wLjAyMjk3NDZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDVmMjJlNDJmYjVmNGRlY2EyMmMxMDIzOTlhZTg5NTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTcyNTNmNzI1M2U1NDExNWE0ZjZjYzQ2ZjNjNjMxMGUgPSAkKCc8ZGl2IGlkPSJodG1sX2E3MjUzZjcyNTNlNTQxMTVhNGY2Y2M0NmYzYzYzMTBlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IQVZBTk5BSCBTVFJFRVQsIDIyMTczMDkuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDVmMjJlNDJmYjVmNGRlY2EyMmMxMDIzOTlhZTg5NTMuc2V0Q29udGVudChodG1sX2E3MjUzZjcyNTNlNTQxMTVhNGY2Y2M0NmYzYzYzMTBlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM1MDAxNzhjMzYyYjRhNTFiMDIxOThkZWM3ZjkyYWY4LmJpbmRQb3B1cChwb3B1cF80NWYyMmU0MmZiNWY0ZGVjYTIyYzEwMjM5OWFlODk1Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMzhkYjJmY2Q4ZWU0ZDEyOWM4MTZjZjJiNTdhMjNlZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ1OTQzMiwtMC4yMjcxNTAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYzYWQ5ODZmNDRlMTQ4MGI5NDEyMzg1MjNkMDg4N2Y4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxNzFjMTQ4NDJiZTQwOTg4ODJhNzU3YTY2MjhhYTY2ID0gJCgnPGRpdiBpZD0iaHRtbF8zMTcxYzE0ODQyYmU0MDk4ODgyYTc1N2E2NjI4YWE2NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SEFaTEVXRUxMIFJPQUQsIDI1MDAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjNhZDk4NmY0NGUxNDgwYjk0MTIzODUyM2QwODg3Zjguc2V0Q29udGVudChodG1sXzMxNzFjMTQ4NDJiZTQwOTg4ODJhNzU3YTY2MjhhYTY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EzOGRiMmZjZDhlZTRkMTI5YzgxNmNmMmI1N2EyM2VmLmJpbmRQb3B1cChwb3B1cF82M2FkOTg2ZjQ0ZTE0ODBiOTQxMjM4NTIzZDA4ODdmOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMWU5NDYxZTQzMjQ0MmFlODE2NGM3NjhjNGMzNDA0NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUxNTUwMTIsLTAuMTkzNTA5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xODUzOWEyNGRjM2Q0ODJkYmMwZDg2ZWQxMTk2OThhMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNTU0NmI1ZTM5NDQ0YmFjOTExOWVmOWJlN2RlODliNCA9ICQoJzxkaXYgaWQ9Imh0bWxfMjU1NDZiNWUzOTQ0NGJhYzkxMTllZjliZTdkZTg5YjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhFUkVGT1JEIE1FV1MsIDIzMTAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTg1MzlhMjRkYzNkNDgyZGJjMGQ4NmVkMTE5Njk4YTMuc2V0Q29udGVudChodG1sXzI1NTQ2YjVlMzk0NDRiYWM5MTE5ZWY5YmU3ZGU4OWI0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MxZTk0NjFlNDMyNDQyYWU4MTY0Yzc2OGM0YzM0MDQ3LmJpbmRQb3B1cChwb3B1cF8xODUzOWEyNGRjM2Q0ODJkYmMwZDg2ZWQxMTk2OThhMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NmU4ODg2ZDgyYTM0NDgxYWMyOTEyZDZmZTcyZjNlMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0MzY3NzUsLTAuMTc0NjI4M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YTNiYTA3MjE1M2Q0MWM3YmZkNzZjY2ViNjhjM2RlNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMDRkMmJhMGZiMzQ0MjliODJiYjgzMmY5Yjc0MWFkZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzA0ZDJiYTBmYjM0NDI5YjgyYmI4MzJmOWI3NDFhZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhFUk9OREFMRSBBVkVOVUUsIDI0NzUwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmEzYmEwNzIxNTNkNDFjN2JmZDc2Y2NlYjY4YzNkZTcuc2V0Q29udGVudChodG1sX2MwNGQyYmEwZmIzNDQyOWI4MmJiODMyZjliNzQxYWRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2ZTg4ODZkODJhMzQ0ODFhYzI5MTJkNmZlNzJmM2UxLmJpbmRQb3B1cChwb3B1cF82YTNiYTA3MjE1M2Q0MWM3YmZkNzZjY2ViNjhjM2RlNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZjExMjg2Njc4Y2Q0ZTM0OWQyZTNhM2I4YTRiMDcyOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU3MTA0MTksLTAuMTQ4OTgzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNWI5MjQwMzczZWE0ZmM3OGRhYzgzMTdmZWVjZTg0YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzQ1NjczOGRiNmM0MjQ2YTJiODBiZGJhMGU3YzQxNiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzM0NTY3MzhkYjZjNDI0NmEyYjgwYmRiYTBlN2M0MTYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhJR0hHQVRFIEhJR0ggU1RSRUVULCAyMjExMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM1YjkyNDAzNzNlYTRmYzc4ZGFjODMxN2ZlZWNlODRiLnNldENvbnRlbnQoaHRtbF83MzQ1NjczOGRiNmM0MjQ2YTJiODBiZGJhMGU3YzQxNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lZjExMjg2Njc4Y2Q0ZTM0OWQyZTNhM2I4YTRiMDcyOS5iaW5kUG9wdXAocG9wdXBfMzViOTI0MDM3M2VhNGZjNzhkYWM4MzE3ZmVlY2U4NGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTJhY2EwOWFmNjUzNGM2OWE4NjJiZGE0ZDE0MjdkMDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS42MzI3MjY3LC0wLjI0MDQ4OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzk1ZWJiYjUxMGRiNGY2MWIxMDYxMGJmYzQ5MmM1NWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjhhMzM1N2IwMGVmNDc2OWE3ZGYwNjMzNzFhZTNiMTUgPSAkKCc8ZGl2IGlkPSJodG1sX2I4YTMzNTdiMDBlZjQ3NjlhN2RmMDYzMzcxYWUzYjE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ISUdIV09PRCBISUxMLCAyMjUyNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc5NWViYmI1MTBkYjRmNjFiMTA2MTBiZmM0OTJjNTVmLnNldENvbnRlbnQoaHRtbF9iOGEzMzU3YjAwZWY0NzY5YTdkZjA2MzM3MWFlM2IxNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMmFjYTA5YWY2NTM0YzY5YTg2MmJkYTRkMTQyN2QwNS5iaW5kUG9wdXAocG9wdXBfNzk1ZWJiYjUxMGRiNGY2MWIxMDYxMGJmYzQ5MmM1NWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTdhZGFkODUwNTI0NDRkODllOGU5ZGNlMzI4MTUyYjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDc4NDY3LC0wLjE5NzM5NjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDEyMDEzNjMxMmY1NDgyZjk5MzI4ZTczNTYyNzk2OWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjUyZjk4YTMxMjY3NGVlN2EyMDVhNTg2NTE3YjlkZjkgPSAkKCc8ZGl2IGlkPSJodG1sXzY1MmY5OGEzMTI2NzRlZTdhMjA1YTU4NjUxN2I5ZGY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ISUxMR0FURSBQTEFDRSwgMjIwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MTIwMTM2MzEyZjU0ODJmOTkzMjhlNzM1NjI3OTY5Zi5zZXRDb250ZW50KGh0bWxfNjUyZjk4YTMxMjY3NGVlN2EyMDVhNTg2NTE3YjlkZjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTdhZGFkODUwNTI0NDRkODllOGU5ZGNlMzI4MTUyYjQuYmluZFBvcHVwKHBvcHVwXzQxMjAxMzYzMTJmNTQ4MmY5OTMyOGU3MzU2Mjc5NjlmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjNGQ5NmZlMTRjNjQ3MmM4MGQ5NzM1MjIzZDk4ZWQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTY1MDk5NywtMC4yOTA2Mzg0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RlZWZkMGM0ZWZkYTRkNjViOTRjZDVkY2RlMTZmY2NjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0ZTY3MGM1Mjk0ZTQyYjBhNDEyNzk0YzNjZDNlMzQ4ID0gJCgnPGRpdiBpZD0iaHRtbF83NGU2NzBjNTI5NGU0MmIwYTQxMjc5NGMzY2QzZTM0OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SE9MTFlDUk9GVCBBVkVOVUUsIDIzNjEzNzUuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGVlZmQwYzRlZmRhNGQ2NWI5NGNkNWRjZGUxNmZjY2Muc2V0Q29udGVudChodG1sXzc0ZTY3MGM1Mjk0ZTQyYjBhNDEyNzk0YzNjZDNlMzQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNjNGQ5NmZlMTRjNjQ3MmM4MGQ5NzM1MjIzZDk4ZWQyLmJpbmRQb3B1cChwb3B1cF9kZWVmZDBjNGVmZGE0ZDY1Yjk0Y2Q1ZGNkZTE2ZmNjYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZDVjMWMzYjM0MTI0MzQ2YjhjNTg5YTliOWIzYjI1MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NjIxMTIsLTAuMTgzNzE4NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNzY3ZTBkNDU1ZWM0ZDVhOWJmMzY0NmUyNGI1YjkwOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YmZlYmFjYzUwN2E0OGU4OTNkODEzZTJmOTg2N2IwZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNWJmZWJhY2M1MDdhNDhlODkzZDgxM2UyZjk4NjdiMGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhPTExZV09PRCBNRVdTLCAyMzUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3NjdlMGQ0NTVlYzRkNWE5YmYzNjQ2ZTI0YjViOTA4LnNldENvbnRlbnQoaHRtbF81YmZlYmFjYzUwN2E0OGU4OTNkODEzZTJmOTg2N2IwZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ZDVjMWMzYjM0MTI0MzQ2YjhjNTg5YTliOWIzYjI1My5iaW5kUG9wdXAocG9wdXBfMzc2N2UwZDQ1NWVjNGQ1YTliZjM2NDZlMjRiNWI5MDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWJhMjFkNDZlZTIzNGY0YjkyYzY3ZTUzZDhlMjg0ZjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NTQzMjk2LC0wLjE2MjcwNzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzQwNmNlNzAwMjdlNGIyZjk1ZWM2MWM0YmNiOTE0MWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmY5ZjI1NmJkN2M0NGEwNzllNjk4ODMzYzE3ZGU3NmQgPSAkKCc8ZGl2IGlkPSJodG1sX2JmOWYyNTZiZDdjNDRhMDc5ZTY5ODgzM2MxN2RlNzZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IT05FWVdFTEwgUk9BRCwgMjI3ODMzMy4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NDA2Y2U3MDAyN2U0YjJmOTVlYzYxYzRiY2I5MTQxZC5zZXRDb250ZW50KGh0bWxfYmY5ZjI1NmJkN2M0NGEwNzllNjk4ODMzYzE3ZGU3NmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWJhMjFkNDZlZTIzNGY0YjkyYzY3ZTUzZDhlMjg0ZjcuYmluZFBvcHVwKHBvcHVwXzc0MDZjZTcwMDI3ZTRiMmY5NWVjNjFjNGJjYjkxNDFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNhN2QzN2M5NzFlODQwMWM5Y2UwODY3Njc4Y2M4ZmZjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgxNzY3OSwtMC4xODUyMzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzEwNWQwOTBlYjAzNDY4YzgzZjA0NmMyZTU3Y2YwM2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjkwZjllMGFmZWE3NDdmZThjN2U4N2VhZjhhNzFiNjcgPSAkKCc8ZGl2IGlkPSJodG1sXzY5MGY5ZTBhZmVhNzQ3ZmU4YzdlODdlYWY4YTcxYjY3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IT1JURU5TSUEgUk9BRCwgMjI3NTkxNi42NjY2NjY2NjY1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMTA1ZDA5MGViMDM0NjhjODNmMDQ2YzJlNTdjZjAzYi5zZXRDb250ZW50KGh0bWxfNjkwZjllMGFmZWE3NDdmZThjN2U4N2VhZjhhNzFiNjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2E3ZDM3Yzk3MWU4NDAxYzljZTA4Njc2NzhjYzhmZmMuYmluZFBvcHVwKHBvcHVwXzMxMDVkMDkwZWIwMzQ2OGM4M2YwNDZjMmU1N2NmMDNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg1NTNhNTAwOGI3NDRjYmM4ZjdhNTUyMjBjNjc5MDg0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI3NTc4NSwtMC4wODExODgxMTA3MTE3MjQ3M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZmMyZWMwYzllODM0M2EyODdmYjJlZGFhOTBmN2UxOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZjY4ODI4MTFlZWY0Mzk0OWYzYjE2ZTYxMTNhMGFhOSA9ICQoJzxkaXYgaWQ9Imh0bWxfYmY2ODgyODExZWVmNDM5NDlmM2IxNmU2MTEzYTBhYTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhPWFRPTiBTUVVBUkUsIDIyMzQyODUuNzE0Mjg1NzE0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZmMyZWMwYzllODM0M2EyODdmYjJlZGFhOTBmN2UxOS5zZXRDb250ZW50KGh0bWxfYmY2ODgyODExZWVmNDM5NDlmM2IxNmU2MTEzYTBhYTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODU1M2E1MDA4Yjc0NGNiYzhmN2E1NTIyMGM2NzkwODQuYmluZFBvcHVwKHBvcHVwXzlmYzJlYzBjOWU4MzQzYTI4N2ZiMmVkYWE5MGY3ZTE5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyMDJhNjE3NjhmMzQwM2FhZWRmYTdkZTJkZjM4OTM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNjUzNDY0NCwxLjI4NzU3M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOTNiNjI2OTE0MGY0YmIxOWY3YTRkM2Q0YmIyNzFlMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MWU2NzMzNDNhMjM0NzliOWRkODE0MzI0YWYxZjQyMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDFlNjczMzQzYTIzNDc5YjlkZDgxNDMyNGFmMWY0MjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhVTlRFUiBST0FELCAyMzAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E5M2I2MjY5MTQwZjRiYjE5ZjdhNGQzZDRiYjI3MWUzLnNldENvbnRlbnQoaHRtbF80MWU2NzMzNDNhMjM0NzliOWRkODE0MzI0YWYxZjQyMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MjAyYTYxNzY4ZjM0MDNhYWVkZmE3ZGUyZGYzODkzNS5iaW5kUG9wdXAocG9wdXBfYTkzYjYyNjkxNDBmNGJiMTlmN2E0ZDNkNGJiMjcxZTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTBmMjA5MzEzMjYxNDE1NDk4ZmEzYTFjMjk2MDAwZmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41NzY2MTE3LC0wLjE0NTMzNjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWY3YTEzYjY4YWM2NGQzODg3MTg0NWEyNjJhMzYzNTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWJlZTdjMDNmN2NmNDQyZDhiOTBiMDZjZjVhN2E0YzkgPSAkKCc8ZGl2IGlkPSJodG1sXzFiZWU3YzAzZjdjZjQ0MmQ4YjkwYjA2Y2Y1YTdhNGM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KQUNLU09OUyBMQU5FLCAyMzYyNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVmN2ExM2I2OGFjNjRkMzg4NzE4NDVhMjYyYTM2MzU0LnNldENvbnRlbnQoaHRtbF8xYmVlN2MwM2Y3Y2Y0NDJkOGI5MGIwNmNmNWE3YTRjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMGYyMDkzMTMyNjE0MTU0OThmYTNhMWMyOTYwMDBmZS5iaW5kUG9wdXAocG9wdXBfNWY3YTEzYjY4YWM2NGQzODg3MTg0NWEyNjJhMzYzNTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjRlOGRkNDI0NDg1NGY4M2FjMDFkNmY4MjhmZTE1ODIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OC4xOTc2MjgyLDE2LjMyMDEyODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTk1YThmNjg0ZDUyNDdkM2JmM2M4MWE4MzZjNjgxYjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTNjODg5NzdhN2UwNGZhMTkzNjQ3YzY5MmFhMmFlZjUgPSAkKCc8ZGl2IGlkPSJodG1sX2UzYzg4OTc3YTdlMDRmYTE5MzY0N2M2OTJhYTJhZWY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KT0hOIFNUUkVFVCwgMjIzNTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OTVhOGY2ODRkNTI0N2QzYmYzYzgxYTgzNmM2ODFiNC5zZXRDb250ZW50KGh0bWxfZTNjODg5NzdhN2UwNGZhMTkzNjQ3YzY5MmFhMmFlZjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjRlOGRkNDI0NDg1NGY4M2FjMDFkNmY4MjhmZTE1ODIuYmluZFBvcHVwKHBvcHVwXzk5NWE4ZjY4NGQ1MjQ3ZDNiZjNjODFhODM2YzY4MWI0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JlZGE4MTM2N2RjZTQ2ZmRhZjcwYzhkNTY5YTAwNGVmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAwNjgyMywtMC4xNTY3MTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2I4N2NmODc3NGI5NDA0OTgxNDIwNmNiMGJjNDM3YWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWQ1MGFmNTUzODg1NDUwNGFmY2VjMWFmMDJjOGJhN2UgPSAkKCc8ZGl2IGlkPSJodG1sX2VkNTBhZjU1Mzg4NTQ1MDRhZmNlYzFhZjAyYzhiYTdlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LSU5ORVJUT04gU1RSRUVULCAyNDg1NjAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdiODdjZjg3NzRiOTQwNDk4MTQyMDZjYjBiYzQzN2FjLnNldENvbnRlbnQoaHRtbF9lZDUwYWY1NTM4ODU0NTA0YWZjZWMxYWYwMmM4YmE3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iZWRhODEzNjdkY2U0NmZkYWY3MGM4ZDU2OWEwMDRlZi5iaW5kUG9wdXAocG9wdXBfN2I4N2NmODc3NGI5NDA0OTgxNDIwNmNiMGJjNDM3YWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGRmZjU0ODI2NDAwNDNmMmI4YTViODc0YWE2MTI5OTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTQ3NjI3LC0wLjE5MTExOTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzRkMzlkY2M5YzZjNGNiNzgzMDZiMDhjZGZmZDcyZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTczM2E3MDg0NDVlNGMyMzhkMTIwYWZjNWI1ODgwYzkgPSAkKCc8ZGl2IGlkPSJodG1sXzE3MzNhNzA4NDQ1ZTRjMjM4ZDEyMGFmYzViNTg4MGM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LTkFSRVNCT1JPVUdIIFBMQUNFLCAyMzI1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM0ZDM5ZGNjOWM2YzRjYjc4MzA2YjA4Y2RmZmQ3MmVlLnNldENvbnRlbnQoaHRtbF8xNzMzYTcwODQ0NWU0YzIzOGQxMjBhZmM1YjU4ODBjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZGZmNTQ4MjY0MDA0M2YyYjhhNWI4NzRhYTYxMjk5NC5iaW5kUG9wdXAocG9wdXBfMzRkMzlkY2M5YzZjNGNiNzgzMDZiMDhjZGZmZDcyZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzViZDczYTA0MzAzNGM2OWJjNzBlZWM1ZGFhZWY5NjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjA4ODY3LC0wLjE2MTQ1NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjBlYmEzMGE3Y2E5NDhhNjg2YzcyZjJlMDI0YWU4YjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWNiYWUxNmMxY2U0NDQ1Nzg0NzgxOGNjM2U5MzhlNGMgPSAkKCc8ZGl2IGlkPSJodG1sX2FjYmFlMTZjMWNlNDQ0NTc4NDc4MThjYzNlOTM4ZTRjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LTk9YIFNUUkVFVCwgMjI1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMGViYTMwYTdjYTk0OGE2ODZjNzJmMmUwMjRhZThiMS5zZXRDb250ZW50KGh0bWxfYWNiYWUxNmMxY2U0NDQ1Nzg0NzgxOGNjM2U5MzhlNGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzViZDczYTA0MzAzNGM2OWJjNzBlZWM1ZGFhZWY5NjYuYmluZFBvcHVwKHBvcHVwX2YwZWJhMzBhN2NhOTQ4YTY4NmM3MmYyZTAyNGFlOGIxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzExZjVhNDUxYTE4NTQ1MzRhMDUyYTJkYjNhYzJiZjUxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTE3MjYzOSwtMC4yMTExMDE5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y0ZWU2Zjc0OTMwNjRkZjlhYTQ0ZGE5YTY2NmZjZDhlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q3NzUwYjc5YWU0NjQ5ZTJhZTNjZDQwZTcyMDNjYjdiID0gJCgnPGRpdiBpZD0iaHRtbF9kNzc1MGI3OWFlNDY0OWUyYWUzY2Q0MGU3MjAzY2I3YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TEFEQlJPS0UgR1JPVkUsIDI0ODMzMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjRlZTZmNzQ5MzA2NGRmOWFhNDRkYTlhNjY2ZmNkOGUuc2V0Q29udGVudChodG1sX2Q3NzUwYjc5YWU0NjQ5ZTJhZTNjZDQwZTcyMDNjYjdiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzExZjVhNDUxYTE4NTQ1MzRhMDUyYTJkYjNhYzJiZjUxLmJpbmRQb3B1cChwb3B1cF9mNGVlNmY3NDkzMDY0ZGY5YWE0NGRhOWE2NjZmY2Q4ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wN2EzN2FmNTc1OTk0MDYxOGU1OTJkMDlmZjRiYTZjNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUxMjIzNjYsLTAuMTc4NzI5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jOWNhOWE2NzUzMDg0NmUzYWM4OTZhOTAyMzU2Yzk3MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iY2VjMmU3YTcwYmI0YTU1OThkN2U5MWNjODc4Y2YzNiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmNlYzJlN2E3MGJiNGE1NTk4ZDdlOTFjYzg3OGNmMzYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxBTkNBU1RFUiBNRVdTLCAyMzEyNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M5Y2E5YTY3NTMwODQ2ZTNhYzg5NmE5MDIzNTZjOTcyLnNldENvbnRlbnQoaHRtbF9iY2VjMmU3YTcwYmI0YTU1OThkN2U5MWNjODc4Y2YzNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wN2EzN2FmNTc1OTk0MDYxOGU1OTJkMDlmZjRiYTZjNy5iaW5kUG9wdXAocG9wdXBfYzljYTlhNjc1MzA4NDZlM2FjODk2YTkwMjM1NmM5NzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjBhMmUwMjgzNzM3NGMxYzkzZjU2MWQxNWNjYWFmZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1My4zMzUyMzMwOTk5OTk5OTYsLTYuMjI4MTc4NDcxNDY1MjI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIxOGNlZjJjZDk0NTRhZGNiZDMwYTNkN2M4YWU5ZGY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxNTFkOTYxZjQ5NjRiNjY4NDNlYjM5YTE1YTliZDBiID0gJCgnPGRpdiBpZD0iaHRtbF84MTUxZDk2MWY0OTY0YjY2ODQzZWIzOWExNWE5YmQwYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TEFOU0RPV05FIFJPQUQsIDIzNjQ4NzUuODUxODUxODUxNzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjE4Y2VmMmNkOTQ1NGFkY2JkMzBhM2Q3YzhhZTlkZjUuc2V0Q29udGVudChodG1sXzgxNTFkOTYxZjQ5NjRiNjY4NDNlYjM5YTE1YTliZDBiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwYTJlMDI4MzczNzRjMWM5M2Y1NjFkMTVjY2FhZmQyLmJpbmRQb3B1cChwb3B1cF8yMThjZWYyY2Q5NDU0YWRjYmQzMGEzZDdjOGFlOWRmNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NWM2NmM1MzZhNzY0NGNlYThjZjVhZjA1ZjY5NzA3MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjAzMDU0MzIsMS4yMTYyOTI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkwYTlhYmNiODlhYzQwNjRhOTIzMzljNjg2NTMyZDRiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxNDk4NTBlM2NjNTQ4ZjY4N2RlNmU5MDVlYWRiZjFmID0gJCgnPGRpdiBpZD0iaHRtbF84MTQ5ODUwZTNjYzU0OGY2ODdkZTZlOTA1ZWFkYmYxZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TEFUSU1FUiBJTkRVU1RSSUFMIEVTVEFURSwgMjQwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MGE5YWJjYjg5YWM0MDY0YTkyMzM5YzY4NjUzMmQ0Yi5zZXRDb250ZW50KGh0bWxfODE0OTg1MGUzY2M1NDhmNjg3ZGU2ZTkwNWVhZGJmMWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTVjNjZjNTM2YTc2NDRjZWE4Y2Y1YWYwNWY2OTcwNzEuYmluZFBvcHVwKHBvcHVwXzkwYTlhYmNiODlhYzQwNjRhOTIzMzljNjg2NTMyZDRiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzliMzNjY2RhNTBhZTRmOWFiYjA1ZDU3MmJmNDQxZmM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI1NjY3OCwtMC4xNDIxMzUxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMxMWFlMzJjYTk0ODQxNzM4NmIzMDU4YjQ3YTNmZTlhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg2ZDg0NjExMDU3ZTQ1ZDdiMTdkYzI2ZGE0NGVjNjM2ID0gJCgnPGRpdiBpZD0iaHRtbF84NmQ4NDYxMTA1N2U0NWQ3YjE3ZGMyNmRhNDRlYzYzNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TEFYVE9OIFBMQUNFLCAyNTAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxMWFlMzJjYTk0ODQxNzM4NmIzMDU4YjQ3YTNmZTlhLnNldENvbnRlbnQoaHRtbF84NmQ4NDYxMTA1N2U0NWQ3YjE3ZGMyNmRhNDRlYzYzNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YjMzY2NkYTUwYWU0ZjlhYmIwNWQ1NzJiZjQ0MWZjNS5iaW5kUG9wdXAocG9wdXBfMzExYWUzMmNhOTQ4NDE3Mzg2YjMwNThiNDdhM2ZlOWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWY0ZWIwMGQ2YzI5NDhlZWFkODRiZDc5MTU1OWJhYWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zMDQ5Mjk3LC0xMjEuODk4NDk2MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNjI3NTI2NjNhODE0MDQ1YWYxNDI2N2JkY2MzZjJiMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzEzZDU3ZDdmODU0ZDRmYjAxNjVmNzE5OGE5YTMyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzMxM2Q1N2Q3Zjg1NGQ0ZmIwMTY1ZjcxOThhOWEzMjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxJTkNPTE4gQVZFTlVFLCAyMjAzNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA2Mjc1MjY2M2E4MTQwNDVhZjE0MjY3YmRjYzNmMmIwLnNldENvbnRlbnQoaHRtbF83MzEzZDU3ZDdmODU0ZDRmYjAxNjVmNzE5OGE5YTMyMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZjRlYjAwZDZjMjk0OGVlYWQ4NGJkNzkxNTU5YmFhYy5iaW5kUG9wdXAocG9wdXBfMDYyNzUyNjYzYTgxNDA0NWFmMTQyNjdiZGNjM2YyYjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjJiYjA4OWE1OGU1NDUwODk3Nzk2YjdkNjY3NzY4OTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi4wODM4NzQsMS4xNDM5ODY1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I4MzM2MzhhODdhYzQ3MTU5Nzc4YjY4MWNkYjk2ZGEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVkODM1ZjE3YTQ3YjRhN2U4ZWQ1Mzc2OTBiNGE2ODE2ID0gJCgnPGRpdiBpZD0iaHRtbF81ZDgzNWYxN2E0N2I0YTdlOGVkNTM3NjkwYjRhNjgxNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TElOR0ZJRUxEIFJPQUQsIDIyNDg3NTAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjgzMzYzOGE4N2FjNDcxNTk3NzhiNjgxY2RiOTZkYTAuc2V0Q29udGVudChodG1sXzVkODM1ZjE3YTQ3YjRhN2U4ZWQ1Mzc2OTBiNGE2ODE2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyYmIwODlhNThlNTQ1MDg5Nzc5NmI3ZDY2Nzc2ODkyLmJpbmRQb3B1cChwb3B1cF9iODMzNjM4YTg3YWM0NzE1OTc3OGI2ODFjZGI5NmRhMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZmNlOTU2OTk1YTM0ZDM5OGNjMjUyMTBlYjYxZDQ5MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUyMTU0NzQsLTAuMTY4MjU2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NjE1YzJmZWZhNjU0NDIyOTNmZTBjOTYxOTA3YzkzZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZDIyZDg1MTI0ODQ0YjZhYTU2YWRkMzIyN2JmOWNiZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZWQyMmQ4NTEyNDg0NGI2YWE1NmFkZDMyMjdiZjljYmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxJU1NPTiBTVFJFRVQsIDI0NjI1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTYxNWMyZmVmYTY1NDQyMjkzZmUwYzk2MTkwN2M5M2Yuc2V0Q29udGVudChodG1sX2VkMjJkODUxMjQ4NDRiNmFhNTZhZGQzMjI3YmY5Y2JkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRmY2U5NTY5OTVhMzRkMzk4Y2MyNTIxMGViNjFkNDkxLmJpbmRQb3B1cChwb3B1cF81NjE1YzJmZWZhNjU0NDIyOTNmZTBjOTYxOTA3YzkzZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jOTcwZjg5MDBlNGQ0MDI2YTg2OGVkMDI5ZGM0NjBhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NjM1MTMsLTAuMDkyMDEyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMWJlNDcxYjAwMTc0M2IzOTA4MWY1Yjc2ZjY0YTRmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZjA5NWI1Y2MzN2E0OWFiYjM3ZTZkZjMwYzkxN2Y4OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZmYwOTViNWNjMzdhNDlhYmIzN2U2ZGYzMGM5MTdmODkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxJVkVSUE9PTCBHUk9WRSwgMjI4ODAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMWJlNDcxYjAwMTc0M2IzOTA4MWY1Yjc2ZjY0YTRmNy5zZXRDb250ZW50KGh0bWxfZmYwOTViNWNjMzdhNDlhYmIzN2U2ZGYzMGM5MTdmODkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzk3MGY4OTAwZTRkNDAyNmE4NjhlZDAyOWRjNDYwYTEuYmluZFBvcHVwKHBvcHVwXzMxYmU0NzFiMDAxNzQzYjM5MDgxZjViNzZmNjRhNGY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcxMzc0NTc3ZGFmYzRmOGJiMDYxZjJlZGNhNWJjMThmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDUxNjAyMSwtMC4yMzgyNzc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM3N2RiNDk4ZWI0YzRlMTdiMzIzNWIyZjRmMzdhYmQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5MmQ5NDE4YWQ2MTRmNmJiMTRhYjQwM2RmZmQ2OTJiID0gJCgnPGRpdiBpZD0iaHRtbF9iOTJkOTQxOGFkNjE0ZjZiYjE0YWI0MDNkZmZkNjkyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TE9OR1dPT0QgRFJJVkUsIDIzNzUwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzc3ZGI0OThlYjRjNGUxN2IzMjM1YjJmNGYzN2FiZDkuc2V0Q29udGVudChodG1sX2I5MmQ5NDE4YWQ2MTRmNmJiMTRhYjQwM2RmZmQ2OTJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcxMzc0NTc3ZGFmYzRmOGJiMDYxZjJlZGNhNWJjMThmLmJpbmRQb3B1cChwb3B1cF8zNzdkYjQ5OGViNGM0ZTE3YjMyMzViMmY0ZjM3YWJkOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNDY1ZDc3OTMzYTQ0YTlkOWM4OGFiNTIxZTE5MjUyMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUzOTU4NDg5OTk5OTk5NCwtMC4xMDgxNzY1MTIyMTE2ODE5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA1YjU2ZWEwNjE2MjQ0Zjk4OGI4NWI4M2QxM2E3ZjU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhMDFjYmYxNWIyODQwYjQ5N2Q1M2ZiMTM2NTJhNzcxID0gJCgnPGRpdiBpZD0iaHRtbF9mYTAxY2JmMTViMjg0MGI0OTdkNTNmYjEzNjUyYTc3MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TE9OU0RBTEUgU1FVQVJFLCAyMzU3NTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA1YjU2ZWEwNjE2MjQ0Zjk4OGI4NWI4M2QxM2E3ZjU0LnNldENvbnRlbnQoaHRtbF9mYTAxY2JmMTViMjg0MGI0OTdkNTNmYjEzNjUyYTc3MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNDY1ZDc3OTMzYTQ0YTlkOWM4OGFiNTIxZTE5MjUyMi5iaW5kUG9wdXAocG9wdXBfMDViNTZlYTA2MTYyNDRmOTg4Yjg1YjgzZDEzYTdmNTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzNmOGEzNTg1NDA0NDg3YWFlNTFlZDQ4MmM4MzNhZmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS4zNTMzOTYzLC03OS4xNTAzMTc2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc4NzA3ZWZhZjQyMDRkNWNhYTA0MTliNWE5ZTIxOGVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk2ZjYzMTBhZDZlNDQzMTQ4MzNjYWFkZGYzMjhlOGFjID0gJCgnPGRpdiBpZD0iaHRtbF85NmY2MzEwYWQ2ZTQ0MzE0ODMzY2FhZGRmMzI4ZThhYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TUFaRSBISUxMLCAyMjUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4NzA3ZWZhZjQyMDRkNWNhYTA0MTliNWE5ZTIxOGVjLnNldENvbnRlbnQoaHRtbF85NmY2MzEwYWQ2ZTQ0MzE0ODMzY2FhZGRmMzI4ZThhYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83M2Y4YTM1ODU0MDQ0ODdhYWU1MWVkNDgyYzgzM2FmZC5iaW5kUG9wdXAocG9wdXBfNzg3MDdlZmFmNDIwNGQ1Y2FhMDQxOWI1YTllMjE4ZWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmI0MTZjMzg3MjY5NDA3ZjkxM2UzNTM0MzIyMzQ3MDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTgyODMxLC0wLjA5OTE5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzkyM2FhZWYyMTA1NGMzNTk3MDk4Y2IxMzI3YzllNWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjYwMThkZWEzODQyNDU4M2E5YWQ5NGU2OTMyOGVlNGIgPSAkKCc8ZGl2IGlkPSJodG1sXzI2MDE4ZGVhMzg0MjQ1ODNhOWFkOTRlNjkzMjhlZTRiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NSURETEVTRVggUEFTU0FHRSwgMjI4MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83OTIzYWFlZjIxMDU0YzM1OTcwOThjYjEzMjdjOWU1Yy5zZXRDb250ZW50KGh0bWxfMjYwMThkZWEzODQyNDU4M2E5YWQ5NGU2OTMyOGVlNGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmI0MTZjMzg3MjY5NDA3ZjkxM2UzNTM0MzIyMzQ3MDEuYmluZFBvcHVwKHBvcHVwXzc5MjNhYWVmMjEwNTRjMzU5NzA5OGNiMTMyN2M5ZTVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU2YWE0OWRiMzUxYjQzMjBiNmU3NWE0OTJlNmRmZDY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuMzQxMDgyNiwxLjAyMTE3NDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTE0ZWNlM2Q3NzA2NGE5Yjk1NDc5MzU2NDg0ZjI3ZmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWM1MWM5ZjZiMjVhNGVlZjgyNmJlMGQ3ZThkNzRlNDEgPSAkKCc8ZGl2IGlkPSJodG1sXzVjNTFjOWY2YjI1YTRlZWY4MjZiZTBkN2U4ZDc0ZTQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NT05UUEVMSUVSIEFWRU5VRSwgMjUwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMTRlY2UzZDc3MDY0YTliOTU0NzkzNTY0ODRmMjdmYi5zZXRDb250ZW50KGh0bWxfNWM1MWM5ZjZiMjVhNGVlZjgyNmJlMGQ3ZThkNzRlNDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTZhYTQ5ZGIzNTFiNDMyMGI2ZTc1YTQ5MmU2ZGZkNjcuYmluZFBvcHVwKHBvcHVwX2ExNGVjZTNkNzcwNjRhOWI5NTQ3OTM1NjQ4NGYyN2ZiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc5YjhiMTBhMTg5NzQzMDZiYTQ1NTA2MWE4ZjhkYmU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk4ODkzOCwtMC4xNjcxNzM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU3OTA2NjBkMWM3YTQxYzZhZjBkNzczNWIxYTk4ZmEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhZWFlYjU4ODc4YzRlOTVhODdkNTk0MWJiMmQ0NzcyID0gJCgnPGRpdiBpZD0iaHRtbF9hYWVhZWI1ODg3OGM0ZTk1YTg3ZDU5NDFiYjJkNDc3MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TU9OVFBFTElFUiBXQUxLLCAyMzIwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU3OTA2NjBkMWM3YTQxYzZhZjBkNzczNWIxYTk4ZmEzLnNldENvbnRlbnQoaHRtbF9hYWVhZWI1ODg3OGM0ZTk1YTg3ZDU5NDFiYjJkNDc3Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OWI4YjEwYTE4OTc0MzA2YmE0NTUwNjFhOGY4ZGJlOS5iaW5kUG9wdXAocG9wdXBfNTc5MDY2MGQxYzdhNDFjNmFmMGQ3NzM1YjFhOThmYTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjI0MDk1YjY1Y2EwNDA0Mzg0MjZmODE2NWU5OTVkZmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NDY1MDMxLC0wLjE3NjE3MzFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDIyMmUxMmUwZjhhNGRhMGE1Y2FmYTA1MTljMzY1OTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDE1YWI0ZmM5MmI2NDhlM2FjNzc0NTE2OGY1NGZlNGEgPSAkKCc8ZGl2IGlkPSJodG1sXzAxNWFiNGZjOTJiNjQ4ZTNhYzc3NDUxNjhmNTRmZTRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NVUxUT04gUk9BRCwgMjMwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMjIyZTEyZTBmOGE0ZGEwYTVjYWZhMDUxOWMzNjU5MS5zZXRDb250ZW50KGh0bWxfMDE1YWI0ZmM5MmI2NDhlM2FjNzc0NTE2OGY1NGZlNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjI0MDk1YjY1Y2EwNDA0Mzg0MjZmODE2NWU5OTVkZmQuYmluZFBvcHVwKHBvcHVwX2QyMjJlMTJlMGY4YTRkYTBhNWNhZmEwNTE5YzM2NTkxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAyMWQ2YmRhY2VjODRjOGViODM5YjY0ZGE5NTAyNDA4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk0Mjc1NCwtMC4yMTI0OTM0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FlYTExNzZkYTk2YzRjYzU4ZjNkMTk3NmQ1MmE2OTA2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q5Y2FiZTkzZDQ0YjQ4OWZiMmFhM2JjNTFiMGUxODI3ID0gJCgnPGRpdiBpZD0iaHRtbF9kOWNhYmU5M2Q0NGI0ODlmYjJhYTNiYzUxYjBlMTgyNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TVVOREVOIFNUUkVFVCwgMjI1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZWExMTc2ZGE5NmM0Y2M1OGYzZDE5NzZkNTJhNjkwNi5zZXRDb250ZW50KGh0bWxfZDljYWJlOTNkNDRiNDg5ZmIyYWEzYmM1MWIwZTE4MjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDIxZDZiZGFjZWM4NGM4ZWI4MzliNjRkYTk1MDI0MDguYmluZFBvcHVwKHBvcHVwX2FlYTExNzZkYTk2YzRjYzU4ZjNkMTk3NmQ1MmE2OTA2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NjY2VlOTZkNjcxZTQyOTE4NTg1ZWY2NTNkMWQ0MjdjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTM0Ljg0NTc5NzksMTQ5LjM5MDUyNzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjUwMzcxODlmNDE4NGEzMmIyNGYwYTgyNzJiYTk2MzkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzdjNjE5NTVmYWVhNDk5Y2I1YWNiZDRhNTFmNDM2Y2MgPSAkKCc8ZGl2IGlkPSJodG1sXzM3YzYxOTU1ZmFlYTQ5OWNiNWFjYmQ0YTUxZjQzNmNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OT1JGT0xLIENSRVNDRU5ULCAyMjIzMzMzLjMzMzMzMzMzMzU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I1MDM3MTg5ZjQxODRhMzJiMjRmMGE4MjcyYmE5NjM5LnNldENvbnRlbnQoaHRtbF8zN2M2MTk1NWZhZWE0OTljYjVhY2JkNGE1MWY0MzZjYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jY2NlZTk2ZDY3MWU0MjkxODU4NWVmNjUzZDFkNDI3Yy5iaW5kUG9wdXAocG9wdXBfYjUwMzcxODlmNDE4NGEzMmIyNGYwYTgyNzJiYTk2MzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTM0Yzk1NTJlNjhlNDM5YTk4ZWNiOGMzNzRjNWZjMDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41OTk3Nzk0LC0wLjAwMDU2NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmNmYjgwNzg4NWMyNDRhNzk1NTg5MDlhNGZkNTc4OTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2RiM2E2N2U3NDI1NDJiZGFhZjgwNWQxM2FhMjdkMTMgPSAkKCc8ZGl2IGlkPSJodG1sXzdkYjNhNjdlNzQyNTQyYmRhYWY4MDVkMTNhYTI3ZDEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OT1JUSCBDSVJDVUxBUiBST0FELCAyMzkzMTQyLjg1NzE0Mjg1NzM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZjZmI4MDc4ODVjMjQ0YTc5NTU4OTA5YTRmZDU3ODkzLnNldENvbnRlbnQoaHRtbF83ZGIzYTY3ZTc0MjU0MmJkYWFmODA1ZDEzYWEyN2QxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MzRjOTU1MmU2OGU0MzlhOThlY2I4YzM3NGM1ZmMwOS5iaW5kUG9wdXAocG9wdXBfNmNmYjgwNzg4NWMyNDRhNzk1NTg5MDlhNGZkNTc4OTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODk3YTBhYmY5MTYwNGUzMWE0ZDA1ZWNlMTBiOWYzY2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjEzMjMxLC0wLjE1MjUzNThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzcxNzk5ZmQzOTJlNDU0NDhhODdiYzkyYTcwZjg2ODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjUyYzBmZWNlMGU4NGU4ZmE4NjY0OTkzZTVhOTNjMDggPSAkKCc8ZGl2IGlkPSJodG1sXzI1MmMwZmVjZTBlODRlOGZhODY2NDk5M2U1YTkzYzA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OT1RUSU5HSEFNIFNUUkVFVCwgMjIyNzUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNzE3OTlmZDM5MmU0NTQ0OGE4N2JjOTJhNzBmODY4NS5zZXRDb250ZW50KGh0bWxfMjUyYzBmZWNlMGU4NGU4ZmE4NjY0OTkzZTVhOTNjMDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODk3YTBhYmY5MTYwNGUzMWE0ZDA1ZWNlMTBiOWYzY2MuYmluZFBvcHVwKHBvcHVwX2M3MTc5OWZkMzkyZTQ1NDQ4YTg3YmM5MmE3MGY4Njg1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYxZmEwNDFkNjE1MDRjYjE4YjZhMDYxNTI4YWE1NTQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgzNjQ4MiwtMC4xNjczMjQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNjZDFkYTU5NTNiMTRkZWY5OTkyMDAyMGFmNDZiNjA2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFmNzc4YmZmODVjYjQ2NjM5ZWVhMjQ0YzhjZTg0YWViID0gJCgnPGRpdiBpZD0iaHRtbF8xZjc3OGJmZjg1Y2I0NjYzOWVlYTI0NGM4Y2U4NGFlYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0FLTEVZIFNUUkVFVCwgMjM4MTI4NS43MTQyODU3MTQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNjZDFkYTU5NTNiMTRkZWY5OTkyMDAyMGFmNDZiNjA2LnNldENvbnRlbnQoaHRtbF8xZjc3OGJmZjg1Y2I0NjYzOWVlYTI0NGM4Y2U4NGFlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MWZhMDQxZDYxNTA0Y2IxOGI2YTA2MTUyOGFhNTU0Ni5iaW5kUG9wdXAocG9wdXBfM2NkMWRhNTk1M2IxNGRlZjk5OTIwMDIwYWY0NmI2MDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjViNTkyMGM4ZTI4NDAxYTk1MjBkYmM3NGQ5ZThjMmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NS41MDU1NTU2LC05Mi45Nzk0NDQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MyMzRiY2NhMDM3ZjRjN2I4ODBmNjMxMDc2MGMxMmJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IwZTE3NjhlMWNkYzRlOTk4NjVmNTkwZGRjNTk2YTg4ID0gJCgnPGRpdiBpZD0iaHRtbF9iMGUxNzY4ZTFjZGM0ZTk5ODY1ZjU5MGRkYzU5NmE4OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0FLV09PRCBDT1VSVCwgMjM0ODc1MC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMjM0YmNjYTAzN2Y0YzdiODgwZjYzMTA3NjBjMTJiZC5zZXRDb250ZW50KGh0bWxfYjBlMTc2OGUxY2RjNGU5OTg2NWY1OTBkZGM1OTZhODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjViNTkyMGM4ZTI4NDAxYTk1MjBkYmM3NGQ5ZThjMmMuYmluZFBvcHVwKHBvcHVwX2MyMzRiY2NhMDM3ZjRjN2I4ODBmNjMxMDc2MGMxMmJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3YmJjNjdkNDljOTRkNjhhYjU4ODdhZjljNGZkZmI3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAzODU2NSwtMC4xOTYwMzQxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQzODVjZTRhMDlhNDQ2NGRhM2U5MGVlZWZmOTVkMTJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI0OTY1MGVjZWQ2YzQxZjY5YzQ5YzkwMjllZWJmZDY4ID0gJCgnPGRpdiBpZD0iaHRtbF8yNDk2NTBlY2VkNmM0MWY2OWM0OWM5MDI5ZWViZmQ2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0JTRVJWQVRPUlkgR0FSREVOUywgMjQyMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Mzg1Y2U0YTA5YTQ0NjRkYTNlOTBlZWVmZjk1ZDEyZS5zZXRDb250ZW50KGh0bWxfMjQ5NjUwZWNlZDZjNDFmNjljNDljOTAyOWVlYmZkNjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzdiYmM2N2Q0OWM5NGQ2OGFiNTg4N2FmOWM0ZmRmYjcuYmluZFBvcHVwKHBvcHVwXzQzODVjZTRhMDlhNDQ2NGRhM2U5MGVlZWZmOTVkMTJlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlmZWE3YmYxMzQzZTQ0M2Y4N2Y2NzI3NzIxYWQ3NzE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAyODY4NSwtMC4xOTExNzA2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EyYWUwNjgyZjdkMTQ3YzNhZGU1ZDBjM2ZmZmYzMzI0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIwZGI5OWQ1NmRhMTQ5ODdiNmJjMTJkZDFiM2I4MzJiID0gJCgnPGRpdiBpZD0iaHRtbF8yMGRiOTlkNTZkYTE0OTg3YjZiYzEyZGQxYjNiODMyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0xEIENPVVJUIFBMQUNFLCAyMzk1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EyYWUwNjgyZjdkMTQ3YzNhZGU1ZDBjM2ZmZmYzMzI0LnNldENvbnRlbnQoaHRtbF8yMGRiOTlkNTZkYTE0OTg3YjZiYzEyZGQxYjNiODMyYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZmVhN2JmMTM0M2U0NDNmODdmNjcyNzcyMWFkNzcxNy5iaW5kUG9wdXAocG9wdXBfYTJhZTA2ODJmN2QxNDdjM2FkZTVkMGMzZmZmZjMzMjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjUxYzdjYWExNTc5NGY1M2FlNTE4NjE5MDc5MGZmMzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTE0ODIyLC0wLjE3NzI1NTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTIxNDdmZGNlMzU2NGVjYTllOGRjMDRhNzc5ZGJlZTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzhjNzYzMGVhYzhlNDNhODhjNDViNzM3MzY5ZjMwOGMgPSAkKCc8ZGl2IGlkPSJodG1sXzM4Yzc2MzBlYWM4ZTQzYTg4YzQ1YjczNzM2OWYzMDhjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PTlNMT1cgTUVXUyBXRVNULCAyMzAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkyMTQ3ZmRjZTM1NjRlY2E5ZThkYzA0YTc3OWRiZWU5LnNldENvbnRlbnQoaHRtbF8zOGM3NjMwZWFjOGU0M2E4OGM0NWI3MzczNjlmMzA4Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNTFjN2NhYTE1Nzk0ZjUzYWU1MTg2MTkwNzkwZmYzNy5iaW5kUG9wdXAocG9wdXBfOTIxNDdmZGNlMzU2NGVjYTllOGRjMDRhNzc5ZGJlZTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTAzNjFjNzNmYTY0NDVmMzg5M2U5YmI4YTQ0ZWNmMDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS43MTE2NjgsLTcwLjQ3NzQxMzk5NDA3MTE0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I2YzE1YTE5YzU2MzQxMTJiNzA3ZTIyMDg5Zjk0ODFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZiNzVkZjM2MmFlZDQzMzU4MTIyMTkzMDA4OWYwYjJiID0gJCgnPGRpdiBpZD0iaHRtbF82Yjc1ZGYzNjJhZWQ0MzM1ODEyMjE5MzAwODlmMGIyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFMQUNFIFBMQUNFLCAyMzAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I2YzE1YTE5YzU2MzQxMTJiNzA3ZTIyMDg5Zjk0ODFkLnNldENvbnRlbnQoaHRtbF82Yjc1ZGYzNjJhZWQ0MzM1ODEyMjE5MzAwODlmMGIyYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMDM2MWM3M2ZhNjQ0NWYzODkzZTliYjhhNDRlY2YwMS5iaW5kUG9wdXAocG9wdXBfYjZjMTVhMTljNTYzNDExMmI3MDdlMjIwODlmOTQ4MWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDNjODcyNjZiMDI4NDdhZjkyMDg0YTFiNjJjYzllOGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDkzMTA2LC0wLjEzMjIwODVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGY0MmIwY2UzYTllNDA5MWI0ZWJmMDg1ZTUxYzg0NWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzY1OTIzNzQ5YTMyNDQxMzllZWNlNWNhMjgzMTdmZDEgPSAkKCc8ZGl2IGlkPSJodG1sXzM2NTkyMzc0OWEzMjQ0MTM5ZWVjZTVjYTI4MzE3ZmQxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QQU5UT04gU1RSRUVULCAyNDc1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBmNDJiMGNlM2E5ZTQwOTFiNGViZjA4NWU1MWM4NDVmLnNldENvbnRlbnQoaHRtbF8zNjU5MjM3NDlhMzI0NDEzOWVlY2U1Y2EyODMxN2ZkMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kM2M4NzI2NmIwMjg0N2FmOTIwODRhMWI2MmNjOWU4Yy5iaW5kUG9wdXAocG9wdXBfMGY0MmIwY2UzYTllNDA5MWI0ZWJmMDg1ZTUxYzg0NWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDM5ZTBlNDM0ZjllNDFhZWI2NTVlYTI5YWU1Y2Y0NzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszOS43MzA5NDUxLC03NS43MDQwOTkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAxYjJkZDMyOGViZDQ3ZThhNTVkMzNmNTIwMmJkZTc4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhiNTYyMmI4MTZkZjQwNzU5ZmEwZGVkYjZhYWY0ZDY4ID0gJCgnPGRpdiBpZD0iaHRtbF84YjU2MjJiODE2ZGY0MDc1OWZhMGRlZGI2YWFmNGQ2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSSyBDUkVTQ0VOVCwgMjQ0MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMWIyZGQzMjhlYmQ0N2U4YTU1ZDMzZjUyMDJiZGU3OC5zZXRDb250ZW50KGh0bWxfOGI1NjIyYjgxNmRmNDA3NTlmYTBkZWRiNmFhZjRkNjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDM5ZTBlNDM0ZjllNDFhZWI2NTVlYTI5YWU1Y2Y0NzcuYmluZFBvcHVwKHBvcHVwXzAxYjJkZDMyOGViZDQ3ZThhNTVkMzNmNTIwMmJkZTc4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmY2RkZGFlMWQxYTQ2MWI5OWZkMWQ5ZjRiNTYzZjFlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzIuODcyNjEyODQ5OTk5OTk2LC05Ni43NjUyNTg2NzY4MTc2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85OThmYjQwMzNiNWY0MjQzODk1MTM3MDI4MzIwNTJjMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNzJkODRhYTY3Zjc0Yzk1YWY5ODA2N2E5ZDI3ZWU4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTcyZDg0YWE2N2Y3NGM5NWFmOTgwNjdhOWQyN2VlOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBBUksgTEFORSwgMjI0MTUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85OThmYjQwMzNiNWY0MjQzODk1MTM3MDI4MzIwNTJjMC5zZXRDb250ZW50KGh0bWxfZTcyZDg0YWE2N2Y3NGM5NWFmOTgwNjdhOWQyN2VlOGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWZjZGRkYWUxZDFhNDYxYjk5ZmQxZDlmNGI1NjNmMWUuYmluZFBvcHVwKHBvcHVwXzk5OGZiNDAzM2I1ZjQyNDM4OTUxMzcwMjgzMjA1MmMwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IyNWVjOThkN2ZiZjQzMWZhYTM2NWE2MmU4OGJhNjY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDAyNzcxMSwtMC40MTM3MzMxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JhNjAwMWMxYWZhODQwYTI4Yzc3ZjU5YjkyNmFhYzA4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNmYmQ2ZThjZDdhZjQwMjg4OGQxMmQ5NTg2MWI5ZmExID0gJCgnPGRpdiBpZD0iaHRtbF8zZmJkNmU4Y2Q3YWY0MDI4ODhkMTJkOTU4NjFiOWZhMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSS0UgUk9BRCwgMjI2MjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYTYwMDFjMWFmYTg0MGEyOGM3N2Y1OWI5MjZhYWMwOC5zZXRDb250ZW50KGh0bWxfM2ZiZDZlOGNkN2FmNDAyODg4ZDEyZDk1ODYxYjlmYTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjI1ZWM5OGQ3ZmJmNDMxZmFhMzY1YTYyZTg4YmE2NjYuYmluZFBvcHVwKHBvcHVwX2JhNjAwMWMxYWZhODQwYTI4Yzc3ZjU5YjkyNmFhYzA4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUyODFkZDI0Y2I2ODRiNzNhNzU2ZDA0YzAyYmU4NTdjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuMzg5Njc0NywtNi45NDczMTEwMjcwMzczNDc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmZjEzZWE0NzIzMTQyYmZiMDBmNjI0NGVkMDJhYzUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I1MTIxZmY4NDhlYTQ0MThhYzQ4OTFiN2JjNjIzZWE1ID0gJCgnPGRpdiBpZD0iaHRtbF9iNTEyMWZmODQ4ZWE0NDE4YWM0ODkxYjdiYzYyM2VhNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSS0ZJRUxEUywgMjIwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZmYxM2VhNDcyMzE0MmJmYjAwZjYyNDRlZDAyYWM1MS5zZXRDb250ZW50KGh0bWxfYjUxMjFmZjg0OGVhNDQxOGFjNDg5MWI3YmM2MjNlYTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTI4MWRkMjRjYjY4NGI3M2E3NTZkMDRjMDJiZTg1N2MuYmluZFBvcHVwKHBvcHVwX2JmZjEzZWE0NzIzMTQyYmZiMDBmNjI0NGVkMDJhYzUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhjOTBkMjJhYTQ5MzQwMmE4MTU0ZDhjOTgxNGZiZjA4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDc1NDgwNCwtMC4xOTYzMTY3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I5ZWFiZGQ0MTVjYzQwNTliZGRlNjAxYmViMmQyZDEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEyOGNmMzExNjJkNDQxOThiMWU4OWMzMDNiMjI5YmU3ID0gJCgnPGRpdiBpZD0iaHRtbF8xMjhjZjMxMTYyZDQ0MTk4YjFlODljMzAzYjIyOWJlNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSVEhFTklBIFJPQUQsIDIyMDA1NjYuNjY2NjY2NjY2NTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjllYWJkZDQxNWNjNDA1OWJkZGU2MDFiZWIyZDJkMTAuc2V0Q29udGVudChodG1sXzEyOGNmMzExNjJkNDQxOThiMWU4OWMzMDNiMjI5YmU3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhjOTBkMjJhYTQ5MzQwMmE4MTU0ZDhjOTgxNGZiZjA4LmJpbmRQb3B1cChwb3B1cF9iOWVhYmRkNDE1Y2M0MDU5YmRkZTYwMWJlYjJkMmQxMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzQ4ODFlMGFmMjM0ZjY5YjE0N2Y0ZWM5NDRmODkxOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjA4NjU5ODQsMS4xNzY0ODg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q1MmJiODRjMDY2OTRhZWNhYTMyZGIwNWRjZjk5MzU1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdiNWZkMGI1NDVhZjRhZGRhZWIyNzY0MmNjNzgwYWU5ID0gJCgnPGRpdiBpZD0iaHRtbF83YjVmZDBiNTQ1YWY0YWRkYWViMjc2NDJjYzc4MGFlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFWSUxJT04gUk9BRCwgMjIwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNTJiYjg0YzA2Njk0YWVjYWEzMmRiMDVkY2Y5OTM1NS5zZXRDb250ZW50KGh0bWxfN2I1ZmQwYjU0NWFmNGFkZGFlYjI3NjQyY2M3ODBhZTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjc0ODgxZTBhZjIzNGY2OWIxNDdmNGVjOTQ0Zjg5MTguYmluZFBvcHVwKHBvcHVwX2Q1MmJiODRjMDY2OTRhZWNhYTMyZGIwNWRjZjk5MzU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmOGU5NDdhZGZmYTRkZmY5NGE0ZDJkNzc5MzA3MjA5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTEyMzYxOSwtMC4xOTgyOTQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzllZDYyYzQzMTEyNjQ3MGM5YmNhZmMzY2Q0MGVmZGE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIzM2YzNjNmZWY2NDRiNjRhMGYyMTJhNTM5NmE0NjhlID0gJCgnPGRpdiBpZD0iaHRtbF8yMzNmMzYzZmVmNjQ0YjY0YTBmMjEyYTUzOTZhNDY4ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEVNQlJJREdFIE1FV1MsIDIyNTEwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWVkNjJjNDMxMTI2NDcwYzliY2FmYzNjZDQwZWZkYTQuc2V0Q29udGVudChodG1sXzIzM2YzNjNmZWY2NDRiNjRhMGYyMTJhNTM5NmE0NjhlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBmOGU5NDdhZGZmYTRkZmY5NGE0ZDJkNzc5MzA3MjA5LmJpbmRQb3B1cChwb3B1cF85ZWQ2MmM0MzExMjY0NzBjOWJjYWZjM2NkNDBlZmRhNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82MzQ2YzE3Y2I4YzI0YTIwOGZmYjU4Njk3M2JlMDhhYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUwOTE0ODMsLTAuMTk3MDg2M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZmQyZjViNDgzZDg0ODMxODU3NmMyMDBmMWFmMjAxMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kODUwNGI2NTM4MjI0YjU2OGJhY2YzOTRlNGY5ZGI0ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDg1MDRiNjUzODIyNGI1NjhiYWNmMzk0ZTRmOWRiNGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBFTUJSSURHRSBST0FELCAyNDAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmZDJmNWI0ODNkODQ4MzE4NTc2YzIwMGYxYWYyMDEzLnNldENvbnRlbnQoaHRtbF9kODUwNGI2NTM4MjI0YjU2OGJhY2YzOTRlNGY5ZGI0Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MzQ2YzE3Y2I4YzI0YTIwOGZmYjU4Njk3M2JlMDhhYi5iaW5kUG9wdXAocG9wdXBfNGZkMmY1YjQ4M2Q4NDgzMTg1NzZjMjAwZjFhZjIwMTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTllMjBlMDg1NTM3NDMzYzlkN2MxNDVmOGFiYzQwZmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMi4yOTMwOTEzNSwtNjQuNzc5MzE4NjY0NDQxMDldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWY3ZmI1YmQyNDVjNDBjMGFhODYzMmMyNmQ0MWE2MjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTdjNjk5YmNjOTM0NDQzNzkyY2YyZDk0Y2VkODI2ZTMgPSAkKCc8ZGl2IGlkPSJodG1sXzk3YzY5OWJjYzkzNDQ0Mzc5MmNmMmQ5NGNlZDgyNmUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QRU1CUk9LRSBTVFVESU9TLCAyNDUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VmN2ZiNWJkMjQ1YzQwYzBhYTg2MzJjMjZkNDFhNjIyLnNldENvbnRlbnQoaHRtbF85N2M2OTliY2M5MzQ0NDM3OTJjZjJkOTRjZWQ4MjZlMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xOWUyMGUwODU1Mzc0MzNjOWQ3YzE0NWY4YWJjNDBmZS5iaW5kUG9wdXAocG9wdXBfZWY3ZmI1YmQyNDVjNDBjMGFhODYzMmMyNmQ0MWE2MjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDNiZjUxMGUzZjQxNDc2ZTljNmEzZTQ3ZDQwYjg4YTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTM1NDcsLTAuMjAwMzgxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MGE0NzYwOTc0OWM0M2FmOGU4N2ZjOWY3Y2M1N2U0NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMGNmZjgyYjRjODU0Y2MzOGFjZTI5YTYyZjU5NmNiMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMjBjZmY4MmI0Yzg1NGNjMzhhY2UyOWE2MmY1OTZjYjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBFTkNPTUJFIE1FV1MsIDIyMDAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjBhNDc2MDk3NDljNDNhZjhlODdmYzlmN2NjNTdlNDYuc2V0Q29udGVudChodG1sXzIwY2ZmODJiNGM4NTRjYzM4YWNlMjlhNjJmNTk2Y2IwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QzYmY1MTBlM2Y0MTQ3NmU5YzZhM2U0N2Q0MGI4OGExLmJpbmRQb3B1cChwb3B1cF82MGE0NzYwOTc0OWM0M2FmOGU4N2ZjOWY3Y2M1N2U0Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80YTc0NTY4NzI0ZGI0ZDkwOTE5N2E5YjBiMjBiZTI1MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjQ2MDcxMjgsLTEuOTM2MDc0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZTM3ZTgwMWVkZTc0Yzk4OGE0NmYyN2QzZDRjMDE2MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NDZkNGQ5MTFhNmU0MDRkYTNlYzM3NDhiMjAyOGIxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDQ2ZDRkOTExYTZlNDA0ZGEzZWMzNzQ4YjIwMjhiMWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBFVEVSU0hBTSBQTEFDRSwgMjMwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZTM3ZTgwMWVkZTc0Yzk4OGE0NmYyN2QzZDRjMDE2MS5zZXRDb250ZW50KGh0bWxfNDQ2ZDRkOTExYTZlNDA0ZGEzZWMzNzQ4YjIwMjhiMWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGE3NDU2ODcyNGRiNGQ5MDkxOTdhOWIwYjIwYmUyNTEuYmluZFBvcHVwKHBvcHVwXzNlMzdlODAxZWRlNzRjOTg4YTQ2ZjI3ZDNkNGMwMTYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MwMGRmYjFjYTc3NTQ4YzRiYzAyOGY5M2QyNzFhNjU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTMyLjA1NDg0MzUwMDAwMDAwNCwxMTUuNzQyNzk2MzAyMDg1OTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQ3M2QwZTQ0OTVlNDYzYjllOWZhMTQ3ZGQ0NWUxMWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmQ0YzdhM2VkOGYzNDU0M2FlZmFmODRhNWFiY2I1OGEgPSAkKCc8ZGl2IGlkPSJodG1sXzJkNGM3YTNlZDhmMzQ1NDNhZWZhZjg0YTVhYmNiNThhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QSElMTElNT1JFIEdBUkRFTlMsIDI0ODQ2NjYuNjY2NjY2NjY2NTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWQ3M2QwZTQ0OTVlNDYzYjllOWZhMTQ3ZGQ0NWUxMWIuc2V0Q29udGVudChodG1sXzJkNGM3YTNlZDhmMzQ1NDNhZWZhZjg0YTVhYmNiNThhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MwMGRmYjFjYTc3NTQ4YzRiYzAyOGY5M2QyNzFhNjU4LmJpbmRQb3B1cChwb3B1cF81ZDczZDBlNDQ5NWU0NjNiOWU5ZmExNDdkZDQ1ZTExYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NjE4MDAyMTRiZjc0MTRmYTM0Y2YyY2U0NWMzNDdkYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NTU5MjgsLTAuMTYxOTg0Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZDk0YjY0MDE4MDA0ZWY5OWQyYzBkYzY4MGYyM2I4OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZDIyNWFiM2QxNzA0ZmI0YjBjY2I4NGE0ODNiOGU0OCA9ICQoJzxkaXYgaWQ9Imh0bWxfYWQyMjVhYjNkMTcwNGZiNGIwY2NiODRhNDgzYjhlNDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBIWVNJQyBQTEFDRSwgMjUwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZDk0YjY0MDE4MDA0ZWY5OWQyYzBkYzY4MGYyM2I4OC5zZXRDb250ZW50KGh0bWxfYWQyMjVhYjNkMTcwNGZiNGIwY2NiODRhNDgzYjhlNDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjYxODAwMjE0YmY3NDE0ZmEzNGNmMmNlNDVjMzQ3ZGIuYmluZFBvcHVwKHBvcHVwXzZkOTRiNjQwMTgwMDRlZjk5ZDJjMGRjNjgwZjIzYjg4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ViNGJmYjc0NWMwODRiNTViOTlhM2ExYjE1M2NlNjY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI5Mjg3NywtMC4wODM0NTI0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZiYjU5NmZmZjVlYzRlYzRhZTIwMmZkNmFmMTc0NmM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FjYTk2OTBkODkyMDRjMDRhMzhjNGE3M2Q3ZDc1YTVmID0gJCgnPGRpdiBpZD0iaHRtbF9hY2E5NjkwZDg5MjA0YzA0YTM4YzRhNzNkN2Q3NWE1ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UElURklFTEQgU1RSRUVULCAyNDgzMzMzLjMzMzMzMzMzMzU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZiYjU5NmZmZjVlYzRlYzRhZTIwMmZkNmFmMTc0NmM4LnNldENvbnRlbnQoaHRtbF9hY2E5NjkwZDg5MjA0YzA0YTM4YzRhNzNkN2Q3NWE1Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYjRiZmI3NDVjMDg0YjU1Yjk5YTNhMWIxNTNjZTY2NC5iaW5kUG9wdXAocG9wdXBfNmJiNTk2ZmZmNWVjNGVjNGFlMjAyZmQ2YWYxNzQ2YzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjk3NTc4YTA1MmVlNGNmMWIwMWVmMjgzMTY1Y2VlYTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS43ODI0ODMsLTQuNzAzOTY3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MjlmZjlmNWEwNTU0NTU2YWVjMzQ2N2YxZGJiNGUxMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYTA0ZGQyZDE0ZmQ0MTk3ODkzYTRhNDFiZmZmZmYyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWEwNGRkMmQxNGZkNDE5Nzg5M2E0YTQxYmZmZmZmMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBSSU5DRVMgR0FURSwgMjQwMzMzMy4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MjlmZjlmNWEwNTU0NTU2YWVjMzQ2N2YxZGJiNGUxMS5zZXRDb250ZW50KGh0bWxfYWEwNGRkMmQxNGZkNDE5Nzg5M2E0YTQxYmZmZmZmMmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjk3NTc4YTA1MmVlNGNmMWIwMWVmMjgzMTY1Y2VlYTguYmluZFBvcHVwKHBvcHVwXzYyOWZmOWY1YTA1NTQ1NTZhZWMzNDY3ZjFkYmI0ZTExKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk4YzQzYjUzYWIxYzQ0YWI5MDIxYTFkYTJiNGY3NjExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuMjY3OTQ3NTUsLTkuMDU1NTk3MzIyMTcyODRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDVkZjQ4ZTJkMDkyNDJiYzkwZjhmZDdiMTRhMTJkNzAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWU2MTJjZmI3NDk1NDE1MGEzNjg3YzgxNzNmY2IwZGEgPSAkKCc8ZGl2IGlkPSJodG1sX2FlNjEyY2ZiNzQ5NTQxNTBhMzY4N2M4MTczZmNiMGRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QUklPUlkgUk9BRCwgMjI0ODI1OC4yMzgwOTUyMzg8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q1ZGY0OGUyZDA5MjQyYmM5MGY4ZmQ3YjE0YTEyZDcwLnNldENvbnRlbnQoaHRtbF9hZTYxMmNmYjc0OTU0MTUwYTM2ODdjODE3M2ZjYjBkYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OGM0M2I1M2FiMWM0NGFiOTAyMWExZGEyYjRmNzYxMS5iaW5kUG9wdXAocG9wdXBfZDVkZjQ4ZTJkMDkyNDJiYzkwZjhmZDdiMTRhMTJkNzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmE5MzI2YmM3MDE1NDcwYTgxMjliMmJiOWY1NmJjMDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41ODU3MzEsLTAuMjI4MDYxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NjU2YTEzYzU5YTA0ODI3OTdkNzUyNjZhYTNjNjBjOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjZhZTA5MDQzNGE0NzJiYTIxMDg4MjI0MzhhNTY5ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfODI2YWUwOTA0MzRhNDcyYmEyMTA4ODIyNDM4YTU2OWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBST1RIRVJPIEdBUkRFTlMsIDIyMDI1MDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjY1NmExM2M1OWEwNDgyNzk3ZDc1MjY2YWEzYzYwYzkuc2V0Q29udGVudChodG1sXzgyNmFlMDkwNDM0YTQ3MmJhMjEwODgyMjQzOGE1NjlkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJhOTMyNmJjNzAxNTQ3MGE4MTI5YjJiYjlmNTZiYzA2LmJpbmRQb3B1cChwb3B1cF82NjU2YTEzYzU5YTA0ODI3OTdkNzUyNjZhYTNjNjBjOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85Y2IyMjJkYjU3NmQ0MGQxOGNjZGJiYjdmYjUxZjQ1OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ2MDk4NjksLTAuMjE3MDYyNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMDcyMTc5OGI0MDE0MmM0OWMzNWFlMzIxOGVkNWFjNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lOThmOTRjNjNkNTk0MGY1ODQ2ZDNmNmRiNjMyMzgyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTk4Zjk0YzYzZDU5NDBmNTg0NmQzZjZkYjYzMjM4MjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBVVE5FWSBISUdIIFNUUkVFVCwgMjM0ODYwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMDcyMTc5OGI0MDE0MmM0OWMzNWFlMzIxOGVkNWFjNi5zZXRDb250ZW50KGh0bWxfZTk4Zjk0YzYzZDU5NDBmNTg0NmQzZjZkYjYzMjM4MjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWNiMjIyZGI1NzZkNDBkMThjY2RiYmI3ZmI1MWY0NTguYmluZFBvcHVwKHBvcHVwXzEwNzIxNzk4YjQwMTQyYzQ5YzM1YWUzMjE4ZWQ1YWM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA3MGM5YTYzNWNjNDQ5M2ZhMjc3YTUwZmVmODdiMDg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDczMTEyNywtMC4xOTU4Njc0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMwYjhjNjA2ZTNiZTRjZDViNWI2NDU0MDY3ZTc0ZGEyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E0YTRjMDg2ZjRjNzQyOGI5NTU4ZDI4YzNhN2FiOTE5ID0gJCgnPGRpdiBpZD0iaHRtbF9hNGE0YzA4NmY0Yzc0MjhiOTU1OGQyOGMzYTdhYjkxOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UVVBUlJFTkRPTiBTVFJFRVQsIDI0Mzc3NTAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzBiOGM2MDZlM2JlNGNkNWI1YjY0NTQwNjdlNzRkYTIuc2V0Q29udGVudChodG1sX2E0YTRjMDg2ZjRjNzQyOGI5NTU4ZDI4YzNhN2FiOTE5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA3MGM5YTYzNWNjNDQ5M2ZhMjc3YTUwZmVmODdiMDg1LmJpbmRQb3B1cChwb3B1cF8zMGI4YzYwNmUzYmU0Y2Q1YjViNjQ1NDA2N2U3NGRhMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMGRmZGZkYmExYmU0YWZiOGZiM2IxYjkxNTA1MGE5MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM2Ljk1Mjk2NjIsLTc2LjUyNjcyNDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTc2NmM0NjBlYzIyNDhmZmExNjdkN2FhMTg4MjJlMjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWVkMjFkZjhhYzhjNDc4YTk0NTkyYjI5ZjUxN2ZhZjQgPSAkKCc8ZGl2IGlkPSJodG1sXzVlZDIxZGY4YWM4YzQ3OGE5NDU5MmIyOWY1MTdmYWY0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RVUVFTlMgR0FURSBURVJSQUNFLCAyMzk5NDQ0LjQ0NDQ0NDQ0NDU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE3NjZjNDYwZWMyMjQ4ZmZhMTY3ZDdhYTE4ODIyZTIxLnNldENvbnRlbnQoaHRtbF81ZWQyMWRmOGFjOGM0NzhhOTQ1OTJiMjlmNTE3ZmFmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMGRmZGZkYmExYmU0YWZiOGZiM2IxYjkxNTA1MGE5MS5iaW5kUG9wdXAocG9wdXBfMTc2NmM0NjBlYzIyNDhmZmExNjdkN2FhMTg4MjJlMjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjRjN2FiZjY3ODMzNDMzZWE3ZGQwNzk3MDdlYmVkZjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NzkwOTA2LC0wLjE2OTQ1NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMDBhOGM4Y2M2ZmM0YmJiOTU4OTdkYjQ0M2ZlMTRjYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hYjAyODM5N2FjZmE0YTEyYjkwNzQzNTM1NGMyNjY2NSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWIwMjgzOTdhY2ZhNGExMmI5MDc0MzUzNTRjMjY2NjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJBRFNUT0NLIFNUUkVFVCwgMjIxNzUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMDBhOGM4Y2M2ZmM0YmJiOTU4OTdkYjQ0M2ZlMTRjYi5zZXRDb250ZW50KGh0bWxfYWIwMjgzOTdhY2ZhNGExMmI5MDc0MzUzNTRjMjY2NjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjRjN2FiZjY3ODMzNDMzZWE3ZGQwNzk3MDdlYmVkZjcuYmluZFBvcHVwKHBvcHVwX2MwMGE4YzhjYzZmYzRiYmI5NTg5N2RiNDQzZmUxNGNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBmM2YyZWYzZDFmZTQxYzJhODlhNDc1NWUwZmRkYTdlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDY4ODkyOCwtMC4yMDY0ODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTc4NzcyYzkzMTU1NGRiMDg3NzM3NmI0NmM2ZWZiZmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDYzNzVhOGRkMjRiNDRlNWJjZGJlMTNmNGI1ODhhYzQgPSAkKCc8ZGl2IGlkPSJodG1sXzQ2Mzc1YThkZDI0YjQ0ZTViY2RiZTEzZjRiNTg4YWM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SQU5FTEFHSCBBVkVOVUUsIDIzMDAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTc4NzcyYzkzMTU1NGRiMDg3NzM3NmI0NmM2ZWZiZmIuc2V0Q29udGVudChodG1sXzQ2Mzc1YThkZDI0YjQ0ZTViY2RiZTEzZjRiNTg4YWM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBmM2YyZWYzZDFmZTQxYzJhODlhNDc1NWUwZmRkYTdlLmJpbmRQb3B1cChwb3B1cF85Nzg3NzJjOTMxNTU0ZGIwODc3Mzc2YjQ2YzZlZmJmYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMjhmYWFmZmE4ZDY0NTczYWEwMDQ1ZmJkMjRkNmVhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjcyNzE3NDMsMC40NjY4MDQ1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk3NmM1MTBjMDZhYzQwMDg5M2I1YzVkMDk4ZjFhODE3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NlOWM4NGRiMTk0ZDQ1Y2Y5NmU3ZTU2MDQzYjUwZjAyID0gJCgnPGRpdiBpZD0iaHRtbF9jZTljODRkYjE5NGQ0NWNmOTZlN2U1NjA0M2I1MGYwMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkVEQ0xJRkZFIFJPQUQsIDI0NDg3NTAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTc2YzUxMGMwNmFjNDAwODkzYjVjNWQwOThmMWE4MTcuc2V0Q29udGVudChodG1sX2NlOWM4NGRiMTk0ZDQ1Y2Y5NmU3ZTU2MDQzYjUwZjAyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MyOGZhYWZmYThkNjQ1NzNhYTAwNDVmYmQyNGQ2ZWE4LmJpbmRQb3B1cChwb3B1cF85NzZjNTEwYzA2YWM0MDA4OTNiNWM1ZDA5OGYxYTgxNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYzk0MTQyOGZhMmI0YmIyOTc2OTI3NGM1OTNmMDk2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUwOTcwNzcsLTAuMTU0MDk4Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMGQzN2Q5NDc4OWY0YWY0YWYyYjI5NDgwNzcyNDc3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NjZhNjJkYjRjMTI0M2NmOWE0ZmJhOGM3MjNlN2Q2NiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTY2YTYyZGI0YzEyNDNjZjlhNGZiYThjNzIzZTdkNjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJFRVZFUyBNRVdTLCAyNDUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YwZDM3ZDk0Nzg5ZjRhZjRhZjJiMjk0ODA3NzI0NzdkLnNldENvbnRlbnQoaHRtbF85NjZhNjJkYjRjMTI0M2NmOWE0ZmJhOGM3MjNlN2Q2Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYzk0MTQyOGZhMmI0YmIyOTc2OTI3NGM1OTNmMDk2Yi5iaW5kUG9wdXAocG9wdXBfZjBkMzdkOTQ3ODlmNGFmNGFmMmIyOTQ4MDc3MjQ3N2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTczZTY2NTQ3MDYyNDMzNTg4ZTI2NGU5YjM5MzZiOWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MzQ5OTY1LC0wLjA5ODEzODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTNhOTllNTJlYTZhNGJhYThmNWQ4MDIwZWNmNTM0YzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmE3NTg2MDY1MmExNDhlYTgzZDk4ZTkyOGQxNzQxOTIgPSAkKCc8ZGl2IGlkPSJodG1sXzJhNzU4NjA2NTJhMTQ4ZWE4M2Q5OGU5MjhkMTc0MTkyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SSEVJRE9MIE1FV1MsIDIzMTAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTNhOTllNTJlYTZhNGJhYThmNWQ4MDIwZWNmNTM0Yzcuc2V0Q29udGVudChodG1sXzJhNzU4NjA2NTJhMTQ4ZWE4M2Q5OGU5MjhkMTc0MTkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk3M2U2NjU0NzA2MjQzMzU4OGUyNjRlOWIzOTM2YjllLmJpbmRQb3B1cChwb3B1cF8xM2E5OWU1MmVhNmE0YmFhOGY1ZDgwMjBlY2Y1MzRjNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NjFiZDczYjYzZTE0YTBlYTFiYTVkYmRiZjE1ZGM1NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU5MzY5OTIsLTAuMTU1NTgxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZjhmYzhkODM4ZTE0ZDJjOWFiNjMzZDRlOWE4NDQ1ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOTkzMDU5Y2ExZGY0NjcyYWZkZWZhODU2YTU1NTc2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzk5MzA1OWNhMWRmNDY3MmFmZGVmYTg1NmE1NTU3NmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJJTkdXT09EIEFWRU5VRSwgMjI3NTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZjhmYzhkODM4ZTE0ZDJjOWFiNjMzZDRlOWE4NDQ1Zi5zZXRDb250ZW50KGh0bWxfMzk5MzA1OWNhMWRmNDY3MmFmZGVmYTg1NmE1NTU3NmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzYxYmQ3M2I2M2UxNGEwZWExYmE1ZGJkYmYxNWRjNTcuYmluZFBvcHVwKHBvcHVwXzdmOGZjOGQ4MzhlMTRkMmM5YWI2MzNkNGU5YTg0NDVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY3MjE0ODMzZTczZDQ5MjJiMmQ1N2UwMmE3ODQ2NDFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNDU3OTE3LC0xLjg2NTEyODRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzBhZWExYTRiYmUxNGUxN2JlM2Y4NmE1MzFjZDdmNmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGE4M2MxNjMxMDNkNGNmYjlkYTAxMTE3ZTYzOWM4MDkgPSAkKCc8ZGl2IGlkPSJodG1sXzRhODNjMTYzMTAzZDRjZmI5ZGEwMTExN2U2MzljODA5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ST0RFUklDSyBST0FELCAyNDAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMwYWVhMWE0YmJlMTRlMTdiZTNmODZhNTMxY2Q3ZjZkLnNldENvbnRlbnQoaHRtbF80YTgzYzE2MzEwM2Q0Y2ZiOWRhMDExMTdlNjM5YzgwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NzIxNDgzM2U3M2Q0OTIyYjJkNTdlMDJhNzg0NjQxZi5iaW5kUG9wdXAocG9wdXBfMzBhZWExYTRiYmUxNGUxN2JlM2Y4NmE1MzFjZDdmNmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDNmY2Q2YzNhY2UxNGE4Yzg0OTkxZjEyNDJiZTJjMGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDk4NDMyNSwtMC4wMzI4MDY1NDUxMDQzNzUwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMWZkMTEwYjQ1NWI0YTUwOTU5N2QxMTI3NDgzZGQ0NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lODViZTcxOGFiM2U0MWZhOGQxYTc4OGZkODAwMDE2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTg1YmU3MThhYjNlNDFmYThkMWE3ODhmZDgwMDAxNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJPUEVNQUtFUlMgRklFTERTLCAyNTAwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UxZmQxMTBiNDU1YjRhNTA5NTk3ZDExMjc0ODNkZDQ3LnNldENvbnRlbnQoaHRtbF9lODViZTcxOGFiM2U0MWZhOGQxYTc4OGZkODAwMDE2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kM2ZjZDZjM2FjZTE0YThjODQ5OTFmMTI0MmJlMmMwZS5iaW5kUG9wdXAocG9wdXBfZTFmZDExMGI0NTViNGE1MDk1OTdkMTEyNzQ4M2RkNDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTRlMWU2NGNjMzVkNDE2M2EwNjg1NDQ4N2EyMjhjNmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zODcyOTU3LC0yLjM2ODQzOTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjIwMGM4OTJhNzczNDk0MzkyOTFmNjcyNGNmMjkwNjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjYwNmQxNWEyY2E4NDk5MmFkNDVmZGI2YmRmM2E4YWIgPSAkKCc8ZGl2IGlkPSJodG1sX2Y2MDZkMTVhMmNhODQ5OTJhZDQ1ZmRiNmJkZjNhOGFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ST1lBTCBDUkVTQ0VOVCwgMjM0ODMzMy4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMjAwYzg5MmE3NzM0OTQzOTI5MWY2NzI0Y2YyOTA2Ni5zZXRDb250ZW50KGh0bWxfZjYwNmQxNWEyY2E4NDk5MmFkNDVmZGI2YmRmM2E4YWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTRlMWU2NGNjMzVkNDE2M2EwNjg1NDQ4N2EyMjhjNmQuYmluZFBvcHVwKHBvcHVwX2YyMDBjODkyYTc3MzQ5NDM5MjkxZjY3MjRjZjI5MDY2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzMTZhY2Y5ZWNlNjRjMjk4NGI1YmQ3MmNkNzk1ZWQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTAuNTM3MDM5MywtMy45NTI1NjcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZmODUyYzcxY2NjYzRiODRhZTdiMmIxM2JiZjU4ZWNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEzODQ0ZDY4Y2I1NjQ0MTJiYmY0MDMyYmI3ZDk1MDlmID0gJCgnPGRpdiBpZD0iaHRtbF8xMzg0NGQ2OGNiNTY0NDEyYmJmNDAzMmJiN2Q5NTA5ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Uk9ZQUwgSElMTCwgMjI1MjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZjg1MmM3MWNjY2M0Yjg0YWU3YjJiMTNiYmY1OGVjYi5zZXRDb250ZW50KGh0bWxfMTM4NDRkNjhjYjU2NDQxMmJiZjQwMzJiYjdkOTUwOWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTMxNmFjZjllY2U2NGMyOTg0YjViZDcyY2Q3OTVlZDIuYmluZFBvcHVwKHBvcHVwX2ZmODUyYzcxY2NjYzRiODRhZTdiMmIxM2JiZjU4ZWNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QyZGZkYzJkM2M1NTQ5N2M5Yjg1YzFlMzAwMjUxYzg4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk5ODY4OSwtMC4yMTE4OTA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI4ZTRiNWZlN2JkZDQ3ZjI4YTc4MTQyNmZkMWQyYTEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYzMDJhYWQwMjIyMTRiYzU4ZWE3YTViYzgzMWQyZDE1ID0gJCgnPGRpdiBpZD0iaHRtbF82MzAyYWFkMDIyMjE0YmM1OGVhN2E1YmM4MzFkMmQxNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UlVTU0VMTCBHQVJERU5TIE1FV1MsIDIzMDAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjhlNGI1ZmU3YmRkNDdmMjhhNzgxNDI2ZmQxZDJhMTAuc2V0Q29udGVudChodG1sXzYzMDJhYWQwMjIyMTRiYzU4ZWE3YTViYzgzMWQyZDE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyZGZkYzJkM2M1NTQ5N2M5Yjg1YzFlMzAwMjUxYzg4LmJpbmRQb3B1cChwb3B1cF8yOGU0YjVmZTdiZGQ0N2YyOGE3ODE0MjZmZDFkMmExMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YThlZmI5ZDFkODY0NTE5YjlkZDRmODk2ODQ3YjdkNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUxNTMyMzcsLTAuMDY0MzI2Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NzhiNjE1YzY2Yjg0ZmYyYmZjNjA2OWM4N2YzNTFlMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNWE4MThhNDdmYjY0NzMxODgwZGJlOTcyY2RjNWZkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzVhODE4YTQ3ZmI2NDczMTg4MGRiZTk3MmNkYzVmZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNFVFRMRVMgU1RSRUVULCAyNDg3NTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY3OGI2MTVjNjZiODRmZjJiZmM2MDY5Yzg3ZjM1MWUwLnNldENvbnRlbnQoaHRtbF9jNWE4MThhNDdmYjY0NzMxODgwZGJlOTcyY2RjNWZkZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YThlZmI5ZDFkODY0NTE5YjlkZDRmODk2ODQ3YjdkNC5iaW5kUG9wdXAocG9wdXBfNjc4YjYxNWM2NmI4NGZmMmJmYzYwNjljODdmMzUxZTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWE5MTgzYTBlNGQ4NDE1N2I5YmEzYzM3YTJkZDViNTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41OTI2MTMsMC4wNzMxNDQ5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U2NTU4M2QzYWI1YzRmODM4NWYwZWY3OWVlZjlkZGNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I1NjgyMzA0Nzc1MDQ2MTA5YWY3ZmExMTE0ZmMwMjIzID0gJCgnPGRpdiBpZD0iaHRtbF9iNTY4MjMwNDc3NTA0NjEwOWFmN2ZhMTExNGZjMDIyMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U0hFTERPTiBBVkVOVUUsIDIzNDk1NDEuNjY2NjY2NjY2NTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTY1NTgzZDNhYjVjNGY4Mzg1ZjBlZjc5ZWVmOWRkY2Iuc2V0Q29udGVudChodG1sX2I1NjgyMzA0Nzc1MDQ2MTA5YWY3ZmExMTE0ZmMwMjIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FhOTE4M2EwZTRkODQxNTdiOWJhM2MzN2EyZGQ1YjU3LmJpbmRQb3B1cChwb3B1cF9lNjU1ODNkM2FiNWM0ZjgzODVmMGVmNzllZWY5ZGRjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xYWY4MDNjMmQyMDQ0M2IzYmE2MDY1ZDJmZGY3NjgwMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5ODc0NjMsLTAuMTg5MDc4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NzRhNzBmYmM2NjY0MGE1OTk0ZDIxN2ZlN2FlZjYzMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83N2JmMDcxZTQwZDU0Y2I5YjE1Zjc3MzEyODUyMjJhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzdiZjA3MWU0MGQ1NGNiOWIxNWY3NzMxMjg1MjIyYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNPVVRIIEVORCBST1csIDI0NzAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTc0YTcwZmJjNjY2NDBhNTk5NGQyMTdmZTdhZWY2MzMuc2V0Q29udGVudChodG1sXzc3YmYwNzFlNDBkNTRjYjliMTVmNzczMTI4NTIyMmExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFhZjgwM2MyZDIwNDQzYjNiYTYwNjVkMmZkZjc2ODAxLmJpbmRQb3B1cChwb3B1cF85NzRhNzBmYmM2NjY0MGE1OTk0ZDIxN2ZlN2FlZjYzMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOTYyMDI0NzU2NDY0NGRjYmY1OWNmNzJiYjE2YmE5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU3NDYyNjcsLTAuMTQ2MjM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IwYWEyODMxMjdjNjQ3NTNhMTg3ZTlkYzAxN2M4YTk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZkMjM0NDJhNjllZjQ4MDE5ZjQ3YTVhYjE0YTE0OTNlID0gJCgnPGRpdiBpZD0iaHRtbF9mZDIzNDQyYTY5ZWY0ODAxOWY0N2E1YWIxNGExNDkzZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U09VVEhXT09EIExBV04gUk9BRCwgMjM1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMGFhMjgzMTI3YzY0NzUzYTE4N2U5ZGMwMTdjOGE5Ny5zZXRDb250ZW50KGh0bWxfZmQyMzQ0MmE2OWVmNDgwMTlmNDdhNWFiMTRhMTQ5M2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDk2MjAyNDc1NjQ2NDRkY2JmNTljZjcyYmIxNmJhOWQuYmluZFBvcHVwKHBvcHVwX2IwYWEyODMxMjdjNjQ3NTNhMTg3ZTlkYzAxN2M4YTk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU5MzQxYzZkZDE0MDRkMTk5OGJhZTViY2RhOGJlNTYyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNjg4MDk0MSwtMi43MjQ1NjcyNTExNDM4NDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDhiM2IwMDgxYzQzNGFhN2I3MmFiMDk1YTYxYjM1ZTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzIyNGNiZDQxMGVjNDRjMzkxODUzODg2ODMxNGQ4NTQgPSAkKCc8ZGl2IGlkPSJodG1sX2MyMjRjYmQ0MTBlYzQ0YzM5MTg1Mzg4NjgzMTRkODU0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TT1ZFUkVJR04gUEFSSywgMjUwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kOGIzYjAwODFjNDM0YWE3YjcyYWIwOTVhNjFiMzVlOC5zZXRDb250ZW50KGh0bWxfYzIyNGNiZDQxMGVjNDRjMzkxODUzODg2ODMxNGQ4NTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTkzNDFjNmRkMTQwNGQxOTk4YmFlNWJjZGE4YmU1NjIuYmluZFBvcHVwKHBvcHVwX2Q4YjNiMDA4MWM0MzRhYTdiNzJhYjA5NWE2MWIzNWU4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkxOTY5ZjVkNmNiMTQ5NjBiMTAxYzMzMmU0M2VjMTNhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMTcuMDk3NTEzNSwtODguNjE2MTE1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZjIyYjk3YTU1OTU0ODBhODRhYWZiNDIzMDliYzBlZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZjMwZTVkOGI2NTA0M2E4Yjg4ZjAyNmU4MjJmOGRkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGYzMGU1ZDhiNjUwNDNhOGI4OGYwMjZlODIyZjhkZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNUIE1BUkdBUkVUUyBDUkVTQ0VOVCwgMjIxNjUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZjIyYjk3YTU1OTU0ODBhODRhYWZiNDIzMDliYzBlZi5zZXRDb250ZW50KGh0bWxfMGYzMGU1ZDhiNjUwNDNhOGI4OGYwMjZlODIyZjhkZDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTE5NjlmNWQ2Y2IxNDk2MGIxMDFjMzMyZTQzZWMxM2EuYmluZFBvcHVwKHBvcHVwX2ZmMjJiOTdhNTU5NTQ4MGE4NGFhZmI0MjMwOWJjMGVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRlNDc2YmIyMjZhNDQ4NzNiYmEyZDIyNGE0NzRmMDAzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDg3MjA3MSwtMC4xMTg1MzQxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YyY2YwZDM4YzEzZTQxOTZiMjRlMzBmZTA4MDNiNTUzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmYWJiMWQwYzI4ZjRhMGQ4YzFhZjVjYjBiN2E5OTI0ID0gJCgnPGRpdiBpZD0iaHRtbF80ZmFiYjFkMGMyOGY0YTBkOGMxYWY1Y2IwYjdhOTkyNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U1QgT1NXQUxEUyBQTEFDRSwgMjI1MDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMmNmMGQzOGMxM2U0MTk2YjI0ZTMwZmUwODAzYjU1My5zZXRDb250ZW50KGh0bWxfNGZhYmIxZDBjMjhmNGEwZDhjMWFmNWNiMGI3YTk5MjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGU0NzZiYjIyNmE0NDg3M2JiYTJkMjI0YTQ3NGYwMDMuYmluZFBvcHVwKHBvcHVwX2YyY2YwZDM4YzEzZTQxOTZiMjRlMzBmZTA4MDNiNTUzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UwYzJiMTYyNWM5MjRhZmQ5MmU5MzA4MWE5MGEyMzYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDEuOTAyMjM1MywxMi40NTczNTczMTAyOTgwMDhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzY3NjkxNmEyMDIzNDNlYThjYTU4ZGJhNTE5Mzk1MDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODE2ZDJhZTkzM2Q1NGNjMGI4MDhkNDRjNWE3Y2I3NDQgPSAkKCc8ZGl2IGlkPSJodG1sXzgxNmQyYWU5MzNkNTRjYzBiODA4ZDQ0YzVhN2NiNzQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TVCBQRVRFUlMgU1FVQVJFLCAyNDY4NzMwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M2NzY5MTZhMjAyMzQzZWE4Y2E1OGRiYTUxOTM5NTA5LnNldENvbnRlbnQoaHRtbF84MTZkMmFlOTMzZDU0Y2MwYjgwOGQ0NGM1YTdjYjc0NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMGMyYjE2MjVjOTI0YWZkOTJlOTMwODFhOTBhMjM2MS5iaW5kUG9wdXAocG9wdXBfYzY3NjkxNmEyMDIzNDNlYThjYTU4ZGJhNTE5Mzk1MDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2Y0NWIwY2E1YTM0NGZkMTgxZjc3NGMyYTQ4OGVjOTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDA5Mzc5LC0wLjE5NjA0OTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWU1OTNiYjc4N2I1NDcyZGJlNmI4M2Q5ZDc4NTQ3ODcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzQxMGNhMTM1MWZmNDBkNmE5NDQzMDJmZTQ0MGNiOWEgPSAkKCc8ZGl2IGlkPSJodG1sXzM0MTBjYTEzNTFmZjQwZDZhOTQ0MzAyZmU0NDBjYjlhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TVEFGRk9SRCBURVJSQUNFLCAyMzU1MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlNTkzYmI3ODdiNTQ3MmRiZTZiODNkOWQ3ODU0Nzg3LnNldENvbnRlbnQoaHRtbF8zNDEwY2ExMzUxZmY0MGQ2YTk0NDMwMmZlNDQwY2I5YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZjQ1YjBjYTVhMzQ0ZmQxODFmNzc0YzJhNDg4ZWM5Mi5iaW5kUG9wdXAocG9wdXBfMWU1OTNiYjc4N2I1NDcyZGJlNmI4M2Q5ZDc4NTQ3ODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGEwMzUxMmZiZmJiNDA0NDk3ZTIwNDhkMDUzMTQyNjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTY2MTY2LC0wLjE5NzI3NjJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTg4Nzk2NzMzMzNhNGFjOTllZjE2YmQxMzM3YmE5ODkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmI1ZjgxYmE3MTMyNGVmODk5ZDBjYTc5M2M2OTllNjEgPSAkKCc8ZGl2IGlkPSJodG1sXzJiNWY4MWJhNzEzMjRlZjg5OWQwY2E3OTNjNjk5ZTYxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TVVRIRVJMQU5EIFBMQUNFLCAyNDU2MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk4ODc5NjczMzMzYTRhYzk5ZWYxNmJkMTMzN2JhOTg5LnNldENvbnRlbnQoaHRtbF8yYjVmODFiYTcxMzI0ZWY4OTlkMGNhNzkzYzY5OWU2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYTAzNTEyZmJmYmI0MDQ0OTdlMjA0OGQwNTMxNDI2NS5iaW5kUG9wdXAocG9wdXBfOTg4Nzk2NzMzMzNhNGFjOTllZjE2YmQxMzM3YmE5ODkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTMyZWMxNmEyYjA2NDRmY2FiZjE5MGQxNzExYjY3NzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS44MDcwODIsMS4wMjM5NjAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y5MTk0ZDU0MjMzODRkNDJiYzIzODA4NDQ2NzMwNTNhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ3MDk3NjUxNThlZjQzNmVhOGU4Y2UwZDhjZDdkMTJhID0gJCgnPGRpdiBpZD0iaHRtbF80NzA5NzY1MTU4ZWY0MzZlYThlOGNlMGQ4Y2Q3ZDEyYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U1lETkVZIFNUUkVFVCwgMjI0MDgzMy4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mOTE5NGQ1NDIzMzg0ZDQyYmMyMzgwODQ0NjczMDUzYS5zZXRDb250ZW50KGh0bWxfNDcwOTc2NTE1OGVmNDM2ZWE4ZThjZTBkOGNkN2QxMmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTMyZWMxNmEyYjA2NDRmY2FiZjE5MGQxNzExYjY3NzMuYmluZFBvcHVwKHBvcHVwX2Y5MTk0ZDU0MjMzODRkNDJiYzIzODA4NDQ2NzMwNTNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiODZiZGVkOWVjMDQwOWFiMzdkMzlhMzEyMWFhMzcxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuMDg1NjY4OSwtMC4yNDMyNzY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkzZDc4MzVmMWEwMzRiZjg5YmZiYmE2ZjU2MTkyODIyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QxMTk1MDk1NjdhMTQ0N2NhMDZjZGMwYjc0NDEzY2U1ID0gJCgnPGRpdiBpZD0iaHRtbF9kMTE5NTA5NTY3YTE0NDdjYTA2Y2RjMGI3NDQxM2NlNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VEhBTUVTIEJBTkssIDI0MDAwMDAuMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTNkNzgzNWYxYTAzNGJmODliZmJiYTZmNTYxOTI4MjIuc2V0Q29udGVudChodG1sX2QxMTk1MDk1NjdhMTQ0N2NhMDZjZGMwYjc0NDEzY2U1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiODZiZGVkOWVjMDQwOWFiMzdkMzlhMzEyMWFhMzcxLmJpbmRQb3B1cChwb3B1cF85M2Q3ODM1ZjFhMDM0YmY4OWJmYmJhNmY1NjE5MjgyMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZTMwMGM4YWMzYmQ0YWM5YTNlMGE3YjE2OTQ5MDYzMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ1Mzg4MjM1LC0wLjk3NzgzNDIxODEwNTI5OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzE5OWUyMzdhYWNjNDY4MGI3OTQ0M2IwOWRiM2MyN2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjllYjNkZTA3MTQ0NDU1OTk4YzQwODU2ZGI4NTBmZGQgPSAkKCc8ZGl2IGlkPSJodG1sXzY5ZWIzZGUwNzE0NDQ1NTk5OGM0MDg1NmRiODUwZmRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5USEUgSEVYQUdPTiwgMjMzNTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMTk5ZTIzN2FhY2M0NjgwYjc5NDQzYjA5ZGIzYzI3Yi5zZXRDb250ZW50KGh0bWxfNjllYjNkZTA3MTQ0NDU1OTk4YzQwODU2ZGI4NTBmZGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWUzMDBjOGFjM2JkNGFjOWEzZTBhN2IxNjk0OTA2MzIuYmluZFBvcHVwKHBvcHVwXzMxOTllMjM3YWFjYzQ2ODBiNzk0NDNiMDlkYjNjMjdiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhNDE4MjM3OWFlMTQ5NTBiNWRiNjdmNTljMTRmM2FjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI3MDkzNSwtMC4wMzIxODQyMzMyNTk2NjA2M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOGQ5MTAwM2EwZTQ0MTY2YWEyMDA2Y2JlZTYzZDI1NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNmE0N2NhMWRkN2U0ZjA2YjU5OWZlOTBmMWJkM2QwYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjZhNDdjYTFkZDdlNGYwNmI1OTlmZTkwZjFiZDNkMGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRSRURFR0FSIFNRVUFSRSwgMjQzNjY2Ni4zMzMzMzMzMzM1PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yOGQ5MTAwM2EwZTQ0MTY2YWEyMDA2Y2JlZTYzZDI1Ny5zZXRDb250ZW50KGh0bWxfMjZhNDdjYTFkZDdlNGYwNmI1OTlmZTkwZjFiZDNkMGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWE0MTgyMzc5YWUxNDk1MGI1ZGI2N2Y1OWMxNGYzYWMuYmluZFBvcHVwKHBvcHVwXzI4ZDkxMDAzYTBlNDQxNjZhYTIwMDZjYmVlNjNkMjU3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE2NzcwOTU1MGQ1NjRiODQ4MWJjNzcxYTU4ZGYwYWIyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNjIzOTM0OSwxLjI4MjEzNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWViNjBkNjg3YTBjNDQ0ZTkwMGNlNDEwZjY2OTEyMzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWQ1N2JlZjdhMzA4NDZhYWI0NzM5MmFjZTY2ZjJmZDMgPSAkKCc8ZGl2IGlkPSJodG1sXzFkNTdiZWY3YTMwODQ2YWFiNDczOTJhY2U2NmYyZmQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UUklOSVRZIFNUUkVFVCwgMjMxNzUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZWI2MGQ2ODdhMGM0NDRlOTAwY2U0MTBmNjY5MTIzMi5zZXRDb250ZW50KGh0bWxfMWQ1N2JlZjdhMzA4NDZhYWI0NzM5MmFjZTY2ZjJmZDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTY3NzA5NTUwZDU2NGI4NDgxYmM3NzFhNThkZjBhYjIuYmluZFBvcHVwKHBvcHVwX2FlYjYwZDY4N2EwYzQ0NGU5MDBjZTQxMGY2NjkxMjMyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYyODM0ODMxNjkzNTQ3M2U4MDIxYjVmYjg0MGFhOGNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTU4NDY3LC0wLjE3NzQ1MjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjVhZWM3MjllY2ZkNDYyMjhhMTgzNjMxNjNiOTU5MjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjEyYWYzODk5ZGQ5NGQ3ZGFkMjBmY2JkZDc0NWRlN2MgPSAkKCc8ZGl2IGlkPSJodG1sXzIxMmFmMzg5OWRkOTRkN2RhZDIwZmNiZGQ3NDVkZTdjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5VUFBFUiBIQU1QU1RFQUQgV0FMSywgMjUwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NWFlYzcyOWVjZmQ0NjIyOGExODM2MzE2M2I5NTkyOS5zZXRDb250ZW50KGh0bWxfMjEyYWYzODk5ZGQ5NGQ3ZGFkMjBmY2JkZDc0NWRlN2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjI4MzQ4MzE2OTM1NDczZTgwMjFiNWZiODQwYWE4Y2YuYmluZFBvcHVwKHBvcHVwXzY1YWVjNzI5ZWNmZDQ2MjI4YTE4MzYzMTYzYjk1OTI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRhYjMwNTU0ZjBlMjQyODU5OTQxYjg1ZWVkNmJhMWFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDM5NzQzNSwtMC4zNDEyNjM0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUyMGNiNmJiNjVlYzRlZjViMmQ0MDM5MzE2ZmVmNDBlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEzN2NlYTk1MWZlZTQxZDdhNDE5NDhmM2E5NzFhZTljID0gJCgnPGRpdiBpZD0iaHRtbF8xMzdjZWE5NTFmZWU0MWQ3YTQxOTQ4ZjNhOTcxYWU5YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0FMUE9MRSBHQVJERU5TLCAyMzAzNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUyMGNiNmJiNjVlYzRlZjViMmQ0MDM5MzE2ZmVmNDBlLnNldENvbnRlbnQoaHRtbF8xMzdjZWE5NTFmZWU0MWQ3YTQxOTQ4ZjNhOTcxYWU5Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YWIzMDU1NGYwZTI0Mjg1OTk0MWI4NWVlZDZiYTFhZi5iaW5kUG9wdXAocG9wdXBfNTIwY2I2YmI2NWVjNGVmNWIyZDQwMzkzMTZmZWY0MGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGY0OTAyMTE0ZGUyNGE4NWEyODhiY2M5Y2M4YTU3NDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi42MjYxODExLDEuMjg1OTMxMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYjc4YzM4N2MzMDg0NzgxYTc4MTg0MDJjNTRiMjc5ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZGQ2ZjY4NmFlMWE0NWE2OWVmNzgwY2EyMDc5YzAwYyA9ICQoJzxkaXYgaWQ9Imh0bWxfZWRkNmY2ODZhZTFhNDVhNjllZjc4MGNhMjA3OWMwMGMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldBTFBPTEUgU1RSRUVULCAyMjQyNTAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFiNzhjMzg3YzMwODQ3ODFhNzgxODQwMmM1NGIyNzlmLnNldENvbnRlbnQoaHRtbF9lZGQ2ZjY4NmFlMWE0NWE2OWVmNzgwY2EyMDc5YzAwYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZjQ5MDIxMTRkZTI0YTg1YTI4OGJjYzljYzhhNTc0NC5iaW5kUG9wdXAocG9wdXBfMWI3OGMzODdjMzA4NDc4MWE3ODE4NDAyYzU0YjI3OWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTM1ODJmMDI2NzRjNDMxMGI3YWVhNDBlOTY4NDhiOWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMy43Mzg1NjMyLC0xMTcuODQ2NTA1Njc3NDY0ODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTJlMTA0ZTMyNDk4NDk5ZjkwNDkyNzgwY2RiZDM5ZDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTZiMDVmOTFhOGQ1NGZmNGE1NmY1OGM1NTdlOTM3MmMgPSAkKCc8ZGl2IGlkPSJodG1sXzk2YjA1ZjkxYThkNTRmZjRhNTZmNThjNTU3ZTkzNzJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XQVJXSUNLIFNRVUFSRSwgMjQzMjI3Mi43MjcyNzI3Mjc8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UyZTEwNGUzMjQ5ODQ5OWY5MDQ5Mjc4MGNkYmQzOWQ3LnNldENvbnRlbnQoaHRtbF85NmIwNWY5MWE4ZDU0ZmY0YTU2ZjU4YzU1N2U5MzcyYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MzU4MmYwMjY3NGM0MzEwYjdhZWE0MGU5Njg0OGI5ZC5iaW5kUG9wdXAocG9wdXBfZTJlMTA0ZTMyNDk4NDk5ZjkwNDkyNzgwY2RiZDM5ZDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmI0ZjY5M2FkZjkwNDdjMzhlNWRmYzQ5YzUxMDlhYmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi41NTgwMDUzLC0wLjI2MjMwOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYWVmZWViMTFkMGM0MjBjYjRkNDM0NjRmNGE3MTFmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOWMwNGYzNDFkNzI0YTZmYWNmYThhZTUwNjViMTRlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzljMDRmMzQxZDcyNGE2ZmFjZmE4YWU1MDY1YjE0ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFTEJFQ0sgV0FZLCAyMjY3MDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VhZWZlZWIxMWQwYzQyMGNiNGQ0MzQ2NGY0YTcxMWY3LnNldENvbnRlbnQoaHRtbF8zOWMwNGYzNDFkNzI0YTZmYWNmYThhZTUwNjViMTRlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mYjRmNjkzYWRmOTA0N2MzOGU1ZGZjNDljNTEwOWFiZC5iaW5kUG9wdXAocG9wdXBfZWFlZmVlYjExZDBjNDIwY2I0ZDQzNDY0ZjRhNzExZjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjVhY2FkMzI0NzhkNGFhMTg4MGEzODA2MjliZmJkNjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjkyMTQyLC0wLjA5MjkzMjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfYThmYmZjYTE5N2Y4NGUwZGE0ZTVhZjViZTU1NzM2ZjcpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGM0Njk0ZjY3OWFjNDU3ZWIxNDJmMThlYTkyYjVjZjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDU1YTgzYTM1OWZmNDEzNGFmMGRlZmVlYTg3MzAyNjYgPSAkKCc8ZGl2IGlkPSJodG1sXzA1NWE4M2EzNTlmZjQxMzRhZjBkZWZlZWE4NzMwMjY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XRUxMRVNMRVkgVEVSUkFDRSwgMjQxMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84YzQ2OTRmNjc5YWM0NTdlYjE0MmYxOGVhOTJiNWNmNy5zZXRDb250ZW50KGh0bWxfMDU1YTgzYTM1OWZmNDEzNGFmMGRlZmVlYTg3MzAyNjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjVhY2FkMzI0NzhkNGFhMTg4MGEzODA2MjliZmJkNjYuYmluZFBvcHVwKHBvcHVwXzhjNDY5NGY2NzlhYzQ1N2ViMTQyZjE4ZWE5MmI1Y2Y3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QyYTNlMzJmYmUxNDRhMjNhY2Q4OTYwOTQ0MTQ0MTI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDUuNDIzNDQ5NSwtNzUuNjk4MDU3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MjIxZDIyOGEzOGE0ZmVlOGI0MTJiN2FmZjI3MDg3NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMTViNDA2NmNmMTM0NjY5OGMyNGY4NjQ2YzkyNWQxZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMDE1YjQwNjZjZjEzNDY2OThjMjRmODY0NmM5MjVkMWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFTExJTkdUT04gU1RSRUVULCAyMjkzMTU1LjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUyMjFkMjI4YTM4YTRmZWU4YjQxMmI3YWZmMjcwODc2LnNldENvbnRlbnQoaHRtbF8wMTViNDA2NmNmMTM0NjY5OGMyNGY4NjQ2YzkyNWQxZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMmEzZTMyZmJlMTQ0YTIzYWNkODk2MDk0NDE0NDEyNC5iaW5kUG9wdXAocG9wdXBfNTIyMWQyMjhhMzhhNGZlZThiNDEyYjdhZmYyNzA4NzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzU1OGUyZWU3ZWJlNGVjNzk0NWU0MmRjNDgzYTU2ZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MC43NDAyMjQyLC0xMTEuODQ2MzIzNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZTRkODBiY2M3OWY0ZDkxYmM4YjcxNjI2Yjg1YTQyYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85N2Y5NTc3MmJmZjQ0YjMyYWE0OTIxYmVjMDViODIyYSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTdmOTU3NzJiZmY0NGIzMmFhNDkyMWJlYzA1YjgyMmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFU1RNT1JFTEFORCBQTEFDRSwgMjMwMDAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZTRkODBiY2M3OWY0ZDkxYmM4YjcxNjI2Yjg1YTQyYi5zZXRDb250ZW50KGh0bWxfOTdmOTU3NzJiZmY0NGIzMmFhNDkyMWJlYzA1YjgyMmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzU1OGUyZWU3ZWJlNGVjNzk0NWU0MmRjNDgzYTU2ZWEuYmluZFBvcHVwKHBvcHVwXzBlNGQ4MGJjYzc5ZjRkOTFiYzhiNzE2MjZiODVhNDJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQxZjQwNTdlMDE0NTQ2M2U5YmI0ZDIxMDM5NDllMzQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTIyOTQzNywtMC4xMzc5MzExXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IxMzVjM2ZkMmI5NjQwNWJiMjIxMDcyODhkZDJmMzhiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhZTBlM2Q2MWMwMTQ4Y2RiOTljMjJlZjBjMWNjZTg1ID0gJCgnPGRpdiBpZD0iaHRtbF80YWUwZTNkNjFjMDE0OGNkYjk5YzIyZWYwYzFjY2U4NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0hJVEZJRUxEIFNUUkVFVCwgMjQ1MTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMTM1YzNmZDJiOTY0MDViYjIyMTA3Mjg4ZGQyZjM4Yi5zZXRDb250ZW50KGh0bWxfNGFlMGUzZDYxYzAxNDhjZGI5OWMyMmVmMGMxY2NlODUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDFmNDA1N2UwMTQ1NDYzZTliYjRkMjEwMzk0OWUzNDQuYmluZFBvcHVwKHBvcHVwX2IxMzVjM2ZkMmI5NjQwNWJiMjIxMDcyODhkZDJmMzhiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UzODU4MDg3ZGIzMDQxZWU5ZTk0Y2IxMDY2MTRjMDcxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk4ODM5NywtMC4xMzkyOTQ5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1MDYwM2E2ZmQyYzRmOTQ5MGQ3ZWFmOWY2NzFiOTA0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhYjljYzFjNThlNDQxODhhNDFlOGYzODlhMjVlZmNlID0gJCgnPGRpdiBpZD0iaHRtbF9mYWI5Y2MxYzU4ZTQ0MTg4YTQxZThmMzg5YTI1ZWZjZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0lMRlJFRCBTVFJFRVQsIDI0MTA1MzguNTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTUwNjAzYTZmZDJjNGY5NDkwZDdlYWY5ZjY3MWI5MDQuc2V0Q29udGVudChodG1sX2ZhYjljYzFjNThlNDQxODhhNDFlOGYzODlhMjVlZmNlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UzODU4MDg3ZGIzMDQxZWU5ZTk0Y2IxMDY2MTRjMDcxLmJpbmRQb3B1cChwb3B1cF9lNTA2MDNhNmZkMmM0Zjk0OTBkN2VhZjlmNjcxYjkwNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMjk3ZTk3ZjUxMzU0NWNjYTExMTI5MGU0NjhiM2EzYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU0MzEwODgsLTAuMDk1NTA3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9hOGZiZmNhMTk3Zjg0ZTBkYTRlNWFmNWJlNTU3MzZmNyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYmYyODJlMzg2NDg0YTdlYmUxNjMzMTU3NTg2OWM5NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lOWRhNjVjZjJkMDM0MDFiOGZiZjEzYTlkNDViZWY2ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTlkYTY1Y2YyZDAzNDAxYjhmYmYxM2E5ZDQ1YmVmNmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldJTExPVyBCUklER0UgUk9BRCwgMjQyNTAwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYmYyODJlMzg2NDg0YTdlYmUxNjMzMTU3NTg2OWM5Ni5zZXRDb250ZW50KGh0bWxfZTlkYTY1Y2YyZDAzNDAxYjhmYmYxM2E5ZDQ1YmVmNmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDI5N2U5N2Y1MTM1NDVjY2ExMTEyOTBlNDY4YjNhM2EuYmluZFBvcHVwKHBvcHVwX2FiZjI4MmUzODY0ODRhN2ViZTE2MzMxNTc1ODY5Yzk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE1ZGQxOTZmNzdjODQ5NTZiMzc1ZDZmZWIxOWUyYTRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzAuNTk3Nzk3MywtODEuNTk1NzU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgzMGViYjIxNzcyMTRlZGNiZWE5MGI3MzAyZjgxMDUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1ZWMwMjlhNDIxZTRlOTRhODc3Mzg0NDI3ZDgwNmZiID0gJCgnPGRpdiBpZD0iaHRtbF85NWVjMDI5YTQyMWU0ZTk0YTg3NzM4NDQyN2Q4MDZmYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0lMU09OIFNUUkVFVCwgMjI1NzUwMC4wPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84MzBlYmIyMTc3MjE0ZWRjYmVhOTBiNzMwMmY4MTA1MS5zZXRDb250ZW50KGh0bWxfOTVlYzAyOWE0MjFlNGU5NGE4NzczODQ0MjdkODA2ZmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTVkZDE5NmY3N2M4NDk1NmIzNzVkNmZlYjE5ZTJhNGEuYmluZFBvcHVwKHBvcHVwXzgzMGViYjIxNzcyMTRlZGNiZWE5MGI3MzAyZjgxMDUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzExZDkyMDRjMTExOTRmODhhMjlhOGM2NjBhYTYyYWY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDMyOTA3NCwtMC4zNDg0NTQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VlZGZkNDNiNmFlNTQ5MjdhZWRhMGFiOTc0YTI1ZTQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyMDFlNTA3MDA1MzQ0MzA5ZmRjYjBmMTQ1OTQwNWJkID0gJCgnPGRpdiBpZD0iaHRtbF82MjAxZTUwNzAwNTM0NDMwOWZkY2IwZjE0NTk0MDViZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0lOQ0hFTkRPTiBST0FELCAyMzUwMDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VlZGZkNDNiNmFlNTQ5MjdhZWRhMGFiOTc0YTI1ZTQ5LnNldENvbnRlbnQoaHRtbF82MjAxZTUwNzAwNTM0NDMwOWZkY2IwZjE0NTk0MDViZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMWQ5MjA0YzExMTk0Zjg4YTI5YThjNjYwYWE2MmFmNy5iaW5kUG9wdXAocG9wdXBfZWVkZmQ0M2I2YWU1NDkyN2FlZGEwYWI5NzRhMjVlNDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDY2ODNkMThhMjk3NDljNjg2NzNiZjg3OWMwZGM4MjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4wOTI1NTcsMS4xNzk0NTU0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2E4ZmJmY2ExOTdmODRlMGRhNGU1YWY1YmU1NTczNmY3KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVlYTFlOTg2MWU0YzQ1NzA5NjM2NmNiOWE5YTdmM2M2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYzNzc5NjA1ZWYwMjQ5MjVhY2IxZDg0ZWQxYzc1NmFlID0gJCgnPGRpdiBpZD0iaHRtbF82Mzc3OTYwNWVmMDI0OTI1YWNiMWQ4NGVkMWM3NTZhZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0lOR0FURSBST0FELCAyMjA2NDAwLjA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVlYTFlOTg2MWU0YzQ1NzA5NjM2NmNiOWE5YTdmM2M2LnNldENvbnRlbnQoaHRtbF82Mzc3OTYwNWVmMDI0OTI1YWNiMWQ4NGVkMWM3NTZhZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NjY4M2QxOGEyOTc0OWM2ODY3M2JmODc5YzBkYzgyNi5iaW5kUG9wdXAocG9wdXBfNWVhMWU5ODYxZTRjNDU3MDk2MzY2Y2I5YTlhN2YzYzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg==" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python

#Define Foursquare Credentials and Version

CLIENT_ID = 'RVTNRWS3NKMHS3MB3TTASSNQTIVZQH4BL5H4Y0XIBSTNDKS1' # Foursquare ID
CLIENT_SECRET = 'GDIRZLF3DNHJRVO3VLG1XJQALXN0GZKNJ3PDJZ2JWG1NAJS3' # Foursquare Secret
VERSION = '20191206' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

```

    Your credentails:
    CLIENT_ID: RVTNRWS3NKMHS3MB3TTASSNQTIVZQH4BL5H4Y0XIBSTNDKS1
    CLIENT_SECRET:GDIRZLF3DNHJRVO3VLG1XJQALXN0GZKNJ3PDJZ2JWG1NAJS3


We can now proceed to the Modeling phase. We will analyze neighborhoods to recommend real estates where home buyers can make a real estate investment. We will then recommend profitable venues according to amenities and essential facilities surrounding such venues i.e. elementary schools, high schools, hospitals & grocery stores.

#### Modeling

After exploring the dataset and gaining insights into it, we are ready to use the clustering methodology to analyze real estates. We will use the k-means clustering technique as it is fast and efficient in terms of computational cost, is highly flexible to account for mutations in real estate market in London and is accurate.


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500, LIMIT=100):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
  # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Street', 
                  'Street Latitude', 
                  'Street Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```


```python
# Run the above function on each location and create a new dataframe called location_venues and display it.
location_venues = getNearbyVenues(names=df1['Street'],
                                   latitudes=df1['Latitude'],
                                   longitudes=df1['Longitude']
                                  )
```

    ALBION SQUARE
    ANHALT ROAD
    ANSDELL TERRACE
    APPLEGARTH ROAD
    BARONSMEAD ROAD
    BEAUCLERC ROAD
    BELVEDERE DRIVE
    BICKENHALL STREET
    BIRCHLANDS AVENUE
    BRAMPTON GROVE
    BRIARDALE GARDENS
    BROOKWAY
    BURBAGE ROAD
    BURY WALK
    CALLCOTT STREET
    CAMPDEN HILL ROAD
    CAMPION ROAD
    CANNING PLACE
    CARLISLE ROAD
    CARLTON GARDENS
    CARLYLE COURT
    CHALCOT SQUARE
    CHARLES LANE
    CHELSEA CRESCENT
    CHESTER CLOSE NORTH
    CHEYNE COURT
    CHEYNE ROW
    CHISWICK MALL
    CITY ROAD
    CLARENDON STREET
    CLONCURRY STREET
    COLBECK MEWS
    COLLEGE CRESCENT
    CORNWALL TERRACE MEWS
    COURT LANE GARDENS
    CRESCENT GROVE
    DALEBURY ROAD
    DEWHURST ROAD
    DORIA ROAD
    DOWNSHIRE HILL
    DUCHESS WALK
    ECCLESTON SQUARE MEWS
    EGBERT STREET
    EGERTON PLACE
    ELM PARK ROAD
    FLORAL STREET
    FRANK DIXON WAY
    FULTON MEWS
    GERARD ROAD
    GERRARD ROAD
    GIRDLERS ROAD
    GLOUCESTER CRESCENT
    GORDON PLACE
    GRAFTON SQUARE
    GRAHAM TERRACE
    HARMAN DRIVE
    HARRIS STREET
    HAVANNAH STREET
    HAZLEWELL ROAD
    HEREFORD MEWS
    HERONDALE AVENUE
    HIGHGATE HIGH STREET
    HIGHWOOD HILL
    HILLGATE PLACE
    HOLLYCROFT AVENUE
    HOLLYWOOD MEWS
    HONEYWELL ROAD
    HORTENSIA ROAD
    HOXTON SQUARE
    HUNTER ROAD
    JACKSONS LANE
    JOHN STREET
    KINNERTON STREET
    KNARESBOROUGH PLACE
    KNOX STREET
    LADBROKE GROVE
    LANCASTER MEWS
    LANSDOWNE ROAD
    LATIMER INDUSTRIAL ESTATE
    LAXTON PLACE
    LINCOLN AVENUE
    LINGFIELD ROAD
    LISSON STREET
    LIVERPOOL GROVE
    LONGWOOD DRIVE
    LONSDALE SQUARE
    MAZE HILL
    MIDDLESEX PASSAGE
    MONTPELIER AVENUE
    MONTPELIER WALK
    MULTON ROAD
    MUNDEN STREET
    NORFOLK CRESCENT
    NORTH CIRCULAR ROAD
    NOTTINGHAM STREET
    OAKLEY STREET
    OAKWOOD COURT
    OBSERVATORY GARDENS
    OLD COURT PLACE
    ONSLOW MEWS WEST
    PALACE PLACE
    PANTON STREET
    PARK CRESCENT
    PARK LANE
    PARKE ROAD
    PARKFIELDS
    PARTHENIA ROAD
    PAVILION ROAD
    PEMBRIDGE MEWS
    PEMBRIDGE ROAD
    PEMBROKE STUDIOS
    PENCOMBE MEWS
    PETERSHAM PLACE
    PHILLIMORE GARDENS
    PHYSIC PLACE
    PITFIELD STREET
    PRINCES GATE
    PRIORY ROAD
    PROTHERO GARDENS
    PUTNEY HIGH STREET
    QUARRENDON STREET
    QUEENS GATE TERRACE
    RADSTOCK STREET
    RANELAGH AVENUE
    REDCLIFFE ROAD
    REEVES MEWS
    RHEIDOL MEWS
    RINGWOOD AVENUE
    RODERICK ROAD
    ROPEMAKERS FIELDS
    ROYAL CRESCENT
    ROYAL HILL
    RUSSELL GARDENS MEWS
    SETTLES STREET
    SHELDON AVENUE
    SOUTH END ROW
    SOUTHWOOD LAWN ROAD
    SOVEREIGN PARK
    ST MARGARETS CRESCENT
    ST OSWALDS PLACE
    ST PETERS SQUARE
    STAFFORD TERRACE
    SUTHERLAND PLACE
    SYDNEY STREET
    THAMES BANK
    THE HEXAGON
    TREDEGAR SQUARE
    TRINITY STREET
    UPPER HAMPSTEAD WALK
    WALPOLE GARDENS
    WALPOLE STREET
    WARWICK SQUARE
    WELBECK WAY
    WELLESLEY TERRACE
    WELLINGTON STREET
    WESTMORELAND PLACE
    WHITFIELD STREET
    WILFRED STREET
    WILLOW BRIDGE ROAD
    WILSON STREET
    WINCHENDON ROAD
    WINGATE ROAD



```python
location_venues
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
      <th>Street</th>
      <th>Street Latitude</th>
      <th>Street Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Free House</td>
      <td>-41.273340</td>
      <td>173.287364</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Indian Cafe</td>
      <td>-41.273308</td>
      <td>173.286530</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Queen's Gardens</td>
      <td>-41.273671</td>
      <td>173.291383</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Deville Cafe</td>
      <td>-41.271941</td>
      <td>173.285535</td>
      <td>Beer Garden</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Urban</td>
      <td>-41.274355</td>
      <td>173.286317</td>
      <td>New American Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Burger Culture</td>
      <td>-41.274750</td>
      <td>173.284030</td>
      <td>Burger Joint</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Fish Stop</td>
      <td>-41.276010</td>
      <td>173.289592</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Vic Mac's Brew Bar</td>
      <td>-41.274757</td>
      <td>173.283914</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Bridge Street Collective</td>
      <td>-41.272520</td>
      <td>173.285517</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Hopgood's</td>
      <td>-41.274749</td>
      <td>173.283831</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Mango</td>
      <td>-41.274460</td>
      <td>173.285345</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>cod and lobster</td>
      <td>-41.275203</td>
      <td>173.283747</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Fresh Choice</td>
      <td>-41.272194</td>
      <td>173.287218</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Sprig &amp; Fern</td>
      <td>-41.274508</td>
      <td>173.286527</td>
      <td>Brewery</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Lambretta's Cafe &amp; Bar</td>
      <td>-41.274372</td>
      <td>173.284462</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Kitchen</td>
      <td>-41.272360</td>
      <td>173.285500</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Morrison Street Cafe</td>
      <td>-41.274505</td>
      <td>173.285461</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Columbus Coffee</td>
      <td>-41.274759</td>
      <td>173.285391</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Suter Art Gallery</td>
      <td>-41.273665</td>
      <td>173.291377</td>
      <td>Art Gallery</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>La Gourmandise</td>
      <td>-41.274262</td>
      <td>173.286211</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>20</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Ford's Restaurant &amp; Bar</td>
      <td>-41.274637</td>
      <td>173.283851</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>7010</td>
      <td>-41.270045</td>
      <td>173.286959</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>22</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Robert Harris Coffee</td>
      <td>-41.272941</td>
      <td>173.283876</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>23</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Nelson Provincial Museum</td>
      <td>-41.274486</td>
      <td>173.283911</td>
      <td>Museum</td>
    </tr>
    <tr>
      <th>24</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>623 In the City</td>
      <td>-41.274049</td>
      <td>173.285020</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>The Prince Albert</td>
      <td>-41.276275</td>
      <td>173.290930</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ALBION SQUARE</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>Starbucks</td>
      <td>-41.273502</td>
      <td>173.283919</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ANHALT ROAD</td>
      <td>51.480316</td>
      <td>-0.166801</td>
      <td>Bayley &amp; Sage</td>
      <td>51.479074</td>
      <td>-0.167052</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>28</th>
      <td>ANHALT ROAD</td>
      <td>51.480316</td>
      <td>-0.166801</td>
      <td>Nutbourne</td>
      <td>51.479305</td>
      <td>-0.168169</td>
      <td>English Restaurant</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ANHALT ROAD</td>
      <td>51.480316</td>
      <td>-0.166801</td>
      <td>The Prince Albert</td>
      <td>51.479602</td>
      <td>-0.165586</td>
      <td>Pub</td>
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
    </tr>
    <tr>
      <th>5916</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Compton Arms</td>
      <td>51.543680</td>
      <td>-0.102013</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>5917</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Canonbury Square</td>
      <td>51.543994</td>
      <td>-0.099789</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>5918</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Palmera Oasis</td>
      <td>51.543634</td>
      <td>-0.090384</td>
      <td>Middle Eastern Restaurant</td>
    </tr>
    <tr>
      <th>5919</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Fig Tree</td>
      <td>51.546947</td>
      <td>-0.098576</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>5920</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Yida Sushi</td>
      <td>51.546676</td>
      <td>-0.099242</td>
      <td>Sushi Restaurant</td>
    </tr>
    <tr>
      <th>5921</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>St. Paul</td>
      <td>51.546313</td>
      <td>-0.100447</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>5922</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Coconut Grove</td>
      <td>51.546963</td>
      <td>-0.098631</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>5923</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>New London</td>
      <td>51.546892</td>
      <td>-0.098447</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>5924</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>The Lord Clyde</td>
      <td>51.543792</td>
      <td>-0.090137</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>5925</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Tootoomoo</td>
      <td>51.546540</td>
      <td>-0.099482</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>5926</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>51.543109</td>
      <td>-0.095508</td>
      <td>Nightingale Park</td>
      <td>51.545019</td>
      <td>-0.090777</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>5927</th>
      <td>WILSON STREET</td>
      <td>30.597797</td>
      <td>-81.595757</td>
      <td>Subway</td>
      <td>30.600741</td>
      <td>-81.599182</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>5928</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Imperial China Restaurant</td>
      <td>51.432460</td>
      <td>-0.347984</td>
      <td>Chinese Restaurant</td>
    </tr>
    <tr>
      <th>5929</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Squires Garden Centre</td>
      <td>51.435512</td>
      <td>-0.351533</td>
      <td>Garden Center</td>
    </tr>
    <tr>
      <th>5930</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>The Old Goat</td>
      <td>51.436101</td>
      <td>-0.350581</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>5931</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Brouge at The Old Goat</td>
      <td>51.436088</td>
      <td>-0.350445</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5932</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Domino's Pizza</td>
      <td>51.432442</td>
      <td>-0.347993</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>5933</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Domino's Pizza</td>
      <td>51.431722</td>
      <td>-0.345421</td>
      <td>Pizza Place</td>
    </tr>
    <tr>
      <th>5934</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Platform 1</td>
      <td>51.433813</td>
      <td>-0.349559</td>
      <td>Platform</td>
    </tr>
    <tr>
      <th>5935</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Bengal Brasserie</td>
      <td>51.432368</td>
      <td>-0.347982</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>5936</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Twickenham Depot</td>
      <td>51.432384</td>
      <td>-0.348018</td>
      <td>Bus Station</td>
    </tr>
    <tr>
      <th>5937</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Cavan Bakery</td>
      <td>51.432426</td>
      <td>-0.348010</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>5938</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Caffe Toscana</td>
      <td>51.432430</td>
      <td>-0.348008</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>5939</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Regency Fish Bar</td>
      <td>51.431236</td>
      <td>-0.345627</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>5940</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>Fulwell Golf Club</td>
      <td>51.433539</td>
      <td>-0.352425</td>
      <td>Golf Course</td>
    </tr>
    <tr>
      <th>5941</th>
      <td>WINCHENDON ROAD</td>
      <td>51.432907</td>
      <td>-0.348455</td>
      <td>The Red Lion</td>
      <td>51.430897</td>
      <td>-0.345217</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>5942</th>
      <td>WINGATE ROAD</td>
      <td>51.092557</td>
      <td>1.179455</td>
      <td>Currys PC World</td>
      <td>51.093848</td>
      <td>1.179104</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>5943</th>
      <td>WINGATE ROAD</td>
      <td>51.092557</td>
      <td>1.179455</td>
      <td>The Hungry Horse</td>
      <td>51.089400</td>
      <td>1.180624</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>5944</th>
      <td>WINGATE ROAD</td>
      <td>51.092557</td>
      <td>1.179455</td>
      <td>Nisa Local</td>
      <td>51.095157</td>
      <td>1.184555</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>5945</th>
      <td>WINGATE ROAD</td>
      <td>51.092557</td>
      <td>1.179455</td>
      <td>Booker</td>
      <td>51.092661</td>
      <td>1.172383</td>
      <td>Warehouse Store</td>
    </tr>
  </tbody>
</table>
<p>5946 rows × 7 columns</p>
</div>




```python
location_venues.groupby('Street').count()
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
      <th>Street Latitude</th>
      <th>Street Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Street</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ALBION SQUARE</th>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
    </tr>
    <tr>
      <th>ANHALT ROAD</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>ANSDELL TERRACE</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>APPLEGARTH ROAD</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>BARONSMEAD ROAD</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>BEAUCLERC ROAD</th>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
      <td>29</td>
    </tr>
    <tr>
      <th>BELVEDERE DRIVE</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>BICKENHALL STREET</th>
      <td>91</td>
      <td>91</td>
      <td>91</td>
      <td>91</td>
      <td>91</td>
      <td>91</td>
    </tr>
    <tr>
      <th>BIRCHLANDS AVENUE</th>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
    <tr>
      <th>BRAMPTON GROVE</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>BRIARDALE GARDENS</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>BROOKWAY</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>BURBAGE ROAD</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>BURY WALK</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>CALLCOTT STREET</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>CAMPDEN HILL ROAD</th>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
    </tr>
    <tr>
      <th>CAMPION ROAD</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>CANNING PLACE</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>CARLISLE ROAD</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>CARLTON GARDENS</th>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
      <td>61</td>
    </tr>
    <tr>
      <th>CARLYLE COURT</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>CHALCOT SQUARE</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>CHARLES LANE</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>CHESTER CLOSE NORTH</th>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
      <td>21</td>
    </tr>
    <tr>
      <th>CHEYNE COURT</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CHEYNE ROW</th>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
      <td>58</td>
    </tr>
    <tr>
      <th>CHISWICK MALL</th>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>CITY ROAD</th>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
      <td>32</td>
    </tr>
    <tr>
      <th>CLARENDON STREET</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>CLONCURRY STREET</th>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
      <td>30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ROPEMAKERS FIELDS</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>ROYAL CRESCENT</th>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
    </tr>
    <tr>
      <th>RUSSELL GARDENS MEWS</th>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
      <td>25</td>
    </tr>
    <tr>
      <th>SETTLES STREET</th>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
      <td>70</td>
    </tr>
    <tr>
      <th>SOUTH END ROW</th>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
      <td>54</td>
    </tr>
    <tr>
      <th>SOUTHWOOD LAWN ROAD</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>SOVEREIGN PARK</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ST OSWALDS PLACE</th>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
      <td>40</td>
    </tr>
    <tr>
      <th>ST PETERS SQUARE</th>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
      <td>86</td>
    </tr>
    <tr>
      <th>STAFFORD TERRACE</th>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
    </tr>
    <tr>
      <th>SUTHERLAND PLACE</th>
      <td>62</td>
      <td>62</td>
      <td>62</td>
      <td>62</td>
      <td>62</td>
      <td>62</td>
    </tr>
    <tr>
      <th>SYDNEY STREET</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>THAMES BANK</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>THE HEXAGON</th>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
      <td>43</td>
    </tr>
    <tr>
      <th>TREDEGAR SQUARE</th>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
    </tr>
    <tr>
      <th>TRINITY STREET</th>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>UPPER HAMPSTEAD WALK</th>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
      <td>63</td>
    </tr>
    <tr>
      <th>WALPOLE GARDENS</th>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>WALPOLE STREET</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>WARWICK SQUARE</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>WELBECK WAY</th>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>WELLESLEY TERRACE</th>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
      <td>51</td>
    </tr>
    <tr>
      <th>WELLINGTON STREET</th>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
    </tr>
    <tr>
      <th>WESTMORELAND PLACE</th>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
      <td>18</td>
    </tr>
    <tr>
      <th>WHITFIELD STREET</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>WILFRED STREET</th>
      <td>80</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
      <td>80</td>
    </tr>
    <tr>
      <th>WILLOW BRIDGE ROAD</th>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>WILSON STREET</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>WINCHENDON ROAD</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>WINGATE ROAD</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>155 rows × 6 columns</p>
</div>




```python
# get the List of Unique Categories
print('There are {} uniques categories.'.format(len(location_venues['Venue Category'].unique())))
location_venues.shape
```

    There are 342 uniques categories.





    (5946, 7)




```python
# one hot encoding
venues_onehot = pd.get_dummies(location_venues[['Venue Category']], prefix="", prefix_sep="")

# add street column back to dataframe
venues_onehot['Street'] = location_venues['Street'] 

# move street column to the first column
fixed_columns = [venues_onehot.columns[-1]] + list(venues_onehot.columns[:-1])

#fixed_columns
venues_onehot = venues_onehot[fixed_columns]

venues_onehot.head()
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
      <th>Street</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Arcade</th>
      <th>Argentinian Restaurant</th>
      <th>...</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Waterfront</th>
      <th>Windmill</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALBION SQUARE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ALBION SQUARE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ALBION SQUARE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ALBION SQUARE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ALBION SQUARE</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 343 columns</p>
</div>




```python
london_grouped = venues_onehot.groupby('Street').mean().reset_index()
london_grouped
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
      <th>Street</th>
      <th>ATM</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>African Restaurant</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>Arcade</th>
      <th>Argentinian Restaurant</th>
      <th>...</th>
      <th>Video Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Warehouse Store</th>
      <th>Waterfront</th>
      <th>Windmill</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
      <th>Zoo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALBION SQUARE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ANHALT ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ANSDELL TERRACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>APPLEGARTH ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BARONSMEAD ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BEAUCLERC ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>BELVEDERE DRIVE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BICKENHALL STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010989</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BRAMPTON GROVE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BRIARDALE GARDENS</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BROOKWAY</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BURBAGE ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>BURY WALK</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CALLCOTT STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>CAMPION ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>CANNING PLACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>CARLISLE ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>CARLTON GARDENS</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032787</td>
      <td>0.000000</td>
      <td>0.016393</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>CARLYLE COURT</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CHALCOT SQUARE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>CHARLES LANE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030303</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>CHEYNE COURT</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>CHEYNE ROW</td>
      <td>0.0</td>
      <td>0.017241</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.017241</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.017241</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.017241</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>CHISWICK MALL</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CITY ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>CLARENDON STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>CLONCURRY STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.033333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
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
    </tr>
    <tr>
      <th>125</th>
      <td>ROPEMAKERS FIELDS</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>126</th>
      <td>ROYAL CRESCENT</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>127</th>
      <td>RUSSELL GARDENS MEWS</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>128</th>
      <td>SETTLES STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.014286</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>129</th>
      <td>SOUTH END ROW</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>130</th>
      <td>SOUTHWOOD LAWN ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>131</th>
      <td>SOVEREIGN PARK</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>132</th>
      <td>ST OSWALDS PLACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>ST PETERS SQUARE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011628</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>134</th>
      <td>STAFFORD TERRACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>135</th>
      <td>SUTHERLAND PLACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.016129</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016129</td>
      <td>0.000000</td>
      <td>0.032258</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>136</th>
      <td>SYDNEY STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>137</th>
      <td>THAMES BANK</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>138</th>
      <td>THE HEXAGON</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.023256</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>139</th>
      <td>TREDEGAR SQUARE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>140</th>
      <td>TRINITY STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.083333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>UPPER HAMPSTEAD WALK</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.015873</td>
      <td>0.015873</td>
      <td>0.0</td>
      <td>0.015873</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.015873</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>142</th>
      <td>WALPOLE GARDENS</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>143</th>
      <td>WALPOLE STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>144</th>
      <td>WARWICK SQUARE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>WELBECK WAY</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.20</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>146</th>
      <td>WELLESLEY TERRACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.039216</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>147</th>
      <td>WELLINGTON STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>WESTMORELAND PLACE</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.055556</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149</th>
      <td>WHITFIELD STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.010000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>150</th>
      <td>WILFRED STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>151</th>
      <td>WILLOW BRIDGE ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>152</th>
      <td>WILSON STREET</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>WINCHENDON ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>154</th>
      <td>WINGATE ROAD</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.25</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>155 rows × 343 columns</p>
</div>




```python
london_grouped.shape
```




    (155, 343)




```python
# What are the top 5 venues/facilities nearby profitable real estate investments?#

num_top_venues = 5

for hood in london_grouped['Street']:
    print("----"+hood+"----")
    temp = london_grouped[london_grouped['Street'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----ALBION SQUARE----
                   venue  freq
    0               Café  0.22
    1         Restaurant  0.07
    2        Coffee Shop  0.07
    3                Pub  0.07
    4  Indian Restaurant  0.07
    
    
    ----ANHALT ROAD----
                      venue  freq
    0                   Pub  0.25
    1         Grocery Store  0.12
    2  Gym / Fitness Center  0.06
    3                 Plaza  0.06
    4                 Diner  0.06
    
    
    ----ANSDELL TERRACE----
                    venue  freq
    0               Hotel  0.06
    1  Italian Restaurant  0.06
    2          Restaurant  0.06
    3                 Pub  0.06
    4      Clothing Store  0.06
    
    
    ----APPLEGARTH ROAD----
                   venue  freq
    0                Pub  0.50
    1             Casino  0.25
    2          Nightclub  0.25
    3                ATM  0.00
    4  Outdoor Sculpture  0.00
    
    
    ----BARONSMEAD ROAD----
                   venue  freq
    0     Farmers Market  0.07
    1     Breakfast Spot  0.07
    2  Food & Drink Shop  0.07
    3          Bookstore  0.07
    4        Coffee Shop  0.07
    
    
    ----BEAUCLERC ROAD----
                 venue  freq
    0              Pub  0.14
    1      Coffee Shop  0.14
    2            Hotel  0.10
    3  Thai Restaurant  0.07
    4    Grocery Store  0.07
    
    
    ----BELVEDERE DRIVE----
                    venue  freq
    0                 Pub   0.2
    1  Seafood Restaurant   0.2
    2   Fish & Chips Shop   0.2
    3     Vacation Rental   0.2
    4          Campground   0.2
    
    
    ----BICKENHALL STREET----
                   venue  freq
    0              Hotel  0.05
    1        Coffee Shop  0.05
    2      Grocery Store  0.03
    3  Indian Restaurant  0.03
    4          Gastropub  0.03
    
    
    ----BIRCHLANDS AVENUE----
                   venue  freq
    0                Pub   0.2
    1  French Restaurant   0.1
    2            Brewery   0.1
    3     Breakfast Spot   0.1
    4             Bakery   0.1
    
    
    ----BRAMPTON GROVE----
                               venue  freq
    0                   Food Service   1.0
    1                            ATM   0.0
    2            Outdoor Event Space   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----BRIARDALE GARDENS----
                      venue  freq
    0    Seafood Restaurant  0.17
    1         Grocery Store  0.17
    2     Indian Restaurant  0.17
    3     Convenience Store  0.17
    4  Other Great Outdoors  0.17
    
    
    ----BROOKWAY----
                  venue  freq
    0  Asian Restaurant   0.5
    1       Art Gallery   0.5
    2               ATM   0.0
    3       Pastry Shop   0.0
    4              Park   0.0
    
    
    ----BURBAGE ROAD----
                    venue  freq
    0             Stadium  0.08
    1  Athletics & Sports  0.08
    2  Italian Restaurant  0.08
    3      Cricket Ground  0.08
    4         Pizza Place  0.08
    
    
    ----BURY WALK----
                    venue  freq
    0         Supermarket  0.21
    1  English Restaurant  0.14
    2                Café  0.07
    3                Park  0.07
    4                 Gym  0.07
    
    
    ----CALLCOTT STREET----
                   venue  freq
    0                Pub  0.15
    1               Park  0.06
    2        Pizza Place  0.04
    3  Indian Restaurant  0.04
    4              Hotel  0.04
    
    
    ----CAMPDEN HILL ROAD----
                    venue  freq
    0                Café  0.06
    1      Clothing Store  0.05
    2               Hotel  0.05
    3              Bakery  0.05
    4  Italian Restaurant  0.05
    
    
    ----CAMPION ROAD----
                      venue  freq
    0              Windmill  0.33
    1                   Spa  0.33
    2          Soccer Field  0.33
    3                   ATM  0.00
    4  Other Great Outdoors  0.00
    
    
    ----CANNING PLACE----
                    venue  freq
    0               Hotel  0.21
    1  Chinese Restaurant  0.06
    2   French Restaurant  0.06
    3  Italian Restaurant  0.06
    4      Clothing Store  0.06
    
    
    ----CARLISLE ROAD----
                   venue  freq
    0             Bakery  0.11
    1  Indian Restaurant  0.11
    2  Fish & Chips Shop  0.11
    3  Convenience Store  0.11
    4           Pharmacy  0.11
    
    
    ----CARLTON GARDENS----
                    venue  freq
    0  Italian Restaurant  0.23
    1        Dessert Shop  0.05
    2                Café  0.05
    3         Coffee Shop  0.03
    4      Ice Cream Shop  0.03
    
    
    ----CARLYLE COURT----
                               venue  freq
    0                           Farm   1.0
    1                            ATM   0.0
    2              Outdoor Sculpture   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----CHALCOT SQUARE----
                    venue  freq
    0                Café  0.08
    1                 Pub  0.06
    2                 Bar  0.06
    3  Italian Restaurant  0.06
    4         Coffee Shop  0.06
    
    
    ----CHARLES LANE----
                   venue  freq
    0     Cricket Ground  0.12
    1  French Restaurant  0.06
    2        Coffee Shop  0.06
    3                Pub  0.06
    4      Deli / Bodega  0.06
    
    
    ----CHESTER CLOSE NORTH----
              venue  freq
    0        Garden  0.14
    1          Park  0.10
    2  Cocktail Bar  0.10
    3         Hotel  0.05
    4        Lounge  0.05
    
    
    ----CHEYNE COURT----
                            venue  freq
    0  Construction & Landscaping   0.5
    1                   Gastropub   0.5
    2                         ATM   0.0
    3         Outdoor Event Space   0.0
    4                        Park   0.0
    
    
    ----CHEYNE ROW----
                      venue  freq
    0                  Café  0.09
    1    Italian Restaurant  0.05
    2                   Pub  0.05
    3                 Plaza  0.03
    4  Gym / Fitness Center  0.03
    
    
    ----CHISWICK MALL----
            venue  freq
    0         Pub  0.33
    1     Brewery  0.17
    2   Gift Shop  0.17
    3   Reservoir  0.17
    4  Art Museum  0.17
    
    
    ----CITY ROAD----
              venue  freq
    0           Pub  0.25
    1   Coffee Shop  0.09
    2   Art Gallery  0.06
    3          Park  0.06
    4  Dance Studio  0.06
    
    
    ----CLARENDON STREET----
                      venue  freq
    0     Indian Restaurant   0.4
    1  Fast Food Restaurant   0.2
    2         Grocery Store   0.2
    3             Pet Store   0.2
    4                   ATM   0.0
    
    
    ----CLONCURRY STREET----
               venue  freq
    0           Café  0.17
    1           Park  0.10
    2    Coffee Shop  0.07
    3  Grocery Store  0.07
    4   Cocktail Bar  0.03
    
    
    ----COLBECK MEWS----
                    venue  freq
    0               Hotel  0.24
    1                 Pub  0.07
    2              Garden  0.05
    3  Italian Restaurant  0.04
    4                Café  0.04
    
    
    ----COLLEGE CRESCENT----
                     venue  freq
    0                  Bar   0.2
    1  College Hockey Rink   0.2
    2         Liquor Store   0.2
    3                Diner   0.2
    4              Butcher   0.2
    
    
    ----CORNWALL TERRACE MEWS----
                 venue  freq
    0             Café  0.07
    1      Coffee Shop  0.07
    2           Garden  0.07
    3  Thai Restaurant  0.04
    4           Museum  0.04
    
    
    ----COURT LANE GARDENS----
                        venue  freq
    0                     Pub  0.13
    1           Grocery Store  0.13
    2                    Café  0.13
    3  Furniture / Home Store  0.07
    4                    Lake  0.07
    
    
    ----CRESCENT GROVE----
                               venue  freq
    0                          Hotel   1.0
    1                            ATM   0.0
    2            Outdoor Event Space   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----DALEBURY ROAD----
                   venue  freq
    0   Asian Restaurant  0.33
    1           Bus Stop  0.17
    2  Indian Restaurant  0.17
    3      Grocery Store  0.17
    4               Café  0.17
    
    
    ----DEWHURST ROAD----
             venue  freq
    0        Hotel  0.16
    1          Pub  0.11
    2  Pizza Place  0.05
    3    Gastropub  0.05
    4         Café  0.05
    
    
    ----DORIA ROAD----
                    venue  freq
    0  Italian Restaurant  0.09
    1                Café  0.08
    2              Bakery  0.06
    3                 Pub  0.06
    4         Coffee Shop  0.06
    
    
    ----DOWNSHIRE HILL----
                    venue  freq
    0                 Pub  0.13
    1                Café  0.13
    2  Italian Restaurant  0.08
    3           Bookstore  0.05
    4     Thai Restaurant  0.05
    
    
    ----DUCHESS WALK----
                    venue  freq
    0                 Pub  0.08
    1         Coffee Shop  0.06
    2                 Bar  0.06
    3  Italian Restaurant  0.04
    4        Cocktail Bar  0.03
    
    
    ----ECCLESTON SQUARE MEWS----
                    venue  freq
    0               Hotel  0.10
    1                 Pub  0.08
    2         Coffee Shop  0.06
    3  Italian Restaurant  0.06
    4                Café  0.05
    
    
    ----EGBERT STREET----
                               venue  freq
    0                            Pub   1.0
    1                            ATM   0.0
    2           Other Great Outdoors   0.0
    3  Paper / Office Supplies Store   0.0
    4                         Palace   0.0
    
    
    ----EGERTON PLACE----
                    venue  freq
    0                Café  0.12
    1  Italian Restaurant  0.11
    2            Boutique  0.06
    3               Hotel  0.05
    4         Coffee Shop  0.05
    
    
    ----ELM PARK ROAD----
               venue  freq
    0    Supermarket  0.17
    1            Pub  0.17
    2         Lounge  0.17
    3  Grocery Store  0.17
    4    Pizza Place  0.17
    
    
    ----FLORAL STREET----
                venue  freq
    0         Theater  0.09
    1     Coffee Shop  0.06
    2          Bakery  0.05
    3  Ice Cream Shop  0.05
    4  Clothing Store  0.04
    
    
    ----FRANK DIXON WAY----
                      venue  freq
    0                  Café  0.25
    1                  Lake  0.12
    2  Gym / Fitness Center  0.12
    3          Tennis Court  0.12
    4           Rugby Pitch  0.12
    
    
    ----FULTON MEWS----
                    venue  freq
    0               Hotel  0.18
    1                 Pub  0.08
    2         Coffee Shop  0.07
    3  Chinese Restaurant  0.06
    4                Café  0.04
    
    
    ----GERARD ROAD----
                      venue  freq
    0                  Park  0.18
    1           Sports Club  0.09
    2           Pizza Place  0.09
    3              Gym Pool  0.09
    4  Gym / Fitness Center  0.09
    
    
    ----GERRARD ROAD----
                  venue  freq
    0               Pub  0.09
    1       Coffee Shop  0.08
    2      Burger Joint  0.04
    3  Sushi Restaurant  0.03
    4              Café  0.03
    
    
    ----GIRDLERS ROAD----
                    venue  freq
    0                 Pub  0.12
    1  Italian Restaurant  0.06
    2           Gastropub  0.06
    3      Sandwich Place  0.06
    4         Pizza Place  0.06
    
    
    ----GORDON PLACE----
                               venue  freq
    0             Seafood Restaurant   1.0
    1                            ATM   0.0
    2            Outdoor Event Space   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----GRAFTON SQUARE----
                    venue  freq
    0                 Pub  0.10
    1        Burger Joint  0.05
    2                Café  0.05
    3          Restaurant  0.05
    4  Italian Restaurant  0.04
    
    
    ----GRAHAM TERRACE----
                    venue  freq
    0              Bakery  0.07
    1  Italian Restaurant  0.06
    2          Restaurant  0.05
    3         Coffee Shop  0.05
    4                 Pub  0.03
    
    
    ----HARMAN DRIVE----
                      venue  freq
    0  Gym / Fitness Center  0.50
    1           Bus Station  0.25
    2           Coffee Shop  0.25
    3                   ATM  0.00
    4  Outdoor Supply Store  0.00
    
    
    ----HARRIS STREET----
                               venue  freq
    0              Indian Restaurant   0.8
    1                           Café   0.2
    2            Outdoor Event Space   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----HAVANNAH STREET----
                    venue  freq
    0   Indian Restaurant  0.11
    1  Italian Restaurant  0.11
    2               Hotel  0.08
    3  Turkish Restaurant  0.05
    4  Chinese Restaurant  0.05
    
    
    ----HAZLEWELL ROAD----
                      venue  freq
    0   Japanese Restaurant  0.25
    1  Gym / Fitness Center  0.12
    2          Tennis Court  0.12
    3         Grocery Store  0.12
    4             Gastropub  0.12
    
    
    ----HEREFORD MEWS----
                      venue  freq
    0                   Pub  0.09
    1                  Café  0.06
    2  Gym / Fitness Center  0.05
    3                Garden  0.04
    4           Pizza Place  0.04
    
    
    ----HERONDALE AVENUE----
                     venue  freq
    0    French Restaurant  0.50
    1         Tennis Court  0.25
    2        Grocery Store  0.25
    3                  ATM  0.00
    4  Outdoor Event Space  0.00
    
    
    ----HIGHGATE HIGH STREET----
                   venue  freq
    0                Pub  0.30
    1             Bakery  0.10
    2               Café  0.10
    3           Tea Room  0.05
    4  Indian Restaurant  0.05
    
    
    ----HIGHWOOD HILL----
                       venue  freq
    0     Athletics & Sports   1.0
    1                    ATM   0.0
    2  Performing Arts Venue   0.0
    3            Pastry Shop   0.0
    4                   Park   0.0
    
    
    ----HILLGATE PLACE----
                   venue  freq
    0                Pub  0.15
    1               Park  0.07
    2      Grocery Store  0.04
    3        Pizza Place  0.04
    4  Indian Restaurant  0.04
    
    
    ----HOLLYCROFT AVENUE----
                               venue  freq
    0                           Café   1.0
    1                            ATM   0.0
    2                           Park   0.0
    3  Paper / Office Supplies Store   0.0
    4                         Palace   0.0
    
    
    ----HOLLYWOOD MEWS----
                      venue  freq
    0                   Pub  0.07
    1                Bakery  0.07
    2  Gym / Fitness Center  0.07
    3     French Restaurant  0.07
    4    Italian Restaurant  0.07
    
    
    ----HONEYWELL ROAD----
                         venue  freq
    0                      Pub  0.19
    1                     Café  0.12
    2  Health & Beauty Service  0.06
    3              Pizza Place  0.06
    4                Pet Store  0.06
    
    
    ----HORTENSIA ROAD----
                    venue  freq
    0  Italian Restaurant  0.10
    1      Sandwich Place  0.06
    2                 Pub  0.06
    3                Café  0.05
    4       Grocery Store  0.05
    
    
    ----HOXTON SQUARE----
              venue  freq
    0   Coffee Shop  0.07
    1          Café  0.06
    2           Bar  0.06
    3  Cocktail Bar  0.05
    4         Hotel  0.05
    
    
    ----HUNTER ROAD----
                   venue  freq
    0       Burger Joint   0.2
    1         Skate Park   0.2
    2  Fish & Chips Shop   0.2
    3      Grocery Store   0.2
    4               Café   0.2
    
    
    ----JACKSONS LANE----
                    venue  freq
    0                 Pub  0.35
    1                Café  0.12
    2  Italian Restaurant  0.06
    3       Historic Site  0.06
    4  Seafood Restaurant  0.06
    
    
    ----JOHN STREET----
                     venue  freq
    0   Chinese Restaurant  0.09
    1  Austrian Restaurant  0.09
    2       Farmers Market  0.04
    3       Ice Cream Shop  0.04
    4              Butcher  0.04
    
    
    ----KINNERTON STREET----
                    venue  freq
    0               Hotel  0.11
    1            Boutique  0.09
    2                Café  0.06
    3  Italian Restaurant  0.06
    4           Hotel Bar  0.04
    
    
    ----KNARESBOROUGH PLACE----
                    venue  freq
    0               Hotel  0.29
    1                 Pub  0.07
    2  Italian Restaurant  0.05
    3         Coffee Shop  0.04
    4              Garden  0.04
    
    
    ----KNOX STREET----
             venue  freq
    0  Coffee Shop  0.08
    1         Café  0.04
    2    Gastropub  0.04
    3          Pub  0.04
    4        Hotel  0.04
    
    
    ----LADBROKE GROVE----
                      venue  freq
    0    Italian Restaurant  0.07
    1  Gym / Fitness Center  0.05
    2                  Café  0.05
    3        Breakfast Spot  0.04
    4             Bookstore  0.04
    
    
    ----LANCASTER MEWS----
             venue  freq
    0        Hotel  0.26
    1          Pub  0.10
    2         Café  0.08
    3  Coffee Shop  0.06
    4       Garden  0.05
    
    
    ----LANSDOWNE ROAD----
                   venue  freq
    0              Hotel  0.17
    1               Café  0.13
    2          Gastropub  0.07
    3                Pub  0.07
    4  Fish & Chips Shop  0.03
    
    
    ----LATIMER INDUSTRIAL ESTATE----
                      venue  freq
    0  Gym / Fitness Center  0.33
    1       Automotive Shop  0.33
    2       Warehouse Store  0.33
    3                   ATM  0.00
    4   Outdoor Event Space  0.00
    
    
    ----LAXTON PLACE----
                   venue  freq
    0        Coffee Shop  0.11
    1                Pub  0.08
    2  Indian Restaurant  0.07
    3              Plaza  0.04
    4               Café  0.03
    
    
    ----LINCOLN AVENUE----
                  venue  freq
    0       Pizza Place  0.05
    1  Sushi Restaurant  0.05
    2    Ice Cream Shop  0.05
    3      Dessert Shop  0.05
    4         Juice Bar  0.03
    
    
    ----LINGFIELD ROAD----
                     venue  freq
    0        Grocery Store   0.4
    1                 Park   0.2
    2                  Pub   0.2
    3               Bakery   0.2
    4  Outdoor Event Space   0.0
    
    
    ----LISSON STREET----
                venue  freq
    0     Coffee Shop  0.06
    1             Pub  0.06
    2  Sandwich Place  0.06
    3            Café  0.05
    4      Bagel Shop  0.03
    
    
    ----LIVERPOOL GROVE----
                      venue  freq
    0                  Café  0.14
    1  Fast Food Restaurant  0.07
    2          Dessert Shop  0.04
    3        Sandwich Place  0.04
    4            Bagel Shop  0.04
    
    
    ----LONGWOOD DRIVE----
               venue  freq
    0    Bus Station  0.19
    1       Bus Stop  0.12
    2           Café  0.12
    3  Grocery Store  0.06
    4            Bar  0.06
    
    
    ----LONSDALE SQUARE----
                          venue  freq
    0                       Pub  0.08
    1         French Restaurant  0.06
    2  Mediterranean Restaurant  0.06
    3                    Bakery  0.05
    4                 Gastropub  0.05
    
    
    ----MIDDLESEX PASSAGE----
                      venue  freq
    0    Italian Restaurant  0.08
    1  Gym / Fitness Center  0.05
    2                 Plaza  0.05
    3           Coffee Shop  0.05
    4              Wine Bar  0.05
    
    
    ----MONTPELIER AVENUE----
                venue  freq
    0             Pub   0.2
    1   Deli / Bodega   0.2
    2  Discount Store   0.2
    3           Hotel   0.2
    4     Supermarket   0.2
    
    
    ----MONTPELIER WALK----
                    venue  freq
    0                Café  0.11
    1  Italian Restaurant  0.11
    2               Hotel  0.08
    3            Boutique  0.06
    4         Coffee Shop  0.06
    
    
    ----MULTON ROAD----
                     venue  freq
    0       Breakfast Spot  0.33
    1                  Pub  0.33
    2    Convenience Store  0.33
    3                  ATM  0.00
    4  Outdoor Event Space  0.00
    
    
    ----MUNDEN STREET----
                   venue  freq
    0              Hotel  0.11
    1               Café  0.11
    2                Pub  0.09
    3        Pizza Place  0.07
    4  Indian Restaurant  0.07
    
    
    ----NORTH CIRCULAR ROAD----
                      venue  freq
    0    Chinese Restaurant  0.25
    1        Cosmetics Shop  0.25
    2          Optical Shop  0.25
    3  Fast Food Restaurant  0.25
    4                   ATM  0.00
    
    
    ----NOTTINGHAM STREET----
                   venue  freq
    0     Sandwich Place  0.06
    1  French Restaurant  0.04
    2               Café  0.04
    3        Coffee Shop  0.04
    4  Indian Restaurant  0.03
    
    
    ----OAKLEY STREET----
                    venue  freq
    0                 Pub  0.11
    1                Café  0.11
    2  Italian Restaurant  0.07
    3           Nightclub  0.06
    4              Garden  0.06
    
    
    ----OAKWOOD COURT----
             venue  freq
    0  Golf Course  0.25
    1   Smoke Shop  0.25
    2  Pizza Place  0.25
    3         Food  0.25
    4          ATM  0.00
    
    
    ----OBSERVATORY GARDENS----
                venue  freq
    0            Café  0.05
    1             Pub  0.05
    2       Juice Bar  0.04
    3           Hotel  0.04
    4  Clothing Store  0.04
    
    
    ----OLD COURT PLACE----
                    venue  freq
    0               Hotel  0.09
    1                 Pub  0.04
    2  Italian Restaurant  0.04
    3   French Restaurant  0.04
    4                Café  0.04
    
    
    ----ONSLOW MEWS WEST----
                    venue  freq
    0               Hotel  0.11
    1              Bakery  0.05
    2  Italian Restaurant  0.05
    3        Burger Joint  0.04
    4         Coffee Shop  0.04
    
    
    ----PALACE PLACE----
                               venue  freq
    0              Electronics Store   1.0
    1                            ATM   0.0
    2              Outdoor Sculpture   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----PANTON STREET----
                    venue  freq
    0             Theater  0.09
    1               Hotel  0.05
    2      Ice Cream Shop  0.04
    3           Bookstore  0.03
    4  Seafood Restaurant  0.03
    
    
    ----PARK CRESCENT----
                               venue  freq
    0                           Food   0.5
    1                   Camera Store   0.5
    2            Outdoor Event Space   0.0
    3  Paper / Office Supplies Store   0.0
    4                         Palace   0.0
    
    
    ----PARK LANE----
                      venue  freq
    0    Mexican Restaurant  0.07
    1  Fast Food Restaurant  0.07
    2             Nightclub  0.05
    3           Pizza Place  0.05
    4        Clothing Store  0.05
    
    
    ----PARKE ROAD----
                               venue  freq
    0                            Pub  0.50
    1                           Park  0.25
    2                          River  0.25
    3           Other Great Outdoors  0.00
    4  Paper / Office Supplies Store  0.00
    
    
    ----PARKFIELDS----
                 venue  freq
    0    Historic Site  0.33
    1      Supermarket  0.17
    2            Diner  0.17
    3            Hotel  0.17
    4  Harbor / Marina  0.17
    
    
    ----PARTHENIA ROAD----
                   venue  freq
    0        Coffee Shop  0.14
    1                Pub  0.11
    2               Café  0.09
    3      Grocery Store  0.09
    4  French Restaurant  0.06
    
    
    ----PAVILION ROAD----
                               venue  freq
    0                           Lake  0.33
    1                  Grocery Store  0.33
    2                           Café  0.33
    3                            ATM  0.00
    4  Paper / Office Supplies Store  0.00
    
    
    ----PEMBRIDGE MEWS----
                    venue  freq
    0                 Pub  0.06
    1  Italian Restaurant  0.05
    2          Restaurant  0.04
    3      Clothing Store  0.03
    4       Grocery Store  0.03
    
    
    ----PEMBRIDGE ROAD----
                   venue  freq
    0                Pub  0.16
    1               Park  0.06
    2              Hotel  0.06
    3        Pizza Place  0.04
    4  Indian Restaurant  0.04
    
    
    ----PEMBROKE STUDIOS----
             venue  freq
    0   Restaurant  0.14
    1          Pub  0.10
    2   Sports Bar  0.07
    3  Supermarket  0.07
    4    Nightclub  0.03
    
    
    ----PENCOMBE MEWS----
                    venue  freq
    0  Italian Restaurant  0.09
    1                 Pub  0.07
    2      Clothing Store  0.04
    3         Coffee Shop  0.04
    4              Bakery  0.04
    
    
    ----PETERSHAM PLACE----
                         venue  freq
    0                      Pub  0.50
    1               Sports Bar  0.25
    2  Health & Beauty Service  0.25
    3                      ATM  0.00
    4      Outdoor Event Space  0.00
    
    
    ----PHILLIMORE GARDENS----
                 venue  freq
    0             Café  0.17
    1  Harbor / Marina  0.05
    2   Ice Cream Shop  0.05
    3            Hotel  0.05
    4        Bookstore  0.04
    
    
    ----PHYSIC PLACE----
             venue  freq
    0         Café  0.07
    1          Pub  0.07
    2       Bakery  0.05
    3  Pizza Place  0.05
    4  Coffee Shop  0.05
    
    
    ----PITFIELD STREET----
                       venue  freq
    0            Coffee Shop  0.10
    1                    Pub  0.07
    2  Vietnamese Restaurant  0.06
    3                   Café  0.05
    4                    Bar  0.05
    
    
    ----PRINCES GATE----
                               venue  freq
    0                      Gift Shop   1.0
    1                            ATM   0.0
    2  Paper / Office Supplies Store   0.0
    3                         Palace   0.0
    4           Pakistani Restaurant   0.0
    
    
    ----PRIORY ROAD----
                    venue  freq
    0                 Pub  0.29
    1          Restaurant  0.12
    2         Pizza Place  0.04
    3  Seafood Restaurant  0.04
    4           Rock Club  0.02
    
    
    ----PROTHERO GARDENS----
                      venue  freq
    0         Grocery Store  0.15
    1           Coffee Shop  0.10
    2   Japanese Restaurant  0.05
    3             BBQ Joint  0.05
    4  Gym / Fitness Center  0.05
    
    
    ----PUTNEY HIGH STREET----
                venue  freq
    0     Coffee Shop  0.09
    1            Café  0.05
    2  Clothing Store  0.05
    3  Sandwich Place  0.04
    4             Bar  0.04
    
    
    ----QUARRENDON STREET----
                    venue  freq
    0         Coffee Shop  0.12
    1  Italian Restaurant  0.10
    2                 Pub  0.10
    3                Café  0.07
    4       Grocery Store  0.07
    
    
    ----QUEENS GATE TERRACE----
                      venue  freq
    0  Fast Food Restaurant   0.2
    1         Shopping Mall   0.1
    2                  Food   0.1
    3    Mexican Restaurant   0.1
    4     Convenience Store   0.1
    
    
    ----RADSTOCK STREET----
                     venue  freq
    0                  Pub  0.29
    1        Grocery Store  0.14
    2  Japanese Restaurant  0.07
    3         Cocktail Bar  0.07
    4    French Restaurant  0.07
    
    
    ----RANELAGH AVENUE----
                    venue  freq
    0                 Pub  0.16
    1  Italian Restaurant  0.12
    2   Convenience Store  0.08
    3                Café  0.08
    4              Garden  0.04
    
    
    ----REDCLIFFE ROAD----
                      venue  freq
    0                   Pub  0.29
    1    Chinese Restaurant  0.14
    2  Gym / Fitness Center  0.14
    3         Grocery Store  0.14
    4        Sandwich Place  0.14
    
    
    ----REEVES MEWS----
                    venue  freq
    0               Hotel  0.12
    1  Italian Restaurant  0.05
    2   French Restaurant  0.05
    3                Café  0.04
    4          Restaurant  0.04
    
    
    ----RHEIDOL MEWS----
                   venue  freq
    0                Pub  0.13
    1               Park  0.07
    2        Coffee Shop  0.07
    3               Café  0.03
    4  French Restaurant  0.03
    
    
    ----RINGWOOD AVENUE----
                          venue  freq
    0                       Pub  0.14
    1         Indian Restaurant  0.14
    2      Gym / Fitness Center  0.14
    3                      Café  0.14
    4  Mediterranean Restaurant  0.14
    
    
    ----RODERICK ROAD----
                      venue  freq
    0      Asian Restaurant  0.14
    1     Indian Restaurant  0.14
    2            Restaurant  0.14
    3  Pakistani Restaurant  0.14
    4     Convenience Store  0.14
    
    
    ----ROPEMAKERS FIELDS----
                      venue  freq
    0     Indian Restaurant  0.12
    1  Gym / Fitness Center  0.12
    2    Italian Restaurant  0.08
    3           Pizza Place  0.08
    4              Bus Stop  0.08
    
    
    ----ROYAL CRESCENT----
                    venue  freq
    0               Hotel  0.12
    1  Italian Restaurant  0.08
    2                 Pub  0.08
    3                Café  0.08
    4      History Museum  0.08
    
    
    ----RUSSELL GARDENS MEWS----
                    venue  freq
    0               Hotel  0.16
    1           Gastropub  0.08
    2  Italian Restaurant  0.08
    3  Persian Restaurant  0.08
    4                 Pub  0.08
    
    
    ----SETTLES STREET----
                   venue  freq
    0              Hotel  0.14
    1        Coffee Shop  0.09
    2               Café  0.07
    3                Pub  0.06
    4  Indian Restaurant  0.06
    
    
    ----SOUTH END ROW----
                    venue  freq
    0               Hotel  0.09
    1  Italian Restaurant  0.06
    2          Restaurant  0.06
    3      Clothing Store  0.06
    4          Hookah Bar  0.04
    
    
    ----SOUTHWOOD LAWN ROAD----
                   venue  freq
    0                Pub  0.29
    1               Café  0.12
    2           Bus Stop  0.06
    3             Bakery  0.06
    4  Indian Restaurant  0.06
    
    
    ----SOVEREIGN PARK----
                   venue  freq
    0             Bakery   0.5
    1  Convenience Store   0.5
    2                ATM   0.0
    3  Outdoor Sculpture   0.0
    4        Pastry Shop   0.0
    
    
    ----ST OSWALDS PLACE----
                    venue  freq
    0                Café  0.12
    1                 Pub  0.10
    2      Cricket Ground  0.08
    3             Gay Bar  0.08
    4  Italian Restaurant  0.05
    
    
    ----ST PETERS SQUARE----
                    venue  freq
    0  Italian Restaurant  0.22
    1      Ice Cream Shop  0.09
    2               Hotel  0.09
    3                Café  0.08
    4              Museum  0.05
    
    
    ----STAFFORD TERRACE----
                venue  freq
    0            Café  0.10
    1          Bakery  0.05
    2  Clothing Store  0.05
    3      Restaurant  0.03
    4   Grocery Store  0.03
    
    
    ----SUTHERLAND PLACE----
                               venue  freq
    0                            Pub  0.10
    1                           Café  0.06
    2             Italian Restaurant  0.05
    3  Vegetarian / Vegan Restaurant  0.03
    4             Persian Restaurant  0.03
    
    
    ----SYDNEY STREET----
                 venue  freq
    0            Beach  0.25
    1    Grocery Store  0.25
    2             Café  0.25
    3  Harbor / Marina  0.25
    4              ATM  0.00
    
    
    ----THAMES BANK----
                      venue  freq
    0  Gym / Fitness Center  0.25
    1          Burger Joint  0.25
    2         Grocery Store  0.25
    3           Pizza Place  0.25
    4                   ATM  0.00
    
    
    ----THE HEXAGON----
                  venue  freq
    0    Clothing Store  0.09
    1  Asian Restaurant  0.07
    2               Pub  0.07
    3       Coffee Shop  0.07
    4               Bar  0.05
    
    
    ----TREDEGAR SQUARE----
              venue  freq
    0           Pub  0.16
    1   Pizza Place  0.16
    2      Bus Stop  0.10
    3   Coffee Shop  0.10
    4  Burger Joint  0.06
    
    
    ----TRINITY STREET----
                venue  freq
    0             Pub  0.33
    1           Hotel  0.17
    2       Gastropub  0.08
    3  Sandwich Place  0.08
    4     Coffee Shop  0.08
    
    
    ----UPPER HAMPSTEAD WALK----
                venue  freq
    0            Café  0.10
    1             Pub  0.10
    2          Bakery  0.06
    3  Ice Cream Shop  0.05
    4  Clothing Store  0.05
    
    
    ----WALPOLE GARDENS----
                   venue  freq
    0                Pub  0.22
    1    Thai Restaurant  0.22
    2               Park  0.11
    3           Bus Stop  0.11
    4  Convenience Store  0.11
    
    
    ----WALPOLE STREET----
                  venue  freq
    0               Pub  0.17
    1             Hotel  0.11
    2  Asian Restaurant  0.06
    3      Burger Joint  0.06
    4              Café  0.06
    
    
    ----WARWICK SQUARE----
                      venue  freq
    0        Sandwich Place  0.50
    1       Automotive Shop  0.25
    2           Coffee Shop  0.25
    3                   ATM  0.00
    4  Outdoor Supply Store  0.00
    
    
    ----WELBECK WAY----
                      venue  freq
    0    Chinese Restaurant   0.2
    1  Fast Food Restaurant   0.2
    2            Playground   0.2
    3       Warehouse Store   0.2
    4     Fish & Chips Shop   0.2
    
    
    ----WELLESLEY TERRACE----
               venue  freq
    0            Pub  0.12
    1  Grocery Store  0.08
    2          Hotel  0.08
    3    Coffee Shop  0.06
    4           Park  0.06
    
    
    ----WELLINGTON STREET----
              venue  freq
    0   Coffee Shop  0.19
    1         Hotel  0.08
    2  Concert Hall  0.04
    3    Food Truck  0.04
    4          Café  0.04
    
    
    ----WESTMORELAND PLACE----
                venue  freq
    0     IT Services  0.06
    1     Video Store  0.06
    2      Bagel Shop  0.06
    3   Grocery Store  0.06
    4  Breakfast Spot  0.06
    
    
    ----WHITFIELD STREET----
                      venue  freq
    0           Coffee Shop  0.14
    1                  Café  0.05
    2                Bakery  0.04
    3    Italian Restaurant  0.04
    4  Gym / Fitness Center  0.03
    
    
    ----WILFRED STREET----
                     venue  freq
    0          Coffee Shop  0.11
    1                Hotel  0.09
    2       Sandwich Place  0.06
    3              Theater  0.05
    4  Sporting Goods Shop  0.02
    
    
    ----WILLOW BRIDGE ROAD----
                     venue  freq
    0                  Pub  0.23
    1          Coffee Shop  0.09
    2                 Park  0.09
    3                 Café  0.09
    4  Japanese Restaurant  0.05
    
    
    ----WILSON STREET----
                               venue  freq
    0                 Sandwich Place   1.0
    1                            ATM   0.0
    2               Pedestrian Plaza   0.0
    3                           Park   0.0
    4  Paper / Office Supplies Store   0.0
    
    
    ----WINCHENDON ROAD----
                    venue  freq
    0                 Pub  0.14
    1         Pizza Place  0.14
    2                Café  0.07
    3   Indian Restaurant  0.07
    4  Chinese Restaurant  0.07
    
    
    ----WINGATE ROAD----
                      venue  freq
    0           Supermarket  0.25
    1             Gastropub  0.25
    2     Electronics Store  0.25
    3       Warehouse Store  0.25
    4  Other Great Outdoors  0.00
    
    



```python
# Define a function to return the most common venues/facilities nearby real estate investments#

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Street']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))
```


```python
# create a new dataframe
venues_sorted = pd.DataFrame(columns=columns)
venues_sorted['Street'] = london_grouped['Street']

for ind in np.arange(london_grouped.shape[0]):
    venues_sorted.iloc[ind, 1:] = return_most_common_venues(london_grouped.iloc[ind, :], num_top_venues)
```


```python
venues_sorted.head()
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
      <th>Street</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALBION SQUARE</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Indian Restaurant</td>
      <td>Beer Garden</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Museum</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ANHALT ROAD</td>
      <td>Pub</td>
      <td>Grocery Store</td>
      <td>Diner</td>
      <td>French Restaurant</td>
      <td>Garden</td>
      <td>English Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Gym / Fitness Center</td>
      <td>Plaza</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ANSDELL TERRACE</td>
      <td>Hotel</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Bakery</td>
      <td>English Restaurant</td>
      <td>French Restaurant</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>APPLEGARTH ROAD</td>
      <td>Pub</td>
      <td>Casino</td>
      <td>Nightclub</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BARONSMEAD ROAD</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Nature Preserve</td>
      <td>Bookstore</td>
      <td>Café</td>
      <td>Farmers Market</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>




```python
venues_sorted.shape
london_grouped.shape
london_grouped=df1
```

After our inspection of venues/facilities/amenities nearby the most profitable real estate investments in London, we could begin by clustering properties by venues/facilities/amenities nearby.


```python

# set number of clusters
kclusters = 5

london_grouped_clustering = london_grouped.drop('Street', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(london_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:50]
```




    array([1, 3, 0, 3, 2, 1, 2, 0, 0, 1, 3, 3, 3, 1, 2, 2, 1, 3, 0, 1, 4, 4,
           3, 1, 1, 0, 3, 4, 1, 0, 3, 2, 3, 2, 2, 4, 3, 3, 2, 0, 1, 2, 4, 0,
           4, 0, 0, 4, 0, 0], dtype=int32)




```python
#Dataframe to include Clusters

london_grouped_clustering=df1
london_grouped_clustering.head()
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
      <th>Street</th>
      <th>Avg_Price</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2450000.0</td>
      <td>-41.273758</td>
      <td>173.289393</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2435000.0</td>
      <td>51.480316</td>
      <td>-0.166801</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2250000.0</td>
      <td>51.499890</td>
      <td>-0.189103</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2400000.0</td>
      <td>53.748654</td>
      <td>-0.326670</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2375000.0</td>
      <td>51.477315</td>
      <td>-0.239457</td>
    </tr>
  </tbody>
</table>
</div>




```python
london_grouped_clustering.shape
```




    (162, 4)




```python
df1.shape
```




    (162, 4)




```python
london_grouped_clustering.dtypes
```




    Street        object
    Avg_Price    float64
    Latitude     float64
    Longitude    float64
    dtype: object




```python
df1.dtypes
```




    Street        object
    Avg_Price    float64
    Latitude     float64
    Longitude    float64
    dtype: object




```python
# add clustering labels
london_grouped_clustering['Cluster Labels'] = kmeans.labels_

# merge london_grouped with london_data to add latitude/longitude for each neighborhood
london_grouped_clustering = london_grouped_clustering.join(venues_sorted.set_index('Street'), on='Street')

london_grouped_clustering.head(30) # check the last columns!
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
      <th>Street</th>
      <th>Avg_Price</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>ALBION SQUARE</td>
      <td>2.450000e+06</td>
      <td>-41.273758</td>
      <td>173.289393</td>
      <td>1</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Indian Restaurant</td>
      <td>Beer Garden</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Museum</td>
    </tr>
    <tr>
      <th>391</th>
      <td>ANHALT ROAD</td>
      <td>2.435000e+06</td>
      <td>51.480316</td>
      <td>-0.166801</td>
      <td>3</td>
      <td>Pub</td>
      <td>Grocery Store</td>
      <td>Diner</td>
      <td>French Restaurant</td>
      <td>Garden</td>
      <td>English Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Gym / Fitness Center</td>
      <td>Plaza</td>
    </tr>
    <tr>
      <th>406</th>
      <td>ANSDELL TERRACE</td>
      <td>2.250000e+06</td>
      <td>51.499890</td>
      <td>-0.189103</td>
      <td>0</td>
      <td>Hotel</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Bakery</td>
      <td>English Restaurant</td>
      <td>French Restaurant</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>422</th>
      <td>APPLEGARTH ROAD</td>
      <td>2.400000e+06</td>
      <td>53.748654</td>
      <td>-0.326670</td>
      <td>3</td>
      <td>Pub</td>
      <td>Casino</td>
      <td>Nightclub</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>855</th>
      <td>BARONSMEAD ROAD</td>
      <td>2.375000e+06</td>
      <td>51.477315</td>
      <td>-0.239457</td>
      <td>2</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Nature Preserve</td>
      <td>Bookstore</td>
      <td>Café</td>
      <td>Farmers Market</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>981</th>
      <td>BEAUCLERC ROAD</td>
      <td>2.480000e+06</td>
      <td>51.499577</td>
      <td>-0.229033</td>
      <td>1</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Grocery Store</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Latin American Restaurant</td>
      <td>Gastropub</td>
      <td>French Restaurant</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>BELVEDERE DRIVE</td>
      <td>2.340000e+06</td>
      <td>52.414209</td>
      <td>1.724415</td>
      <td>2</td>
      <td>Pub</td>
      <td>Seafood Restaurant</td>
      <td>Campground</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Vacation Rental</td>
      <td>Fast Food Restaurant</td>
      <td>Exhibit</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>BICKENHALL STREET</td>
      <td>2.208500e+06</td>
      <td>51.521201</td>
      <td>-0.158908</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Chinese Restaurant</td>
      <td>Grocery Store</td>
      <td>Gastropub</td>
      <td>Indian Restaurant</td>
      <td>Movie Theater</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>BIRCHLANDS AVENUE</td>
      <td>2.217000e+06</td>
      <td>51.448394</td>
      <td>-0.160468</td>
      <td>0</td>
      <td>Pub</td>
      <td>Breakfast Spot</td>
      <td>Coffee Shop</td>
      <td>Brewery</td>
      <td>Chinese Restaurant</td>
      <td>Train Station</td>
      <td>Lake</td>
      <td>Bakery</td>
      <td>French Restaurant</td>
      <td>Flower Shop</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>BRAMPTON GROVE</td>
      <td>2.456875e+06</td>
      <td>51.589961</td>
      <td>-0.318525</td>
      <td>1</td>
      <td>Food Service</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>BRIARDALE GARDENS</td>
      <td>2.397132e+06</td>
      <td>51.560175</td>
      <td>-0.195431</td>
      <td>3</td>
      <td>Grocery Store</td>
      <td>Other Great Outdoors</td>
      <td>Indian Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Convenience Store</td>
      <td>Coffee Shop</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>BROOKWAY</td>
      <td>2.400000e+06</td>
      <td>45.432185</td>
      <td>-122.802812</td>
      <td>3</td>
      <td>Art Gallery</td>
      <td>Asian Restaurant</td>
      <td>Zoo</td>
      <td>Flea Market</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>BURBAGE ROAD</td>
      <td>2.445000e+06</td>
      <td>51.448260</td>
      <td>-0.088507</td>
      <td>3</td>
      <td>Gastropub</td>
      <td>Athletics &amp; Sports</td>
      <td>Pub</td>
      <td>Greek Restaurant</td>
      <td>Food &amp; Drink Shop</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Cricket Ground</td>
      <td>Art Gallery</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>BURY WALK</td>
      <td>2.492500e+06</td>
      <td>52.145529</td>
      <td>-0.423593</td>
      <td>1</td>
      <td>Supermarket</td>
      <td>English Restaurant</td>
      <td>Café</td>
      <td>Gym</td>
      <td>Dry Cleaner</td>
      <td>Hardware Store</td>
      <td>Fast Food Restaurant</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Rental Car Location</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>CALLCOTT STREET</td>
      <td>2.375000e+06</td>
      <td>51.508350</td>
      <td>-0.198328</td>
      <td>2</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Indian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Greek Restaurant</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>CAMPDEN HILL ROAD</td>
      <td>2.379653e+06</td>
      <td>51.501410</td>
      <td>-0.195116</td>
      <td>2</td>
      <td>Café</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Bakery</td>
      <td>French Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Juice Bar</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>CAMPION ROAD</td>
      <td>2.461000e+06</td>
      <td>52.681375</td>
      <td>0.965471</td>
      <td>1</td>
      <td>Soccer Field</td>
      <td>Windmill</td>
      <td>Spa</td>
      <td>Zoo</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>2158</th>
      <td>CANNING PLACE</td>
      <td>2.425000e+06</td>
      <td>51.499570</td>
      <td>-0.184248</td>
      <td>3</td>
      <td>Hotel</td>
      <td>Pub</td>
      <td>French Restaurant</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Malay Restaurant</td>
      <td>Mediterranean Restaurant</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>CARLISLE ROAD</td>
      <td>2.200000e+06</td>
      <td>-36.709171</td>
      <td>174.728281</td>
      <td>0</td>
      <td>Indian Restaurant</td>
      <td>Pharmacy</td>
      <td>Café</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Liquor Store</td>
      <td>Sandwich Place</td>
      <td>Bakery</td>
      <td>Convenience Store</td>
      <td>Pizza Place</td>
      <td>Design Studio</td>
    </tr>
    <tr>
      <th>2230</th>
      <td>CARLTON GARDENS</td>
      <td>2.483500e+06</td>
      <td>-37.801943</td>
      <td>144.971970</td>
      <td>1</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Dessert Shop</td>
      <td>Convenience Store</td>
      <td>Deli / Bodega</td>
      <td>Light Rail Station</td>
      <td>Lebanese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Hotel</td>
    </tr>
    <tr>
      <th>2242</th>
      <td>CARLYLE COURT</td>
      <td>2.300000e+06</td>
      <td>32.972701</td>
      <td>-97.173392</td>
      <td>4</td>
      <td>Farm</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>CHALCOT SQUARE</td>
      <td>2.286679e+06</td>
      <td>51.541196</td>
      <td>-0.155817</td>
      <td>4</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>French Restaurant</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Belgian Restaurant</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>CHARLES LANE</td>
      <td>2.414000e+06</td>
      <td>51.533837</td>
      <td>-0.170298</td>
      <td>3</td>
      <td>Cricket Ground</td>
      <td>Café</td>
      <td>Deli / Bodega</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>French Restaurant</td>
      <td>Lebanese Restaurant</td>
      <td>Gastropub</td>
      <td>Garden</td>
      <td>Modern European Restaurant</td>
    </tr>
    <tr>
      <th>2561</th>
      <td>CHELSEA CRESCENT</td>
      <td>2.495000e+06</td>
      <td>34.522443</td>
      <td>-85.443891</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>CHESTER CLOSE NORTH</td>
      <td>2.450000e+06</td>
      <td>51.529205</td>
      <td>-0.145081</td>
      <td>1</td>
      <td>Garden</td>
      <td>Cocktail Bar</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Gym / Fitness Center</td>
      <td>Performing Arts Venue</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Buffet</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>CHEYNE COURT</td>
      <td>2.250000e+06</td>
      <td>51.599677</td>
      <td>0.525623</td>
      <td>0</td>
      <td>Construction &amp; Landscaping</td>
      <td>Gastropub</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>2640</th>
      <td>CHEYNE ROW</td>
      <td>2.410000e+06</td>
      <td>51.483717</td>
      <td>-0.169603</td>
      <td>3</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Italian Restaurant</td>
      <td>Pizza Place</td>
      <td>Bakery</td>
      <td>Gym / Fitness Center</td>
      <td>Nightclub</td>
      <td>Plaza</td>
      <td>Burger Joint</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>CHISWICK MALL</td>
      <td>2.287500e+06</td>
      <td>51.487185</td>
      <td>-0.248017</td>
      <td>4</td>
      <td>Pub</td>
      <td>Art Museum</td>
      <td>Reservoir</td>
      <td>Gift Shop</td>
      <td>Brewery</td>
      <td>Hunan Restaurant</td>
      <td>English Restaurant</td>
      <td>Event Space</td>
      <td>Exhibit</td>
      <td>Factory</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>CITY ROAD</td>
      <td>2.468340e+06</td>
      <td>51.529697</td>
      <td>-0.097763</td>
      <td>1</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>Art Gallery</td>
      <td>Park</td>
      <td>Dance Studio</td>
      <td>Spa</td>
      <td>School</td>
      <td>Sandwich Place</td>
      <td>Gay Bar</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>CLARENDON STREET</td>
      <td>2.250000e+06</td>
      <td>51.365160</td>
      <td>1.108569</td>
      <td>0</td>
      <td>Indian Restaurant</td>
      <td>Grocery Store</td>
      <td>Pet Store</td>
      <td>Fast Food Restaurant</td>
      <td>Zoo</td>
      <td>Film Studio</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create Map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(london_grouped_clustering['Latitude'], london_grouped_clustering['Longitude'], london_grouped_clustering['Street'], london_grouped_clustering['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2YyA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2YycsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNTEuNTA3MzIxOSwtMC4xMjc2NDc0XSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHpvb206IDExLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbWF4Qm91bmRzOiBib3VuZHMsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBsYXllcnM6IFtdLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgd29ybGRDb3B5SnVtcDogZmFsc2UsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjcnM6IEwuQ1JTLkVQU0czODU3CiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIH0pOwogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgdGlsZV9sYXllcl82MGFhOTc4ZWIxY2U0MzkwOTMyOTE4NzI3MzdiNDIzYSA9IEwudGlsZUxheWVyKAogICAgICAgICAgICAgICAgJ2h0dHBzOi8ve3N9LnRpbGUub3BlbnN0cmVldG1hcC5vcmcve3p9L3t4fS97eX0ucG5nJywKICAgICAgICAgICAgICAgIHsKICAiYXR0cmlidXRpb24iOiBudWxsLAogICJkZXRlY3RSZXRpbmEiOiBmYWxzZSwKICAibWF4Wm9vbSI6IDE4LAogICJtaW5ab29tIjogMSwKICAibm9XcmFwIjogZmFsc2UsCiAgInN1YmRvbWFpbnMiOiAiYWJjIgp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzk0MTM1NWIyMDBjNGVlZDk1YTcxZTE1NWU0MDY5YzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstNDEuMjczNzU3NTUsMTczLjI4OTM5MzIzOTEwMzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RiY2ZhYzhlYjkwODQ3NDRhMzAwNjQyYmU1Y2Y4NGIyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhYTQ3NzhiNGNjYjRlMmZiMThmNmJlODQ0ODQxYzhlID0gJCgnPGRpdiBpZD0iaHRtbF9hYWE0Nzc4YjRjY2I0ZTJmYjE4ZjZiZTg0NDg0MWM4ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QUxCSU9OIFNRVUFSRSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RiY2ZhYzhlYjkwODQ3NDRhMzAwNjQyYmU1Y2Y4NGIyLnNldENvbnRlbnQoaHRtbF9hYWE0Nzc4YjRjY2I0ZTJmYjE4ZjZiZTg0NDg0MWM4ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OTQxMzU1YjIwMGM0ZWVkOTVhNzFlMTU1ZTQwNjljMS5iaW5kUG9wdXAocG9wdXBfZGJjZmFjOGViOTA4NDc0NGEzMDA2NDJiZTVjZjg0YjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmUyNjc5YTkwZTFhNDMxOGFkZjY4ZmZhYjUxOTg3OGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40ODAzMTY0LC0wLjE2NjgwMTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjU2MTZmNjAyOTU2NGNiMThmZTgwOWM3MDNhNjdmNmIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGU1Nzc4NTk5NmMxNDE5ZTgxYzViZGU4NDNjODFlMDYgPSAkKCc8ZGl2IGlkPSJodG1sXzBlNTc3ODU5OTZjMTQxOWU4MWM1YmRlODQzYzgxZTA2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BTkhBTFQgUk9BRCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y1NjE2ZjYwMjk1NjRjYjE4ZmU4MDljNzAzYTY3ZjZiLnNldENvbnRlbnQoaHRtbF8wZTU3Nzg1OTk2YzE0MTllODFjNWJkZTg0M2M4MWUwNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZTI2NzlhOTBlMWE0MzE4YWRmNjhmZmFiNTE5ODc4ZS5iaW5kUG9wdXAocG9wdXBfZjU2MTZmNjAyOTU2NGNiMThmZTgwOWM3MDNhNjdmNmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTgxNjlhZTU3MmFlNGJlZjllOGQ3MDIzYzkxZTQ2YzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTk4ODk5LC0wLjE4OTEwMjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDE1MmQ2M2IyZWY5NGEwMzlmYWQ2YjRiNzY4ZmRhMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGQ3M2ZmMjUxZmVhNDA5ZGEwYzhlOWZiNDQ4ZDQ2NmUgPSAkKCc8ZGl2IGlkPSJodG1sXzhkNzNmZjI1MWZlYTQwOWRhMGM4ZTlmYjQ0OGQ0NjZlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BTlNERUxMIFRFUlJBQ0UgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80MTUyZDYzYjJlZjk0YTAzOWZhZDZiNGI3NjhmZGEwMy5zZXRDb250ZW50KGh0bWxfOGQ3M2ZmMjUxZmVhNDA5ZGEwYzhlOWZiNDQ4ZDQ2NmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTgxNjlhZTU3MmFlNGJlZjllOGQ3MDIzYzkxZTQ2YzAuYmluZFBvcHVwKHBvcHVwXzQxNTJkNjNiMmVmOTRhMDM5ZmFkNmI0Yjc2OGZkYTAzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdlM2JkYzNkMDJjZDRmMGI4YWQ4NjFjZGQwZmQ5ZTVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuNzQ4NjUzOSwtMC4zMjY2NzA0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZhZjI0Y2UwNzFjODQ3N2M5NjAxZmE1NjJlMjEwYTQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhkNmZlODc0MmY1NzQyZTFiNmY3NWY4NGZmZjg2NDg2ID0gJCgnPGRpdiBpZD0iaHRtbF84ZDZmZTg3NDJmNTc0MmUxYjZmNzVmODRmZmY4NjQ4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QVBQTEVHQVJUSCBST0FEIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmFmMjRjZTA3MWM4NDc3Yzk2MDFmYTU2MmUyMTBhNDQuc2V0Q29udGVudChodG1sXzhkNmZlODc0MmY1NzQyZTFiNmY3NWY4NGZmZjg2NDg2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdlM2JkYzNkMDJjZDRmMGI4YWQ4NjFjZGQwZmQ5ZTVlLmJpbmRQb3B1cChwb3B1cF9mYWYyNGNlMDcxYzg0NzdjOTYwMWZhNTYyZTIxMGE0NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMDdhNzFmZmQ0YmQ0MGI4YWE4NGQ0ZjA3ODdkMDJmYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ3NzMxNDcsLTAuMjM5NDU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFkM2VmYmY0Y2JkNDQ0ODQ4MDFlOTAxMWE3ODFiOWQ1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JkNTU3N2Q3ZDRlMDRiMTY5YzlmZTE0ZjdmMWQwNWVhID0gJCgnPGRpdiBpZD0iaHRtbF9iZDU1NzdkN2Q0ZTA0YjE2OWM5ZmUxNGY3ZjFkMDVlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QkFST05TTUVBRCBST0FEIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWQzZWZiZjRjYmQ0NDQ4NDgwMWU5MDExYTc4MWI5ZDUuc2V0Q29udGVudChodG1sX2JkNTU3N2Q3ZDRlMDRiMTY5YzlmZTE0ZjdmMWQwNWVhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YwN2E3MWZmZDRiZDQwYjhhYTg0ZDRmMDc4N2QwMmZiLmJpbmRQb3B1cChwb3B1cF8xZDNlZmJmNGNiZDQ0NDg0ODAxZTkwMTFhNzgxYjlkNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZDNlZmZjZmNhYTU0MjFmODc4MDVkMWQ5OTA0ZjRhYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5OTU3NzEsLTAuMjI5MDMzMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMWIzMThmMTI4ZjY0YWY2ODZkMjk0MDM4M2JiM2M3MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yOGY2Y2I4ZDA2YTY0ZjA2OTJkNzdiNzhjNGRkYTU5MyA9ICQoJzxkaXYgaWQ9Imh0bWxfMjhmNmNiOGQwNmE2NGYwNjkyZDc3Yjc4YzRkZGE1OTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJFQVVDTEVSQyBST0FEIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjFiMzE4ZjEyOGY2NGFmNjg2ZDI5NDAzODNiYjNjNzEuc2V0Q29udGVudChodG1sXzI4ZjZjYjhkMDZhNjRmMDY5MmQ3N2I3OGM0ZGRhNTkzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZkM2VmZmNmY2FhNTQyMWY4NzgwNWQxZDk5MDRmNGFjLmJpbmRQb3B1cChwb3B1cF9mMWIzMThmMTI4ZjY0YWY2ODZkMjk0MDM4M2JiM2M3MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MTRjY2UzN2I4Zjk0NTE1OTAwZDM2NWZhMDBmOWJiZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjQxNDIwODksMS43MjQ0MTUyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMzM2NiYWRhMTIzNjQ2NjQ4YjIzOTZmM2Y1MGFjMjRmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I3MGQ2ZjNmNTU5MDQ2MzQ4M2Q3OTljZGE5MDRhNzViID0gJCgnPGRpdiBpZD0iaHRtbF9iNzBkNmYzZjU1OTA0NjM0ODNkNzk5Y2RhOTA0YTc1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QkVMVkVERVJFIERSSVZFIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzMzY2JhZGExMjM2NDY2NDhiMjM5NmYzZjUwYWMyNGYuc2V0Q29udGVudChodG1sX2I3MGQ2ZjNmNTU5MDQ2MzQ4M2Q3OTljZGE5MDRhNzViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkxNGNjZTM3YjhmOTQ1MTU5MDBkMzY1ZmEwMGY5YmJlLmJpbmRQb3B1cChwb3B1cF8zMzNjYmFkYTEyMzY0NjY0OGIyMzk2ZjNmNTBhYzI0Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NzE0MDVjZjk5ZjI0ZTc1YmNjY2EzMmEzZmYzN2JjZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUyMTIwMTQsLTAuMTU4OTA4Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNWM4NDI2MmQ0MmI0ODhkOGI4MzNhOTdjNGI4MGE3YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMGZiM2QyNjdkYjk0ZTU2OGQ5ZGIyNjM1ZjVmOTUzMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTBmYjNkMjY3ZGI5NGU1NjhkOWRiMjYzNWY1Zjk1MzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJJQ0tFTkhBTEwgU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjVjODQyNjJkNDJiNDg4ZDhiODMzYTk3YzRiODBhN2Iuc2V0Q29udGVudChodG1sX2EwZmIzZDI2N2RiOTRlNTY4ZDlkYjI2MzVmNWY5NTMxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY3MTQwNWNmOTlmMjRlNzViY2NjYTMyYTNmZjM3YmNmLmJpbmRQb3B1cChwb3B1cF8yNWM4NDI2MmQ0MmI0ODhkOGI4MzNhOTdjNGI4MGE3Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNzA5NzMzMmE2Zjg0YmZlYmZjMDNjMmVmNTc4NzRmYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0ODM5NDEsLTAuMTYwNDY3Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYzJhMTE0YjM3YTk0YzhlYTVlZWQ5NzlkMDMyMmI4YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MmYwZTFmMDliMDQ0NmQyYmRlNDNjYzU3ZDJmOWE1ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDJmMGUxZjA5YjA0NDZkMmJkZTQzY2M1N2QyZjlhNWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJJUkNITEFORFMgQVZFTlVFIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGMyYTExNGIzN2E5NGM4ZWE1ZWVkOTc5ZDAzMjJiOGEuc2V0Q29udGVudChodG1sXzQyZjBlMWYwOWIwNDQ2ZDJiZGU0M2NjNTdkMmY5YTVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y3MDk3MzMyYTZmODRiZmViZmMwM2MyZWY1Nzg3NGZjLmJpbmRQb3B1cChwb3B1cF9kYzJhMTE0YjM3YTk0YzhlYTVlZWQ5NzlkMDMyMmI4YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lOTAxMjQ1MWJkMjk0MzA3YmQwZDI1ZWZiNTk5YmJiMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU4OTk2MDcsLTAuMzE4NTI0OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83OTM0ZmVjZDA2ZmM0MDZlOGE4ZGJlZTZkMzg4YzhhNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMjg3MGU5OTE5YWE0NjdjOWY4YmU5OGYzMzI1Nzg0MSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTI4NzBlOTkxOWFhNDY3YzlmOGJlOThmMzMyNTc4NDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJSQU1QVE9OIEdST1ZFIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzkzNGZlY2QwNmZjNDA2ZThhOGRiZWU2ZDM4OGM4YTUuc2V0Q29udGVudChodG1sX2EyODcwZTk5MTlhYTQ2N2M5ZjhiZTk4ZjMzMjU3ODQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U5MDEyNDUxYmQyOTQzMDdiZDBkMjVlZmI1OTliYmIxLmJpbmRQb3B1cChwb3B1cF83OTM0ZmVjZDA2ZmM0MDZlOGE4ZGJlZTZkMzg4YzhhNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83Y2Q3YzBmN2UwNmE0ZDMxYTU2MTZjZGM3YzIwZWMxYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU2MDE3NDgsLTAuMTk1NDMwNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZGJjNDk5YWY4Nzk0MTY0YTIzZGRjMmFlMzFkODhkNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NTAyMDcwYzg3ZmM0ZjVjOTFhOGYwZWM2ZjEwNmRlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzUwMjA3MGM4N2ZjNGY1YzkxYThmMGVjNmYxMDZkZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJSSUFSREFMRSBHQVJERU5TIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmRiYzQ5OWFmODc5NDE2NGEyM2RkYzJhZTMxZDg4ZDUuc2V0Q29udGVudChodG1sXzc1MDIwNzBjODdmYzRmNWM5MWE4ZjBlYzZmMTA2ZGVhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdjZDdjMGY3ZTA2YTRkMzFhNTYxNmNkYzdjMjBlYzFjLmJpbmRQb3B1cChwb3B1cF82ZGJjNDk5YWY4Nzk0MTY0YTIzZGRjMmFlMzFkODhkNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZDgxMzkyNzVhY2Y0YjYyYTczODk3OTU1YzM0OTQ5MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjQzMjE4NDg5OTk5OTk5NiwtMTIyLjgwMjgxMTY2MTE1Nzc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y3MGM0Y2NkNDBhNDQ2YmE5NjBhMTJhMThmNmQ3M2M3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM0MzljN2ZlYjFmNDQyNjM5NzRlMWYwNDJjOWJjMjgzID0gJCgnPGRpdiBpZD0iaHRtbF8zNDM5YzdmZWIxZjQ0MjYzOTc0ZTFmMDQyYzliYzI4MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QlJPT0tXQVkgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNzBjNGNjZDQwYTQ0NmJhOTYwYTEyYTE4ZjZkNzNjNy5zZXRDb250ZW50KGh0bWxfMzQzOWM3ZmViMWY0NDI2Mzk3NGUxZjA0MmM5YmMyODMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWQ4MTM5Mjc1YWNmNGI2MmE3Mzg5Nzk1NWMzNDk0OTAuYmluZFBvcHVwKHBvcHVwX2Y3MGM0Y2NkNDBhNDQ2YmE5NjBhMTJhMThmNmQ3M2M3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmZDljYjM1M2Y3ZjQ5ZTE4ZWJkNDVlOTViZjU5ZmM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDQ4MjYwMywtMC4wODg1MDczXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY2ZTA2OGU2YWY3NTQwMzU4OWVjZTlkYWMzOWY4ZjFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRjZDk2NzdiZWEwYzQ3NWM4ODNhYzc2MjVkMjYyZDBiID0gJCgnPGRpdiBpZD0iaHRtbF80Y2Q5Njc3YmVhMGM0NzVjODgzYWM3NjI1ZDI2MmQwYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QlVSQkFHRSBST0FEIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjZlMDY4ZTZhZjc1NDAzNTg5ZWNlOWRhYzM5ZjhmMWUuc2V0Q29udGVudChodG1sXzRjZDk2NzdiZWEwYzQ3NWM4ODNhYzc2MjVkMjYyZDBiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmZDljYjM1M2Y3ZjQ5ZTE4ZWJkNDVlOTViZjU5ZmM5LmJpbmRQb3B1cChwb3B1cF82NmUwNjhlNmFmNzU0MDM1ODllY2U5ZGFjMzlmOGYxZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xZTdlMmRiZmJkMjA0ZDVkOWZhNTgwOTEwMjhkZTE0MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjE0NTUyOTQsLTAuNDIzNTkzM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNjY4Nzc0OTI4MDk0Y2QyYTJiYzU0NmQ4M2U5NTFhZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjNhNTc0ZTk4OWE0ZTZkYWM5ZDFkNzE2Zjg5ZDU3MSA9ICQoJzxkaXYgaWQ9Imh0bWxfYjYzYTU3NGU5ODlhNGU2ZGFjOWQxZDcxNmY4OWQ1NzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJVUlkgV0FMSyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U2Njg3NzQ5MjgwOTRjZDJhMmJjNTQ2ZDgzZTk1MWFkLnNldENvbnRlbnQoaHRtbF9iNjNhNTc0ZTk4OWE0ZTZkYWM5ZDFkNzE2Zjg5ZDU3MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZTdlMmRiZmJkMjA0ZDVkOWZhNTgwOTEwMjhkZTE0My5iaW5kUG9wdXAocG9wdXBfZTY2ODc3NDkyODA5NGNkMmEyYmM1NDZkODNlOTUxYWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmI2YTlmMGJiY2M4NDA3ZGI3NjAzNmQ2YmJmNGY0OWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDgzNDk5LC0wLjE5ODMyNzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzkxYjE3OWRlOWIxNGJjN2EzZGJjZTY3OGE5MDMxNjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzE2Y2I0YzVjMjU1NDMyMmE1NmFiNzEyMDE3ODNmYWUgPSAkKCc8ZGl2IGlkPSJodG1sXzMxNmNiNGM1YzI1NTQzMjJhNTZhYjcxMjAxNzgzZmFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DQUxMQ09UVCBTVFJFRVQgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOTFiMTc5ZGU5YjE0YmM3YTNkYmNlNjc4YTkwMzE2Mi5zZXRDb250ZW50KGh0bWxfMzE2Y2I0YzVjMjU1NDMyMmE1NmFiNzEyMDE3ODNmYWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmI2YTlmMGJiY2M4NDA3ZGI3NjAzNmQ2YmJmNGY0OWIuYmluZFBvcHVwKHBvcHVwXzM5MWIxNzlkZTliMTRiYzdhM2RiY2U2NzhhOTAzMTYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMwYmVjMjI3NWY5NjRlNjM4YTA5ZmY2NDg2MzI5ZDA2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAxNDEsLTAuMTk1MTE1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MTc2YmY2ODVjYmM0ZTNiOTViZWRiNmYxNjkxNDAyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNjU0MTMxYmZiNjQ0ZmFiOTBjYTgyNmM4ZWFjYWVhZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDY1NDEzMWJmYjY0NGZhYjkwY2E4MjZjOGVhY2FlYWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBTVBERU4gSElMTCBST0FEIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjE3NmJmNjg1Y2JjNGUzYjk1YmVkYjZmMTY5MTQwMjAuc2V0Q29udGVudChodG1sXzA2NTQxMzFiZmI2NDRmYWI5MGNhODI2YzhlYWNhZWFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMwYmVjMjI3NWY5NjRlNjM4YTA5ZmY2NDg2MzI5ZDA2LmJpbmRQb3B1cChwb3B1cF82MTc2YmY2ODVjYmM0ZTNiOTViZWRiNmYxNjkxNDAyMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YzM5YTc1OTkzNWQ0ZmY5OTk5MjlkN2NhNDRiMmVkZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjY4MTM3NDksMC45NjU0NzEzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzM2U1NjI2NTA4ZjRkZTRiMzY4ZGEwYTY4ZjU1ZmQ2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFiNWYzNmM4MTIwYzQyNjlhYjdkYzQ5Y2EyZjFhZTgzID0gJCgnPGRpdiBpZD0iaHRtbF8xYjVmMzZjODEyMGM0MjY5YWI3ZGM0OWNhMmYxYWU4MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0FNUElPTiBST0FEIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTMzZTU2MjY1MDhmNGRlNGIzNjhkYTBhNjhmNTVmZDYuc2V0Q29udGVudChodG1sXzFiNWYzNmM4MTIwYzQyNjlhYjdkYzQ5Y2EyZjFhZTgzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVjMzlhNzU5OTM1ZDRmZjk5OTkyOWQ3Y2E0NGIyZWRkLmJpbmRQb3B1cChwb3B1cF81MzNlNTYyNjUwOGY0ZGU0YjM2OGRhMGE2OGY1NWZkNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNGE4NTVhZGZiMWU0YmU5ODU1ZDQzMjFjYmI5YzdhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5OTU2OTYsLTAuMTg0MjQ3N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hOGZhMmRhZTRjYzI0NmFiOTM2NWI1NmRkNDNlZjhhYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lM2I1NDhiY2IwYmY0ZDAyYWVjNDMxMzY3NDVjY2EyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTNiNTQ4YmNiMGJmNGQwMmFlYzQzMTM2NzQ1Y2NhMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBTk5JTkcgUExBQ0UgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hOGZhMmRhZTRjYzI0NmFiOTM2NWI1NmRkNDNlZjhhYy5zZXRDb250ZW50KGh0bWxfZTNiNTQ4YmNiMGJmNGQwMmFlYzQzMTM2NzQ1Y2NhMmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRhODU1YWRmYjFlNGJlOTg1NWQ0MzIxY2JiOWM3YWUuYmluZFBvcHVwKHBvcHVwX2E4ZmEyZGFlNGNjMjQ2YWI5MzY1YjU2ZGQ0M2VmOGFjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RjNTg4YTYyNGE0MjRiMzBiZDVmMmYzY2U2MDRlMTA5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTM2LjcwOTE3MTUsMTc0LjcyODI4MDVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDEzOTc0NTk0MWZhNGU2ZGI2NzM2ZWRhYTMyMjMyOWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjEzZDdiNTcyMmVhNDE1ZThjZDljZjExMjQ4MTkzMTcgPSAkKCc8ZGl2IGlkPSJodG1sXzIxM2Q3YjU3MjJlYTQxNWU4Y2Q5Y2YxMTI0ODE5MzE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DQVJMSVNMRSBST0FEIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDEzOTc0NTk0MWZhNGU2ZGI2NzM2ZWRhYTMyMjMyOWMuc2V0Q29udGVudChodG1sXzIxM2Q3YjU3MjJlYTQxNWU4Y2Q5Y2YxMTI0ODE5MzE3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RjNTg4YTYyNGE0MjRiMzBiZDVmMmYzY2U2MDRlMTA5LmJpbmRQb3B1cChwb3B1cF8wMTM5NzQ1OTQxZmE0ZTZkYjY3MzZlZGFhMzIyMzI5Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZTg3Yzg4OWQ3Yzk0MDIwYjcyZTk2MWQ2ZTY2M2NlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWy0zNy44MDE5NDMzNSwxNDQuOTcxOTcwMTcxMDE3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMmNjODY1ZGUxYjQ0YTMyOTViOTk5NTliZDFiOWZlNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hOWYxYjQzNTAzOTk0NTE1OWJmYWYyZTdiMDRjNDlmOSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTlmMWI0MzUwMzk5NDUxNTliZmFmMmU3YjA0YzQ5ZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNBUkxUT04gR0FSREVOUyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IyY2M4NjVkZTFiNDRhMzI5NWI5OTk1OWJkMWI5ZmU0LnNldENvbnRlbnQoaHRtbF9hOWYxYjQzNTAzOTk0NTE1OWJmYWYyZTdiMDRjNDlmOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZTg3Yzg4OWQ3Yzk0MDIwYjcyZTk2MWQ2ZTY2M2NlNi5iaW5kUG9wdXAocG9wdXBfYjJjYzg2NWRlMWI0NGEzMjk1Yjk5OTU5YmQxYjlmZTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzBjYjg1ZTViODBhNDU3ODhhYzAyMDBhYzU4ZWYzZDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMi45NzI3MDA5NTAwMDAwMDQsLTk3LjE3MzM5MTcwOTc3MTk1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkZDgxZTZhYTNkOTRkM2JiYmI2NmI4NDQyYjVkZDRjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5OTFhNWM2ODllODRhZmI4MTU2MmUzMmM5NGMxZWYyID0gJCgnPGRpdiBpZD0iaHRtbF9iOTkxYTVjNjg5ZTg0YWZiODE1NjJlMzJjOTRjMWVmMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0FSTFlMRSBDT1VSVCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NkZDgxZTZhYTNkOTRkM2JiYmI2NmI4NDQyYjVkZDRjLnNldENvbnRlbnQoaHRtbF9iOTkxYTVjNjg5ZTg0YWZiODE1NjJlMzJjOTRjMWVmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MGNiODVlNWI4MGE0NTc4OGFjMDIwMGFjNThlZjNkOC5iaW5kUG9wdXAocG9wdXBfY2RkODFlNmFhM2Q5NGQzYmJiYjY2Yjg0NDJiNWRkNGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2ZhMjRjZWMxZDM3NGRjOTk3YWY3MzcyMWNjZmNmYzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41NDExOTU1LC0wLjE1NTgxNjhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWUwNDc4OWU1M2UyNGQzMWFlNjEzZGQzZmEwN2MyZGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDQzZWFmNDJhNThhNDFmOWEyYzYxZmZjMDBiYjRhNzYgPSAkKCc8ZGl2IGlkPSJodG1sXzQ0M2VhZjQyYTU4YTQxZjlhMmM2MWZmYzAwYmI0YTc2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEFMQ09UIFNRVUFSRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlMDQ3ODllNTNlMjRkMzFhZTYxM2RkM2ZhMDdjMmRhLnNldENvbnRlbnQoaHRtbF80NDNlYWY0MmE1OGE0MWY5YTJjNjFmZmMwMGJiNGE3Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZmEyNGNlYzFkMzc0ZGM5OTdhZjczNzIxY2NmY2ZjNi5iaW5kUG9wdXAocG9wdXBfMWUwNDc4OWU1M2UyNGQzMWFlNjEzZGQzZmEwN2MyZGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWUxZTU2NGFmMmM3NDgxMjkxYmQ4OTU2NTMyMjBkNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MzM4MzcsLTAuMTcwMjk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk2ZDk0NzI5YjllZDRlNWRhYmQ3Nzc2N2E4ZGYyOGZhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdhZTgwNjQ1NTFhZDQzN2ZiYTJiM2YyMGEwNzM4ZGQ2ID0gJCgnPGRpdiBpZD0iaHRtbF83YWU4MDY0NTUxYWQ0MzdmYmEyYjNmMjBhMDczOGRkNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0hBUkxFUyBMQU5FIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTZkOTQ3MjliOWVkNGU1ZGFiZDc3NzY3YThkZjI4ZmEuc2V0Q29udGVudChodG1sXzdhZTgwNjQ1NTFhZDQzN2ZiYTJiM2YyMGEwNzM4ZGQ2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlMWU1NjRhZjJjNzQ4MTI5MWJkODk1NjUzMjIwZDY4LmJpbmRQb3B1cChwb3B1cF85NmQ5NDcyOWI5ZWQ0ZTVkYWJkNzc3NjdhOGRmMjhmYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMTZjNmFkMGVjODU0YmEyOTU2ZmFmNDIwNjhjMjRlZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM0LjUyMjQ0MywtODUuNDQzODkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRmN2RjN2M5NjMzYzQ0ZjY4OTBjN2FkZTgxMTM4NGI3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIxMjUwMTQxMDFkNzRmNDQ4OWI1NTVhMzMzNjFhMmI5ID0gJCgnPGRpdiBpZD0iaHRtbF8yMTI1MDE0MTAxZDc0ZjQ0ODliNTU1YTMzMzYxYTJiOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0hFTFNFQSBDUkVTQ0VOVCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmN2RjN2M5NjMzYzQ0ZjY4OTBjN2FkZTgxMTM4NGI3LnNldENvbnRlbnQoaHRtbF8yMTI1MDE0MTAxZDc0ZjQ0ODliNTU1YTMzMzYxYTJiOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMTZjNmFkMGVjODU0YmEyOTU2ZmFmNDIwNjhjMjRlZC5iaW5kUG9wdXAocG9wdXBfNGY3ZGM3Yzk2MzNjNDRmNjg5MGM3YWRlODExMzg0YjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2Q2ODdhNzI0YmNiNDBlZWE4OTg2YmM1MzFhMmMwYzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjkyMDU0LC0wLjE0NTA4MTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTdkMDgxZjZhMDg1NDFmODljYTgzNzM5MTgzODhlMDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmU5MTJmOTllMzRmNGY4MGE4NGYyOWE0ZGFmYWU0N2UgPSAkKCc8ZGl2IGlkPSJodG1sXzZlOTEyZjk5ZTM0ZjRmODBhODRmMjlhNGRhZmFlNDdlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEVTVEVSIENMT1NFIE5PUlRIIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTdkMDgxZjZhMDg1NDFmODljYTgzNzM5MTgzODhlMDAuc2V0Q29udGVudChodG1sXzZlOTEyZjk5ZTM0ZjRmODBhODRmMjlhNGRhZmFlNDdlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdkNjg3YTcyNGJjYjQwZWVhODk4NmJjNTMxYTJjMGMyLmJpbmRQb3B1cChwb3B1cF85N2QwODFmNmEwODU0MWY4OWNhODM3MzkxODM4OGUwMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YmMxY2FmNDA2OTY0NGMwYjI5MGEyYzJhMzFiZDcxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU5OTY3NywwLjUyNTYyMzFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmY4NWJhODUxMzM4NGJmY2I4OGIyZGY4ZmZkYTg1YmQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTFiYmRiMDkzMThiNDY0NzgxYzEyZjc2YWNkYzY5OTAgPSAkKCc8ZGl2IGlkPSJodG1sX2ExYmJkYjA5MzE4YjQ2NDc4MWMxMmY3NmFjZGM2OTkwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEVZTkUgQ09VUlQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82Zjg1YmE4NTEzMzg0YmZjYjg4YjJkZjhmZmRhODViZC5zZXRDb250ZW50KGh0bWxfYTFiYmRiMDkzMThiNDY0NzgxYzEyZjc2YWNkYzY5OTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWJjMWNhZjQwNjk2NDRjMGIyOTBhMmMyYTMxYmQ3MTQuYmluZFBvcHVwKHBvcHVwXzZmODViYTg1MTMzODRiZmNiODhiMmRmOGZmZGE4NWJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5ZGM0OWNmNmI0NzQ4ZjNhZGI5YWZiYTNjOTQ4NWQwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgzNzE3MywtMC4xNjk2MDNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzJmMWQyZWZlOWMwNDBjOThkMzZlZTQ4ZDllYjM3YzUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDg1N2RlMzVhZDUyNDBjYjhkZDU5NjViYThkYTI4YjkgPSAkKCc8ZGl2IGlkPSJodG1sXzQ4NTdkZTM1YWQ1MjQwY2I4ZGQ1OTY1YmE4ZGEyOGI5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DSEVZTkUgUk9XIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzJmMWQyZWZlOWMwNDBjOThkMzZlZTQ4ZDllYjM3YzUuc2V0Q29udGVudChodG1sXzQ4NTdkZTM1YWQ1MjQwY2I4ZGQ1OTY1YmE4ZGEyOGI5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E5ZGM0OWNmNmI0NzQ4ZjNhZGI5YWZiYTNjOTQ4NWQwLmJpbmRQb3B1cChwb3B1cF83MmYxZDJlZmU5YzA0MGM5OGQzNmVlNDhkOWViMzdjNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNmRkNjczMTQxNWQ0MzJhODM5NzZlOWQyNmI5MTk1ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NzE4NDksLTAuMjQ4MDE2OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMTQ5NmE5YmMyMjQ0ZjIxYmVhM2NhMjgwOTBjNTY3MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMDNmZTI4ZmJiNWI0MjNjOWZkN2ZkOTJjOWE2NTIyNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjAzZmUyOGZiYjViNDIzYzlmZDdmZDkyYzlhNjUyMjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNISVNXSUNLIE1BTEwgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMTQ5NmE5YmMyMjQ0ZjIxYmVhM2NhMjgwOTBjNTY3MC5zZXRDb250ZW50KGh0bWxfMjAzZmUyOGZiYjViNDIzYzlmZDdmZDkyYzlhNjUyMjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjZkZDY3MzE0MTVkNDMyYTgzOTc2ZTlkMjZiOTE5NWQuYmluZFBvcHVwKHBvcHVwX2YxNDk2YTliYzIyNDRmMjFiZWEzY2EyODA5MGM1NjcwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NmOTczMWRmNjgzNjRiZmZhNGY0ZTY3MWI4MGVjZTA5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI5Njk3MiwtMC4wOTc3NjI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RmOGY2ZWE3MDQ4NDRmMWM5MTFiOWNhZmM5OGE1ZDUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0NDNjNWQ4ZTJkOTQzNDdiMjU1MTRiNWVkYzBiMWJiID0gJCgnPGRpdiBpZD0iaHRtbF83NDQzYzVkOGUyZDk0MzQ3YjI1NTE0YjVlZGMwYjFiYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0lUWSBST0FEIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGY4ZjZlYTcwNDg0NGYxYzkxMWI5Y2FmYzk4YTVkNTEuc2V0Q29udGVudChodG1sXzc0NDNjNWQ4ZTJkOTQzNDdiMjU1MTRiNWVkYzBiMWJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NmOTczMWRmNjgzNjRiZmZhNGY0ZTY3MWI4MGVjZTA5LmJpbmRQb3B1cChwb3B1cF9kZjhmNmVhNzA0ODQ0ZjFjOTExYjljYWZjOThhNWQ1MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xM2U3YjNhMjBkZDk0ODAzYjZhOWU0YTE3ZTg1Njg1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjM2NTE2LDEuMTA4NTY5Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83NGRkYWEyNDU4NTA0NTJlYmZmZTk1ZmQ1NTdiNWM1YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNWVlMDZlYjY0YTM0NjlhOTBmZDEzYWM1MTJhN2YyYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMzVlZTA2ZWI2NGEzNDY5YTkwZmQxM2FjNTEyYTdmMmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNMQVJFTkRPTiBTVFJFRVQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NGRkYWEyNDU4NTA0NTJlYmZmZTk1ZmQ1NTdiNWM1YS5zZXRDb250ZW50KGh0bWxfMzVlZTA2ZWI2NGEzNDY5YTkwZmQxM2FjNTEyYTdmMmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTNlN2IzYTIwZGQ5NDgwM2I2YTllNGExN2U4NTY4NTkuYmluZFBvcHVwKHBvcHVwXzc0ZGRhYTI0NTg1MDQ1MmViZmZlOTVmZDU1N2I1YzVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzE2OWQ4MGIwYzZkZDQyYTE4MDk3MjEwYThmYTJhOWU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDczNzYzMiwtMC4yMTYyNDQyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZkMzc2NTllYmQ5ZTQwMWM4OWFhYjkzZmIzZmJlODg5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JlM2I4N2U4N2MyYjQ4ZjI5OTg0YzJhOTkyMjE3YWMxID0gJCgnPGRpdiBpZD0iaHRtbF9iZTNiODdlODdjMmI0OGYyOTk4NGMyYTk5MjIxN2FjMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q0xPTkNVUlJZIFNUUkVFVCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZkMzc2NTllYmQ5ZTQwMWM4OWFhYjkzZmIzZmJlODg5LnNldENvbnRlbnQoaHRtbF9iZTNiODdlODdjMmI0OGYyOTk4NGMyYTk5MjIxN2FjMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNjlkODBiMGM2ZGQ0MmExODA5NzIxMGE4ZmEyYTllOS5iaW5kUG9wdXAocG9wdXBfZmQzNzY1OWViZDllNDAxYzg5YWFiOTNmYjNmYmU4ODkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWFhYmY4ZDliY2U3NDRlZjhjYzgyNzQ0NzBlYTZkNzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTI5NDk3LC0wLjE4NTkyMTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWYzMzY1MDBkNDQ2NGY1OWI0ZDQ1YzY0MzQ4MThjYTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmVlYjU0MDEyYmQwNGQ0N2FjODRmZjkxMzMzOWU2MTAgPSAkKCc8ZGl2IGlkPSJodG1sXzZlZWI1NDAxMmJkMDRkNDdhYzg0ZmY5MTMzMzllNjEwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DT0xCRUNLIE1FV1MgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xZjMzNjUwMGQ0NDY0ZjU5YjRkNDVjNjQzNDgxOGNhOS5zZXRDb250ZW50KGh0bWxfNmVlYjU0MDEyYmQwNGQ0N2FjODRmZjkxMzMzOWU2MTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWFhYmY4ZDliY2U3NDRlZjhjYzgyNzQ0NzBlYTZkNzQuYmluZFBvcHVwKHBvcHVwXzFmMzM2NTAwZDQ0NjRmNTliNGQ0NWM2NDM0ODE4Y2E5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMxOTVhM2UzOTQ5NTQzYjliY2Q1NmEyMWU5MmM3NTdkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDIuODE4NjIxOSwtNzMuOTI1ODgxM10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMGQ0ZWZkNjE0NmI0ZmFiOTA4NDczMGI5ZDFmOTFhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNTc0MzY0MDExOTc0Nzk3YWRhYWRiNGNhZTkzNmI1MyA9ICQoJzxkaXYgaWQ9Imh0bWxfYTU3NDM2NDAxMTk3NDc5N2FkYWFkYjRjYWU5MzZiNTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNPTExFR0UgQ1JFU0NFTlQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMGQ0ZWZkNjE0NmI0ZmFiOTA4NDczMGI5ZDFmOTFhOC5zZXRDb250ZW50KGh0bWxfYTU3NDM2NDAxMTk3NDc5N2FkYWFkYjRjYWU5MzZiNTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzE5NWEzZTM5NDk1NDNiOWJjZDU2YTIxZTkyYzc1N2QuYmluZFBvcHVwKHBvcHVwXzMwZDRlZmQ2MTQ2YjRmYWI5MDg0NzMwYjlkMWY5MWE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I0NzU0ZDIwNTkwYjRiM2U5NDhiZjE5Njc0MjJmZjY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTI0MDY1NywtMC4xNTc2NjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGFmMTY3MjliOGJlNGM3ODkxMGMxOWNmNWM0NDc0YzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjFiNjE0MGIxMzViNGEwNmE0NDRhYWU2ZWQ3MzViNDQgPSAkKCc8ZGl2IGlkPSJodG1sX2IxYjYxNDBiMTM1YjRhMDZhNDQ0YWFlNmVkNzM1YjQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DT1JOV0FMTCBURVJSQUNFIE1FV1MgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYWYxNjcyOWI4YmU0Yzc4OTEwYzE5Y2Y1YzQ0NzRjNi5zZXRDb250ZW50KGh0bWxfYjFiNjE0MGIxMzViNGEwNmE0NDRhYWU2ZWQ3MzViNDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjQ3NTRkMjA1OTBiNGIzZTk0OGJmMTk2NzQyMmZmNjQuYmluZFBvcHVwKHBvcHVwX2RhZjE2NzI5YjhiZTRjNzg5MTBjMTljZjVjNDQ3NGM2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI0MzQ1OGE2OTYxNTQ1Njg5MmVkM2IxYjJkOTUxOTg0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDQ4NTAwMiwtMC4wODAxMDE2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVhZTQyMmVkMjBhZDQ4ODU5NmRkZjc5ODE0ODM3YmVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA5N2RmOWM1MGY2ZjQzYTRiZDk4MDhhNDBhYjVkNzZmID0gJCgnPGRpdiBpZD0iaHRtbF8wOTdkZjljNTBmNmY0M2E0YmQ5ODA4YTQwYWI1ZDc2ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q09VUlQgTEFORSBHQVJERU5TIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWFlNDIyZWQyMGFkNDg4NTk2ZGRmNzk4MTQ4MzdiZWMuc2V0Q29udGVudChodG1sXzA5N2RmOWM1MGY2ZjQzYTRiZDk4MDhhNDBhYjVkNzZmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI0MzQ1OGE2OTYxNTQ1Njg5MmVkM2IxYjJkOTUxOTg0LmJpbmRQb3B1cChwb3B1cF81YWU0MjJlZDIwYWQ0ODg1OTZkZGY3OTgxNDgzN2JlYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYTcwNzgzN2E0ZDE0MzI3OGM4M2ZmY2Y1OWNmY2VlMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ2LjExNTU4NTYsLTYwLjcyNTMzMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmMyZTU1NDAyYjNiNDFhYmI4NDFlN2M2OGVmZmZiNWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmY4MzNmMDU0YzVkNDNmY2JhMWYwNjY3OGZlYjQ3MzkgPSAkKCc8ZGl2IGlkPSJodG1sXzZmODMzZjA1NGM1ZDQzZmNiYTFmMDY2NzhmZWI0NzM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DUkVTQ0VOVCBHUk9WRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZjMmU1NTQwMmIzYjQxYWJiODQxZTdjNjhlZmZmYjVlLnNldENvbnRlbnQoaHRtbF82ZjgzM2YwNTRjNWQ0M2ZjYmExZjA2Njc4ZmViNDczOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYTcwNzgzN2E0ZDE0MzI3OGM4M2ZmY2Y1OWNmY2VlMC5iaW5kUG9wdXAocG9wdXBfNmMyZTU1NDAyYjNiNDFhYmI4NDFlN2M2OGVmZmZiNWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWEwM2IzODJiZWVhNGE3MmFkYjkyNmMyNDJkOWM3ZDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40MzgyNjgzLC0wLjE2NzY2OTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjE3OTdhNDY1ZTU0NDU4Y2FlODY5MjBmNDdmNjNjYTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjUzNTNjOGFiNzY2NGI4MmIzYTA2YjI4MjBkMTE0MjIgPSAkKCc8ZGl2IGlkPSJodG1sX2Y1MzUzYzhhYjc2NjRiODJiM2EwNmIyODIwZDExNDIyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EQUxFQlVSWSBST0FEIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjE3OTdhNDY1ZTU0NDU4Y2FlODY5MjBmNDdmNjNjYTUuc2V0Q29udGVudChodG1sX2Y1MzUzYzhhYjc2NjRiODJiM2EwNmIyODIwZDExNDIyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFhMDNiMzgyYmVlYTRhNzJhZGI5MjZjMjQyZDljN2Q5LmJpbmRQb3B1cChwb3B1cF9iMTc5N2E0NjVlNTQ0NThjYWU4NjkyMGY0N2Y2M2NhNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYWQzMmRkZGNmZjE0MzU3OWQ5ZDQ0ZGM5MDkwOGU4MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5ODczNzUsLTAuMjIwNjY2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMDE5ZmMyNTM5YTQ0MmMxYjI0ZTc3YzIxNDA5OTQ1NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNWM0ZjljOTdkZmE0NmNhYjQ0YjY2NjQ0YmY1NGFlMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzVjNGY5Yzk3ZGZhNDZjYWI0NGI2NjY0NGJmNTRhZTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRFV0hVUlNUIFJPQUQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMDE5ZmMyNTM5YTQ0MmMxYjI0ZTc3YzIxNDA5OTQ1NC5zZXRDb250ZW50KGh0bWxfMzVjNGY5Yzk3ZGZhNDZjYWI0NGI2NjY0NGJmNTRhZTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGFkMzJkZGRjZmYxNDM1NzlkOWQ0NGRjOTA5MDhlODMuYmluZFBvcHVwKHBvcHVwXzMwMTlmYzI1MzlhNDQyYzFiMjRlNzdjMjE0MDk5NDU0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E0NGIyMDY2N2I4YTRlZTFiMTEwODAwNzRkYWViNDBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDczMTE1NywtMC4yMDE3NDhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWFiODkxMzZlZmVlNDRhNGFjNTRkZjU2NDQzYjQwMTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWQ4MGZmYzc4NGY1NDllODk3M2VkMGQyMTY4ZDRkMTMgPSAkKCc8ZGl2IGlkPSJodG1sXzlkODBmZmM3ODRmNTQ5ZTg5NzNlZDBkMjE2OGQ0ZDEzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ET1JJQSBST0FEIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWFiODkxMzZlZmVlNDRhNGFjNTRkZjU2NDQzYjQwMTIuc2V0Q29udGVudChodG1sXzlkODBmZmM3ODRmNTQ5ZTg5NzNlZDBkMjE2OGQ0ZDEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E0NGIyMDY2N2I4YTRlZTFiMTEwODAwNzRkYWViNDBmLmJpbmRQb3B1cChwb3B1cF85YWI4OTEzNmVmZWU0NGE0YWM1NGRmNTY0NDNiNDAxMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kODhkYmNiYTZiMjM0ODk5YWU4NGMxYmI3NzM4NWU2MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU1NTY2MiwtMC4xNzAyOTM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIwZjM5MTVmY2NhNzRkMzA4MTdkYjNkZmQyM2ZmNzRjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxODQ2YzNlNzk2NTRiMWM4YjhmODU1ODJlY2Q3NGM2ID0gJCgnPGRpdiBpZD0iaHRtbF8zMTg0NmMzZTc5NjU0YjFjOGI4Zjg1NTgyZWNkNzRjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RE9XTlNISVJFIEhJTEwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yMGYzOTE1ZmNjYTc0ZDMwODE3ZGIzZGZkMjNmZjc0Yy5zZXRDb250ZW50KGh0bWxfMzE4NDZjM2U3OTY1NGIxYzhiOGY4NTU4MmVjZDc0YzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDg4ZGJjYmE2YjIzNDg5OWFlODRjMWJiNzczODVlNjIuYmluZFBvcHVwKHBvcHVwXzIwZjM5MTVmY2NhNzRkMzA4MTdkYjNkZmQyM2ZmNzRjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkYjg1MmFlMmNhNjQ2YWNhMjA1NGI4MmY2ZTNmOWI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAzODAxLC0wLjA3NjkyMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmRlZmQwZWRmN2Y4NDc1OThjZDRjNTBiOTYyYjY0YTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWUyYTk5ODgyMjQwNGQ2Njg1ZTRlNzUwNWU4MjgxNGEgPSAkKCc8ZGl2IGlkPSJodG1sXzVlMmE5OTg4MjI0MDRkNjY4NWU0ZTc1MDVlODI4MTRhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EVUNIRVNTIFdBTEsgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZGVmZDBlZGY3Zjg0NzU5OGNkNGM1MGI5NjJiNjRhOC5zZXRDb250ZW50KGh0bWxfNWUyYTk5ODgyMjQwNGQ2Njg1ZTRlNzUwNWU4MjgxNGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2RiODUyYWUyY2E2NDZhY2EyMDU0YjgyZjZlM2Y5YjguYmluZFBvcHVwKHBvcHVwX2ZkZWZkMGVkZjdmODQ3NTk4Y2Q0YzUwYjk2MmI2NGE4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAyMTc2MzI0YjI4NzQ5ZjViOTYxOWViY2ZhMDNhN2VhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDkxNzg0OSwtMC4xNDIyNTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUyY2I1ZTI4OGYzYTRmOWM4ODdkOGRiY2E5ODUzNWRlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg2NWNiMmJkNTIyMzRhYmY4YzFiNDYxOTNhZjJhOGMyID0gJCgnPGRpdiBpZD0iaHRtbF84NjVjYjJiZDUyMjM0YWJmOGMxYjQ2MTkzYWYyYThjMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RUNDTEVTVE9OIFNRVUFSRSBNRVdTIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTJjYjVlMjg4ZjNhNGY5Yzg4N2Q4ZGJjYTk4NTM1ZGUuc2V0Q29udGVudChodG1sXzg2NWNiMmJkNTIyMzRhYmY4YzFiNDYxOTNhZjJhOGMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAyMTc2MzI0YjI4NzQ5ZjViOTYxOWViY2ZhMDNhN2VhLmJpbmRQb3B1cChwb3B1cF81MmNiNWUyODhmM2E0ZjljODg3ZDhkYmNhOTg1MzVkZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNWM0MzI1YTRlYjc0NDMzYjI3ZThjNzk3MjkwMDhhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUzLjUwNzIxOCwtMi4xOTA2MDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RjZTI5YjhmZTU4NTRkMmQ4ZDJkNzM5ZjEzZjllMDMyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI4NWI2YTliMTQyZTQ2MWZiNjM1ZTliZjA1Y2JiM2Q1ID0gJCgnPGRpdiBpZD0iaHRtbF8yODViNmE5YjE0MmU0NjFmYjYzNWU5YmYwNWNiYjNkNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RUdCRVJUIFNUUkVFVCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RjZTI5YjhmZTU4NTRkMmQ4ZDJkNzM5ZjEzZjllMDMyLnNldENvbnRlbnQoaHRtbF8yODViNmE5YjE0MmU0NjFmYjYzNWU5YmYwNWNiYjNkNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNWM0MzI1YTRlYjc0NDMzYjI3ZThjNzk3MjkwMDhhMi5iaW5kUG9wdXAocG9wdXBfZGNlMjliOGZlNTg1NGQyZDhkMmQ3MzlmMTNmOWUwMzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODk0NWRmYzRkMDJlNDBhZWIyZjg0MTU5MjZjMzhkYTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTY2ODY3LC0wLjE2Njk0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTlmOWNhOWRiM2Y0NDEyOGE5ZmJhY2E4YzU0MzgyZjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmQwNmQzOTI3MmNiNGVjYjk4ZWE0MmUwODliOWQzZGUgPSAkKCc8ZGl2IGlkPSJodG1sX2ZkMDZkMzkyNzJjYjRlY2I5OGVhNDJlMDg5YjlkM2RlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FR0VSVE9OIFBMQUNFIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTlmOWNhOWRiM2Y0NDEyOGE5ZmJhY2E4YzU0MzgyZjAuc2V0Q29udGVudChodG1sX2ZkMDZkMzkyNzJjYjRlY2I5OGVhNDJlMDg5YjlkM2RlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg5NDVkZmM0ZDAyZTQwYWViMmY4NDE1OTI2YzM4ZGE4LmJpbmRQb3B1cChwb3B1cF9lOWY5Y2E5ZGIzZjQ0MTI4YTlmYmFjYThjNTQzODJmMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMTk5YWE1OGFiZTk0NGJjOWExNmRmZjYyMWIwYTJmYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjYzMzk4NzMsLTAuMDkyNTI0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwNjFiMTQyNjE1ODQ0Nzc4Nzc1MTljMGNjMDY5NTc4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U2M2MzOGNlNDZjMTQ0ZDJiZDViOWJkYTY4NTVkNjcxID0gJCgnPGRpdiBpZD0iaHRtbF9lNjNjMzhjZTQ2YzE0NGQyYmQ1YjliZGE2ODU1ZDY3MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RUxNIFBBUksgUk9BRCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwNjFiMTQyNjE1ODQ0Nzc4Nzc1MTljMGNjMDY5NTc4LnNldENvbnRlbnQoaHRtbF9lNjNjMzhjZTQ2YzE0NGQyYmQ1YjliZGE2ODU1ZDY3MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMTk5YWE1OGFiZTk0NGJjOWExNmRmZjYyMWIwYTJmYS5iaW5kUG9wdXAocG9wdXBfZDA2MWIxNDI2MTU4NDQ3Nzg3NzUxOWMwY2MwNjk1NzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDJmZTI4ZjI1NmU3NGU1OGE2ZmNmMzFhNTgwY2I2ODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTMyMTA1LC0wLjEyMjk3MzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDk1NTVhMjM3ZDBmNDg3MGI1MTA2OTExOTZkNThjMTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGIwNjhmZjdkOGMwNDU1NmJlMDhiMjViNzU5MGQxMDggPSAkKCc8ZGl2IGlkPSJodG1sXzRiMDY4ZmY3ZDhjMDQ1NTZiZTA4YjI1Yjc1OTBkMTA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GTE9SQUwgU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDk1NTVhMjM3ZDBmNDg3MGI1MTA2OTExOTZkNThjMTMuc2V0Q29udGVudChodG1sXzRiMDY4ZmY3ZDhjMDQ1NTZiZTA4YjI1Yjc1OTBkMTA4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyZmUyOGYyNTZlNzRlNThhNmZjZjMxYTU4MGNiNjg3LmJpbmRQb3B1cChwb3B1cF80OTU1NWEyMzdkMGY0ODcwYjUxMDY5MTE5NmQ1OGMxMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNGQ1MGNhNjMyZTE0M2RmYTQzMDM3MTc0YTc5Nzc0ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0Mjc5NDksLTAuMDgwMzk3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MTJjMDMwYjYxNzI0MmZlYWNiMzY1NGUwOGZmZmI4NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNzcwMjAwN2U0YjI0MGE1OGNkYWFhZjU1MTA3MWMxZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDc3MDIwMDdlNGIyNDBhNThjZGFhYWY1NTEwNzFjMWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZSQU5LIERJWE9OIFdBWSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQxMmMwMzBiNjE3MjQyZmVhY2IzNjU0ZTA4ZmZmYjg3LnNldENvbnRlbnQoaHRtbF8wNzcwMjAwN2U0YjI0MGE1OGNkYWFhZjU1MTA3MWMxZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNGQ1MGNhNjMyZTE0M2RmYTQzMDM3MTc0YTc5Nzc0ZC5iaW5kUG9wdXAocG9wdXBfNDEyYzAzMGI2MTcyNDJmZWFjYjM2NTRlMDhmZmZiODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGQxYjAxYTA4M2FmNDU4OGJmZWViZjFhZjI1YmQ5NzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTI1NTgyLC0wLjE4NDYyODVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWQ3ZGY1MjU1MzgyNGE4YjlhYTk0ZWRjMDdlOTYwMzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzRiNzMzZTRlZTcyNGRlODgxMjJiNDMxMzZmYzg5OTcgPSAkKCc8ZGl2IGlkPSJodG1sXzM0YjczM2U0ZWU3MjRkZTg4MTIyYjQzMTM2ZmM4OTk3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GVUxUT04gTUVXUyBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFkN2RmNTI1NTM4MjRhOGI5YWE5NGVkYzA3ZTk2MDM3LnNldENvbnRlbnQoaHRtbF8zNGI3MzNlNGVlNzI0ZGU4ODEyMmI0MzEzNmZjODk5Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZDFiMDFhMDgzYWY0NTg4YmZlZWJmMWFmMjViZDk3NS5iaW5kUG9wdXAocG9wdXBfMWQ3ZGY1MjU1MzgyNGE4YjlhYTk0ZWRjMDdlOTYwMzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjEzNTZlNWNjZmViNDliMjg5N2MxYjg2OWU0ZGFhYzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi4yMDk2OTMxLDAuMTU4ODc1Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNWU3OWU0YTBmZGE0NjY4ODVmMmU0NDM3MzVjODYyNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZGEzYzFjN2Q1OTg0YTMyYjlhN2RjN2NhOWM5YjhmOSA9ICQoJzxkaXYgaWQ9Imh0bWxfY2RhM2MxYzdkNTk4NGEzMmI5YTdkYzdjYTljOWI4ZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdFUkFSRCBST0FEIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTVlNzllNGEwZmRhNDY2ODg1ZjJlNDQzNzM1Yzg2Mjcuc2V0Q29udGVudChodG1sX2NkYTNjMWM3ZDU5ODRhMzJiOWE3ZGM3Y2E5YzliOGY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIxMzU2ZTVjY2ZlYjQ5YjI4OTdjMWI4NjllNGRhYWM5LmJpbmRQb3B1cChwb3B1cF8xNWU3OWU0YTBmZGE0NjY4ODVmMmU0NDM3MzVjODYyNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMDk2ZGVhZDQ2NWE0NTYyOTQ0ZjViNDdlMzgwMWJlNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUzMzg2NTYsLTAuMTAxMjQ3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZWM0ZjE2NThlZjM0NmQwYmI2YmEwY2ZlZGM5YWMxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNTIzNzhlOGRjYTU0M2I3YjE1ZDRiMTY5MjE1NTVlOCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTUyMzc4ZThkY2E1NDNiN2IxNWQ0YjE2OTIxNTU1ZTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdFUlJBUkQgUk9BRCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZlYzRmMTY1OGVmMzQ2ZDBiYjZiYTBjZmVkYzlhYzFmLnNldENvbnRlbnQoaHRtbF9lNTIzNzhlOGRjYTU0M2I3YjE1ZDRiMTY5MjE1NTVlOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMDk2ZGVhZDQ2NWE0NTYyOTQ0ZjViNDdlMzgwMWJlNC5iaW5kUG9wdXAocG9wdXBfZmVjNGYxNjU4ZWYzNDZkMGJiNmJhMGNmZWRjOWFjMWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzA4OTUzYzM3NTBhNDk4MTg0NWNkMGU2ZTMwMDFlNDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTY2OTY4LC0wLjIxNTUzMDZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDk5NzIzZjU5YjY3NDJlOWI3YjZhNTIxOGI1NDhmZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzcxYmEwMDhhY2QzNDQ3ZWIxNTI4MDkwMmQ3MDQ0MmMgPSAkKCc8ZGl2IGlkPSJodG1sX2M3MWJhMDA4YWNkMzQ0N2ViMTUyODA5MDJkNzA0NDJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HSVJETEVSUyBST0FEIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDk5NzIzZjU5YjY3NDJlOWI3YjZhNTIxOGI1NDhmZjYuc2V0Q29udGVudChodG1sX2M3MWJhMDA4YWNkMzQ0N2ViMTUyODA5MDJkNzA0NDJjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMwODk1M2MzNzUwYTQ5ODE4NDVjZDBlNmUzMDAxZTQ1LmJpbmRQb3B1cChwb3B1cF9kOTk3MjNmNTliNjc0MmU5YjdiNmE1MjE4YjU0OGZmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZWFiYTYxYzMyNjU0OTJiODZiOTZlZGYyOWU1YWJkNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ5LjYzMzgyMzUsLTExOC40MDY2ODU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E4Y2ZlNjhiMGY0ZjQ1NDM4ODFmMDg3NjhiZjhlZjQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q4MTA5NjY4YjMzMTRmNzdiMjRkMzFmNzA0OWFmZjk0ID0gJCgnPGRpdiBpZD0iaHRtbF9kODEwOTY2OGIzMzE0Zjc3YjI0ZDMxZjcwNDlhZmY5NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R0xPVUNFU1RFUiBDUkVTQ0VOVCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E4Y2ZlNjhiMGY0ZjQ1NDM4ODFmMDg3NjhiZjhlZjQyLnNldENvbnRlbnQoaHRtbF9kODEwOTY2OGIzMzE0Zjc3YjI0ZDMxZjcwNDlhZmY5NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZWFiYTYxYzMyNjU0OTJiODZiOTZlZGYyOWU1YWJkNi5iaW5kUG9wdXAocG9wdXBfYThjZmU2OGIwZjRmNDU0Mzg4MWYwODc2OGJmOGVmNDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTE1MTg5MGIzNWQ2NGVmZjgzY2VhOTY1ZGMzZjE3ODIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstNDEuMTY2NDk3OSwxNDYuMzQ2MzU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NmMTc0ZTczOGQyYTQ4ZjliOTk0NjVhNjkxNGFkNGVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YyYzlkYjdiYTcxMTQxYjQ4YTA1YTE2MjhhNmQzOTExID0gJCgnPGRpdiBpZD0iaHRtbF9mMmM5ZGI3YmE3MTE0MWI0OGEwNWExNjI4YTZkMzkxMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R09SRE9OIFBMQUNFIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2YxNzRlNzM4ZDJhNDhmOWI5OTQ2NWE2OTE0YWQ0ZWUuc2V0Q29udGVudChodG1sX2YyYzlkYjdiYTcxMTQxYjQ4YTA1YTE2MjhhNmQzOTExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ExNTE4OTBiMzVkNjRlZmY4M2NlYTk2NWRjM2YxNzgyLmJpbmRQb3B1cChwb3B1cF9jZjE3NGU3MzhkMmE0OGY5Yjk5NDY1YTY5MTRhZDRlZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZGRlMzEzZDE4ZjU0ZDZhODY5MTdkNzFmNjZiNGRlZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ2Mzk2NTksLTAuMTM5MDg0M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85OWEzZmNlNGMzMjU0MjExOTlmMGZkODA0NDc2ZDk4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZjI2YWI2OGQ2OTA0NzBkOThkYmJjMDc5NzJjMjQwYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNWYyNmFiNjhkNjkwNDcwZDk4ZGJiYzA3OTcyYzI0MGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdSQUZUT04gU1FVQVJFIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTlhM2ZjZTRjMzI1NDIxMTk5ZjBmZDgwNDQ3NmQ5ODEuc2V0Q29udGVudChodG1sXzVmMjZhYjY4ZDY5MDQ3MGQ5OGRiYmMwNzk3MmMyNDBiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVkZGUzMTNkMThmNTRkNmE4NjkxN2Q3MWY2NmI0ZGVkLmJpbmRQb3B1cChwb3B1cF85OWEzZmNlNGMzMjU0MjExOTlmMGZkODA0NDc2ZDk4MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lNmRmNTU5ODJhZmM0NzA2YjY4NWY5M2IzNjc1MzljOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5MTU0NzQsLTAuMTU0Mjc1MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84MzBkODU0MmM2MjY0NmE5OWU2MDAyYzY0Y2MzM2UyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMTA0ZGExZjE3NGQ0M2FmYmMwMDZlYTdlMjc1MjNkNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzEwNGRhMWYxNzRkNDNhZmJjMDA2ZWE3ZTI3NTIzZDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdSQUhBTSBURVJSQUNFIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODMwZDg1NDJjNjI2NDZhOTllNjAwMmM2NGNjMzNlMjAuc2V0Q29udGVudChodG1sXzMxMDRkYTFmMTc0ZDQzYWZiYzAwNmVhN2UyNzUyM2Q1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U2ZGY1NTk4MmFmYzQ3MDZiNjg1ZjkzYjM2NzUzOWM5LmJpbmRQb3B1cChwb3B1cF84MzBkODU0MmM2MjY0NmE5OWU2MDAyYzY0Y2MzM2UyMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZWZlMTBmNGRmMTA0MTAzOTc1ZjUxMmQzODBjOGMzZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU1ODczNzksLTAuMjA2MzA2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZDJiMjE1M2I3YzQ0NDNhOTBkNjgxODY4YTFjNjc5OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zY2RiM2QzZWVjMDM0NTA0YmZkMDE3MjIyNzIwYzJiMiA9ICQoJzxkaXYgaWQ9Imh0bWxfM2NkYjNkM2VlYzAzNDUwNGJmZDAxNzIyMjcyMGMyYjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhBUk1BTiBEUklWRSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRkMmIyMTUzYjdjNDQ0M2E5MGQ2ODE4NjhhMWM2Nzk5LnNldENvbnRlbnQoaHRtbF8zY2RiM2QzZWVjMDM0NTA0YmZkMDE3MjIyNzIwYzJiMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZWZlMTBmNGRmMTA0MTAzOTc1ZjUxMmQzODBjOGMzZC5iaW5kUG9wdXAocG9wdXBfNGQyYjIxNTNiN2M0NDQzYTkwZDY4MTg2OGExYzY3OTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDk0YzlkYzhkNjM2NDMxYmJlODYzNzVhNGNkYjJhMWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFstMzMuODIzOTc2NSwxNTEuMDEwNTEyMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYWVjMjg3M2JiYTM0ZGFmODdhZjUyNTVmYzIxZmViNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNjM2Y2M2Njg5ZDY0NTIyOGFkM2YxYzhlYzk1ZTk5MSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzYzNmNjNjY4OWQ2NDUyMjhhZDNmMWM4ZWM5NWU5OTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhBUlJJUyBTVFJFRVQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYWVjMjg3M2JiYTM0ZGFmODdhZjUyNTVmYzIxZmViNi5zZXRDb250ZW50KGh0bWxfYzYzNmNjNjY4OWQ2NDUyMjhhZDNmMWM4ZWM5NWU5OTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDk0YzlkYzhkNjM2NDMxYmJlODYzNzVhNGNkYjJhMWMuYmluZFBvcHVwKHBvcHVwX2RhZWMyODczYmJhMzRkYWY4N2FmNTI1NWZjMjFmZWI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I0NGE4YzgzNWFiOTQwNmNhN2QyZWRjZDA2YTQwMDFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk5NjMyNiwtMC4wMjI5NzQ2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAwZmU4YjU3NTZmNzQxMTU4NjMyMTBhZDQ1NDc5NjM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzOTU5NzhhNzdkYjQyZjVhZmY4NTZjM2MzNjEwMzc2ID0gJCgnPGRpdiBpZD0iaHRtbF85Mzk1OTc4YTc3ZGI0MmY1YWZmODU2YzNjMzYxMDM3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SEFWQU5OQUggU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDBmZThiNTc1NmY3NDExNTg2MzIxMGFkNDU0Nzk2MzQuc2V0Q29udGVudChodG1sXzkzOTU5NzhhNzdkYjQyZjVhZmY4NTZjM2MzNjEwMzc2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I0NGE4YzgzNWFiOTQwNmNhN2QyZWRjZDA2YTQwMDFmLmJpbmRQb3B1cChwb3B1cF8wMGZlOGI1NzU2Zjc0MTE1ODYzMjEwYWQ0NTQ3OTYzNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMDU4ODVkYzliNTk0YzU4YjQyMGE3Njk0OTM3YmU1NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ1OTQzMiwtMC4yMjcxNTAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBhZTVkZGU2NGVjYzQwYmRhOTg2ZWZiZmYxMTUwNjBhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEyNWVmNzZjNGQxZjQxYWViNGNiYWZkNzEzOTk5YzZhID0gJCgnPGRpdiBpZD0iaHRtbF8xMjVlZjc2YzRkMWY0MWFlYjRjYmFmZDcxMzk5OWM2YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SEFaTEVXRUxMIFJPQUQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYWU1ZGRlNjRlY2M0MGJkYTk4NmVmYmZmMTE1MDYwYS5zZXRDb250ZW50KGh0bWxfMTI1ZWY3NmM0ZDFmNDFhZWI0Y2JhZmQ3MTM5OTljNmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTA1ODg1ZGM5YjU5NGM1OGI0MjBhNzY5NDkzN2JlNTUuYmluZFBvcHVwKHBvcHVwXzBhZTVkZGU2NGVjYzQwYmRhOTg2ZWZiZmYxMTUwNjBhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3OTFjNTQ4YzNmODRmZjU5ZDdiNTdmYmM1Y2IwNTI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTE1NTAxMiwtMC4xOTM1MDk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE1ZThjMTIzNDRjNzQ2MzJiNDVjMDNlMGM0MjE1NDFiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y2NTk4ZmEwMTFlMzQ4MDQ5YzBiMjFlNWQ5YWM4ZjUzID0gJCgnPGRpdiBpZD0iaHRtbF9mNjU5OGZhMDExZTM0ODA0OWMwYjIxZTVkOWFjOGY1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SEVSRUZPUkQgTUVXUyBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE1ZThjMTIzNDRjNzQ2MzJiNDVjMDNlMGM0MjE1NDFiLnNldENvbnRlbnQoaHRtbF9mNjU5OGZhMDExZTM0ODA0OWMwYjIxZTVkOWFjOGY1Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NzkxYzU0OGMzZjg0ZmY1OWQ3YjU3ZmJjNWNiMDUyNC5iaW5kUG9wdXAocG9wdXBfMTVlOGMxMjM0NGM3NDYzMmI0NWMwM2UwYzQyMTU0MWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTk2MjNiNGIzMDJhNDQxN2IyZjYxOWUzZDg5MDcyZjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NDM2Nzc1LC0wLjE3NDYyODNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDgwYmQzYTY2OGY1NGEyNzg2YjMwYTFiZGIyNjE4NTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2ZhYmM4M2ViNzJjNDM5NmFjZjdlMDZiNjUxNjU5ODkgPSAkKCc8ZGl2IGlkPSJodG1sXzNmYWJjODNlYjcyYzQzOTZhY2Y3ZTA2YjY1MTY1OTg5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IRVJPTkRBTEUgQVZFTlVFIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDgwYmQzYTY2OGY1NGEyNzg2YjMwYTFiZGIyNjE4NTcuc2V0Q29udGVudChodG1sXzNmYWJjODNlYjcyYzQzOTZhY2Y3ZTA2YjY1MTY1OTg5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk5NjIzYjRiMzAyYTQ0MTdiMmY2MTllM2Q4OTA3MmY2LmJpbmRQb3B1cChwb3B1cF80ODBiZDNhNjY4ZjU0YTI3ODZiMzBhMWJkYjI2MTg1Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NTFlZmNmY2JmNDk0MDI3YjM3MGJiNjM0MGM0MTBkNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU3MTA0MTksLTAuMTQ4OTgzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ODAxNTlhMmY5ODQ0MTYzOTZkOTdmZmY4MTQxY2EwOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MzJhNWI3YjQ2NTY0OGM1YTdmYzMwMDI1NWI1ODI5MSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDMyYTViN2I0NjU2NDhjNWE3ZmMzMDAyNTViNTgyOTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhJR0hHQVRFIEhJR0ggU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDgwMTU5YTJmOTg0NDE2Mzk2ZDk3ZmZmODE0MWNhMDguc2V0Q29udGVudChodG1sXzQzMmE1YjdiNDY1NjQ4YzVhN2ZjMzAwMjU1YjU4MjkxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU1MWVmY2ZjYmY0OTQwMjdiMzcwYmI2MzQwYzQxMGQ3LmJpbmRQb3B1cChwb3B1cF80ODAxNTlhMmY5ODQ0MTYzOTZkOTdmZmY4MTQxY2EwOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYTc0NmQyZmZiMWU0ZjllYWZkZWU0YmZkMjYyMTkzZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjYzMjcyNjcsLTAuMjQwNDg5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YWFkYTk3MGIzZjM0MTExYWUyNTc2YTcwOGU0NGZhNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTY4NjExMTdmNTc0MTg2OGRkMmZjMDY5ZmYwNTA3ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfN2E2ODYxMTE3ZjU3NDE4NjhkZDJmYzA2OWZmMDUwN2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhJR0hXT09EIEhJTEwgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YWFkYTk3MGIzZjM0MTExYWUyNTc2YTcwOGU0NGZhNS5zZXRDb250ZW50KGh0bWxfN2E2ODYxMTE3ZjU3NDE4NjhkZDJmYzA2OWZmMDUwN2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGE3NDZkMmZmYjFlNGY5ZWFmZGVlNGJmZDI2MjE5M2YuYmluZFBvcHVwKHBvcHVwXzZhYWRhOTcwYjNmMzQxMTFhZTI1NzZhNzA4ZTQ0ZmE1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FhYjU1NmE1Y2VmZTRmMTlhOGQ1ZmZkZmI3NzBkMTU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTA3ODQ2NywtMC4xOTczOTY2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRhNGVhMTE4NmVhYTRiYjZhNjVlNDMyNjI1ZDA5MmVmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJkNTMzZjFjMGU3MTQyYzc5MTNiYzBiNTc4OWJkN2JhID0gJCgnPGRpdiBpZD0iaHRtbF8yZDUzM2YxYzBlNzE0MmM3OTEzYmMwYjU3ODliZDdiYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SElMTEdBVEUgUExBQ0UgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YTRlYTExODZlYWE0YmI2YTY1ZTQzMjYyNWQwOTJlZi5zZXRDb250ZW50KGh0bWxfMmQ1MzNmMWMwZTcxNDJjNzkxM2JjMGI1Nzg5YmQ3YmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWFiNTU2YTVjZWZlNGYxOWE4ZDVmZmRmYjc3MGQxNTguYmluZFBvcHVwKHBvcHVwXzRhNGVhMTE4NmVhYTRiYjZhNjVlNDMyNjI1ZDA5MmVmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEyOWFkZTY0MGJkYjQ0YWVhYmIyNzlmOWJhMWZhZTlmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTY1MDk5NywtMC4yOTA2Mzg0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q2MjE4MjcyMjc3NDQxM2I5YWVkNDFlOGJhYTUzNzY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk1NzkyYmIyNTZkNTQwM2U4MDFmN2U3MWY0NjI0MTIyID0gJCgnPGRpdiBpZD0iaHRtbF85NTc5MmJiMjU2ZDU0MDNlODAxZjdlNzFmNDYyNDEyMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SE9MTFlDUk9GVCBBVkVOVUUgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNjIxODI3MjI3NzQ0MTNiOWFlZDQxZThiYWE1Mzc2NC5zZXRDb250ZW50KGh0bWxfOTU3OTJiYjI1NmQ1NDAzZTgwMWY3ZTcxZjQ2MjQxMjIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTI5YWRlNjQwYmRiNDRhZWFiYjI3OWY5YmExZmFlOWYuYmluZFBvcHVwKHBvcHVwX2Q2MjE4MjcyMjc3NDQxM2I5YWVkNDFlOGJhYTUzNzY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEwZTY4ZmNkMTE5MDQzNTBhOTI5MzRkMWIzMzM5NTFjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDg2MjExMiwtMC4xODM3MTg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y2YTdhM2M2ZTVkNTQ1MjU5ZDk5NjQ0ZjkwYzA2ZmJkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q1ZGIyMmY0ZWJlZjQ3ZGNhNDc0ODY5YWYzMWJjODY0ID0gJCgnPGRpdiBpZD0iaHRtbF9kNWRiMjJmNGViZWY0N2RjYTQ3NDg2OWFmMzFiYzg2NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SE9MTFlXT09EIE1FV1MgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNmE3YTNjNmU1ZDU0NTI1OWQ5OTY0NGY5MGMwNmZiZC5zZXRDb250ZW50KGh0bWxfZDVkYjIyZjRlYmVmNDdkY2E0NzQ4NjlhZjMxYmM4NjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTBlNjhmY2QxMTkwNDM1MGE5MjkzNGQxYjMzMzk1MWMuYmluZFBvcHVwKHBvcHVwX2Y2YTdhM2M2ZTVkNTQ1MjU5ZDk5NjQ0ZjkwYzA2ZmJkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcxNzdlYjU5NGE2MzRjZDBhYzE0MmY3ODJhNzA5ZTRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDU0MzI5NiwtMC4xNjI3MDcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JjMmRlMzliOTBjNjQzZGQ4NzU4ZjI2NzEwOTM0NmFlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg2NGQ1NTdlN2MwYjQ5ZDY5ZWFhOTkxOTU3YjBhZmU0ID0gJCgnPGRpdiBpZD0iaHRtbF84NjRkNTU3ZTdjMGI0OWQ2OWVhYTk5MTk1N2IwYWZlNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SE9ORVlXRUxMIFJPQUQgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYzJkZTM5YjkwYzY0M2RkODc1OGYyNjcxMDkzNDZhZS5zZXRDb250ZW50KGh0bWxfODY0ZDU1N2U3YzBiNDlkNjllYWE5OTE5NTdiMGFmZTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzE3N2ViNTk0YTYzNGNkMGFjMTQyZjc4MmE3MDllNGIuYmluZFBvcHVwKHBvcHVwX2JjMmRlMzliOTBjNjQzZGQ4NzU4ZjI2NzEwOTM0NmFlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg0NjMyMjQ5NGRkMTQ2ZmE4YWI2NzBjNzA1MGVkNzY3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgxNzY3OSwtMC4xODUyMzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzZjMWI1NWEzNjQ1NDI5MTk5M2Q4MzVkNTYzOTE3MDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODc2MWE1NjZhOTIwNGU4NGE2OTA0MzkzM2MyNDlkNGQgPSAkKCc8ZGl2IGlkPSJodG1sXzg3NjFhNTY2YTkyMDRlODRhNjkwNDM5MzNjMjQ5ZDRkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IT1JURU5TSUEgUk9BRCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM2YzFiNTVhMzY0NTQyOTE5OTNkODM1ZDU2MzkxNzA3LnNldENvbnRlbnQoaHRtbF84NzYxYTU2NmE5MjA0ZTg0YTY5MDQzOTMzYzI0OWQ0ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NDYzMjI0OTRkZDE0NmZhOGFiNjcwYzcwNTBlZDc2Ny5iaW5kUG9wdXAocG9wdXBfMzZjMWI1NWEzNjQ1NDI5MTk5M2Q4MzVkNTYzOTE3MDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWQ2ZDE4MTQ0YmVjNGZiZWJlNmRjZDQ0YTkwZGNlMTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41Mjc1Nzg1LC0wLjA4MTE4ODExMDcxMTcyNDczXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzExYjUyYzRjMjNkYjRkNmE4YzNjNzMwZWM4Yzk3YzI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZlMDcxZTcyMjQ5NDQ1MzU5ODUxMmQ5NGRhZjg3NDA3ID0gJCgnPGRpdiBpZD0iaHRtbF82ZTA3MWU3MjI0OTQ0NTM1OTg1MTJkOTRkYWY4NzQwNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SE9YVE9OIFNRVUFSRSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzExYjUyYzRjMjNkYjRkNmE4YzNjNzMwZWM4Yzk3YzI5LnNldENvbnRlbnQoaHRtbF82ZTA3MWU3MjI0OTQ0NTM1OTg1MTJkOTRkYWY4NzQwNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZDZkMTgxNDRiZWM0ZmJlYmU2ZGNkNDRhOTBkY2UxNy5iaW5kUG9wdXAocG9wdXBfMTFiNTJjNGMyM2RiNGQ2YThjM2M3MzBlYzhjOTdjMjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzIxM2Q0ODQyMjEzNDY1OWEyODY1ZTdjZjlmZTZhYzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi42NTM0NjQ0LDEuMjg3NTczXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U5NmZjMDU5NWFmODRiZThiMDg1ODM4ZmZiYmFjNmJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIyY2JhMTVmNTQ1YjRhYmNhNjljZjQwMTI3NmRjODgyID0gJCgnPGRpdiBpZD0iaHRtbF8yMmNiYTE1ZjU0NWI0YWJjYTY5Y2Y0MDEyNzZkYzg4MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SFVOVEVSIFJPQUQgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOTZmYzA1OTVhZjg0YmU4YjA4NTgzOGZmYmJhYzZiYi5zZXRDb250ZW50KGh0bWxfMjJjYmExNWY1NDViNGFiY2E2OWNmNDAxMjc2ZGM4ODIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzIxM2Q0ODQyMjEzNDY1OWEyODY1ZTdjZjlmZTZhYzEuYmluZFBvcHVwKHBvcHVwX2U5NmZjMDU5NWFmODRiZThiMDg1ODM4ZmZiYmFjNmJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I4YjFjNTM0MGRhNTQyODE5YjRlMjU3OTdlMmI3MzZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTc2NjExNywtMC4xNDUzMzY3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4ZDU3NzI1MmMyZjQ4MGM5YThiYzE3YTViYzliNDhmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJhYTkwZWM4M2NiNDQ2MDJhYWY3OTE0ZWM1YzNmNzg4ID0gJCgnPGRpdiBpZD0iaHRtbF8yYWE5MGVjODNjYjQ0NjAyYWFmNzkxNGVjNWMzZjc4OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SkFDS1NPTlMgTEFORSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4ZDU3NzI1MmMyZjQ4MGM5YThiYzE3YTViYzliNDhmLnNldENvbnRlbnQoaHRtbF8yYWE5MGVjODNjYjQ0NjAyYWFmNzkxNGVjNWMzZjc4OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iOGIxYzUzNDBkYTU0MjgxOWI0ZTI1Nzk3ZTJiNzM2Yi5iaW5kUG9wdXAocG9wdXBfZjhkNTc3MjUyYzJmNDgwYzlhOGJjMTdhNWJjOWI0OGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzZmNzc5MTIzM2ZiNDk4ZDk1N2VkZTExMWI1ZTVkMTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0OC4xOTc2MjgyLDE2LjMyMDEyODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDUwNmE4MDIxMDExNGQ2NmJmM2QyM2JjM2E1NTliZmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzFiZDUxZTAzOWM3NDYyZjkxOTBkNzgxZjVjZDdmMzQgPSAkKCc8ZGl2IGlkPSJodG1sXzMxYmQ1MWUwMzljNzQ2MmY5MTkwZDc4MWY1Y2Q3ZjM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5KT0hOIFNUUkVFVCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ1MDZhODAyMTAxMTRkNjZiZjNkMjNiYzNhNTU5YmZjLnNldENvbnRlbnQoaHRtbF8zMWJkNTFlMDM5Yzc0NjJmOTE5MGQ3ODFmNWNkN2YzNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NmY3NzkxMjMzZmI0OThkOTU3ZWRlMTExYjVlNWQxMy5iaW5kUG9wdXAocG9wdXBfNDUwNmE4MDIxMDExNGQ2NmJmM2QyM2JjM2E1NTliZmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTgyOTI3YWJmZDE5NDVkNGI0NmFlYjY3MDNhM2ZiZGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDA2ODIzLC0wLjE1NjcxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNmEzYTk3ZDViZDY0OTkxODg5ZjM3Nzg0ZmQ2ZTg2YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MmM4ZDBmNTE5NzU0MjQzYTg3MTU0NTk0ZThjNDA3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTJjOGQwZjUxOTc1NDI0M2E4NzE1NDU5NGU4YzQwN2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktJTk5FUlRPTiBTVFJFRVQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNmEzYTk3ZDViZDY0OTkxODg5ZjM3Nzg0ZmQ2ZTg2Yy5zZXRDb250ZW50KGh0bWxfNTJjOGQwZjUxOTc1NDI0M2E4NzE1NDU5NGU4YzQwN2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTgyOTI3YWJmZDE5NDVkNGI0NmFlYjY3MDNhM2ZiZGEuYmluZFBvcHVwKHBvcHVwXzI2YTNhOTdkNWJkNjQ5OTE4ODlmMzc3ODRmZDZlODZjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjYjAyNWYyZGVlZDQ2ZjI4MDg3Zjg5MTY1YjA4MWFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk0NzYyNywtMC4xOTExMTk0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMxNmM2NWVkNWE3ODQ4ODNhNWMwYTQ1MDk5ZmFiNWY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y3MjQxNTFkZGJlMDQwZjM4YmQzYTM5MTQ0YjEwYzExID0gJCgnPGRpdiBpZD0iaHRtbF9mNzI0MTUxZGRiZTA0MGYzOGJkM2EzOTE0NGIxMGMxMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S05BUkVTQk9ST1VHSCBQTEFDRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxNmM2NWVkNWE3ODQ4ODNhNWMwYTQ1MDk5ZmFiNWY0LnNldENvbnRlbnQoaHRtbF9mNzI0MTUxZGRiZTA0MGYzOGJkM2EzOTE0NGIxMGMxMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zY2IwMjVmMmRlZWQ0NmYyODA4N2Y4OTE2NWIwODFhZi5iaW5kUG9wdXAocG9wdXBfMzE2YzY1ZWQ1YTc4NDg4M2E1YzBhNDUwOTlmYWI1ZjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGVhOGYzMTAzZGY2NDZmZTk1ZDVkY2RhYjNhZmZiYzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjA4ODY3LC0wLjE2MTQ1NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzc4NDA2OGFjYWMxNDIwNjg4YzY5YmQ0M2FmMDcyMzYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWI3ZThlYTBmMzAwNGIzOWEzNzQxZjU4NDRlN2VmZDAgPSAkKCc8ZGl2IGlkPSJodG1sX2ViN2U4ZWEwZjMwMDRiMzlhMzc0MWY1ODQ0ZTdlZmQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LTk9YIFNUUkVFVCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc3ODQwNjhhY2FjMTQyMDY4OGM2OWJkNDNhZjA3MjM2LnNldENvbnRlbnQoaHRtbF9lYjdlOGVhMGYzMDA0YjM5YTM3NDFmNTg0NGU3ZWZkMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZWE4ZjMxMDNkZjY0NmZlOTVkNWRjZGFiM2FmZmJjOC5iaW5kUG9wdXAocG9wdXBfNzc4NDA2OGFjYWMxNDIwNjg4YzY5YmQ0M2FmMDcyMzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDkyNmI4NGExNmFjNGYzZWIwOTIyMWMwMWFjMDMyMDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTcyNjM5LC0wLjIxMTEwMTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWViZDMxZWFlMWY2NDUwMjhhNDdjYWE3YThlOTQ0MWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmNhYzRkNmFjMDgwNGU2OWIyMzBkNWVlMDg0OWI4OTMgPSAkKCc8ZGl2IGlkPSJodG1sX2JjYWM0ZDZhYzA4MDRlNjliMjMwZDVlZTA4NDliODkzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MQURCUk9LRSBHUk9WRSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FlYmQzMWVhZTFmNjQ1MDI4YTQ3Y2FhN2E4ZTk0NDFmLnNldENvbnRlbnQoaHRtbF9iY2FjNGQ2YWMwODA0ZTY5YjIzMGQ1ZWUwODQ5Yjg5Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kOTI2Yjg0YTE2YWM0ZjNlYjA5MjIxYzAxYWMwMzIwMy5iaW5kUG9wdXAocG9wdXBfYWViZDMxZWFlMWY2NDUwMjhhNDdjYWE3YThlOTQ0MWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTU1NDMyN2YwNWY2NDExZDg3NmMxY2RhYTMyMzcwZGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTIyMzY2LC0wLjE3ODcyOTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDRkZGNjMWM3NjRmNDFiYjkxNTg2NjA3NzQxZjk1OTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDU2YTU4ODY0MmMwNGViMTlmNTg2OGI0MGE3NmE1NTMgPSAkKCc8ZGl2IGlkPSJodG1sX2Q1NmE1ODg2NDJjMDRlYjE5ZjU4NjhiNDBhNzZhNTUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MQU5DQVNURVIgTUVXUyBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA0ZGRjYzFjNzY0ZjQxYmI5MTU4NjYwNzc0MWY5NTkxLnNldENvbnRlbnQoaHRtbF9kNTZhNTg4NjQyYzA0ZWIxOWY1ODY4YjQwYTc2YTU1Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNTU0MzI3ZjA1ZjY0MTFkODc2YzFjZGFhMzIzNzBkYi5iaW5kUG9wdXAocG9wdXBfMDRkZGNjMWM3NjRmNDFiYjkxNTg2NjA3NzQxZjk1OTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2I1NWQyNDE2MDU4NDA4NmI0YTRjNWY5M2U4MGQ2ODQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1My4zMzUyMzMwOTk5OTk5OTYsLTYuMjI4MTc4NDcxNDY1MjI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlhNzcyM2EwMzc4MTRkNjViZGRiMTBjOWQxMDYzNjk2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVkYjM2NzVjYTdlZTRjNmRhNzI3ODM3ZjRiZTI1NTdiID0gJCgnPGRpdiBpZD0iaHRtbF81ZGIzNjc1Y2E3ZWU0YzZkYTcyNzgzN2Y0YmUyNTU3YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TEFOU0RPV05FIFJPQUQgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85YTc3MjNhMDM3ODE0ZDY1YmRkYjEwYzlkMTA2MzY5Ni5zZXRDb250ZW50KGh0bWxfNWRiMzY3NWNhN2VlNGM2ZGE3Mjc4MzdmNGJlMjU1N2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2I1NWQyNDE2MDU4NDA4NmI0YTRjNWY5M2U4MGQ2ODQuYmluZFBvcHVwKHBvcHVwXzlhNzcyM2EwMzc4MTRkNjViZGRiMTBjOWQxMDYzNjk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U0ZjViM2M1M2Y0MTQ5ZmVhYmQxZTRkYWVmMmMxZGI5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuMDMwNTQzMiwxLjIxNjI5MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWU1NDEwOTIzNjdkNDBiYWIzMzBlODAxYTQ0YTA1YjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjM0NDRjNjBlNWYzNGM5MGFhMmMxMjI2YWZjNDVkNDAgPSAkKCc8ZGl2IGlkPSJodG1sX2IzNDQ0YzYwZTVmMzRjOTBhYTJjMTIyNmFmYzQ1ZDQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MQVRJTUVSIElORFVTVFJJQUwgRVNUQVRFIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWU1NDEwOTIzNjdkNDBiYWIzMzBlODAxYTQ0YTA1YjAuc2V0Q29udGVudChodG1sX2IzNDQ0YzYwZTVmMzRjOTBhYTJjMTIyNmFmYzQ1ZDQwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U0ZjViM2M1M2Y0MTQ5ZmVhYmQxZTRkYWVmMmMxZGI5LmJpbmRQb3B1cChwb3B1cF8xZTU0MTA5MjM2N2Q0MGJhYjMzMGU4MDFhNDRhMDViMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNTM0ZWE3NTFkNmY0YjQ5YTlmZDEyNzdkZDBkN2ZhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUyNTY2NzgsLTAuMTQyMTM1MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYzIyMjhjMjMxZmQ0NzQyOTVhNWIyM2M1NDM2ZGEyZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOGFhZDVlNjhhZjI0OTJkOWYwMzhiY2NlNGJhMTVkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzhhYWQ1ZTY4YWYyNDkyZDlmMDM4YmNjZTRiYTE1ZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxBWFRPTiBQTEFDRSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VjMjIyOGMyMzFmZDQ3NDI5NWE1YjIzYzU0MzZkYTJlLnNldENvbnRlbnQoaHRtbF9jOGFhZDVlNjhhZjI0OTJkOWYwMzhiY2NlNGJhMTVkMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNTM0ZWE3NTFkNmY0YjQ5YTlmZDEyNzdkZDBkN2ZhMi5iaW5kUG9wdXAocG9wdXBfZWMyMjI4YzIzMWZkNDc0Mjk1YTViMjNjNTQzNmRhMmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjE2MmQ1MDBlODRmNDIyOGI2OGJmNzU4OTUwZTQxMDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszNy4zMDQ5Mjk3LC0xMjEuODk4NDk2MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MmM4ZTFlOThjMmQ0YmYwOTc3ZDE0N2I3NWMxNmNkMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MjA2ZjlkOWVhOGQ0OTk0OGUxOWRlNjFkNWFlMDE2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTIwNmY5ZDllYThkNDk5NDhlMTlkZTYxZDVhZTAxNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxJTkNPTE4gQVZFTlVFIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzJjOGUxZTk4YzJkNGJmMDk3N2QxNDdiNzVjMTZjZDEuc2V0Q29udGVudChodG1sXzkyMDZmOWQ5ZWE4ZDQ5OTQ4ZTE5ZGU2MWQ1YWUwMTY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2YxNjJkNTAwZTg0ZjQyMjhiNjhiZjc1ODk1MGU0MTAwLmJpbmRQb3B1cChwb3B1cF83MmM4ZTFlOThjMmQ0YmYwOTc3ZDE0N2I3NWMxNmNkMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMTljZjczZDIwYzc0OTk4ODNlOWE0YTkxM2VmMTQyZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjA4Mzg3NCwxLjE0Mzk4NjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmE3YTZlZDFiZDczNDYxNzhkM2E1NWExMTNjNjYyNTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTU3YzE1OTE2NmUwNDFmN2E2ODBkYzEwM2Q3YTNmMGMgPSAkKCc8ZGl2IGlkPSJodG1sX2U1N2MxNTkxNjZlMDQxZjdhNjgwZGMxMDNkN2EzZjBjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MSU5HRklFTEQgUk9BRCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZhN2E2ZWQxYmQ3MzQ2MTc4ZDNhNTVhMTEzYzY2MjUzLnNldENvbnRlbnQoaHRtbF9lNTdjMTU5MTY2ZTA0MWY3YTY4MGRjMTAzZDdhM2YwYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMTljZjczZDIwYzc0OTk4ODNlOWE0YTkxM2VmMTQyZS5iaW5kUG9wdXAocG9wdXBfZmE3YTZlZDFiZDczNDYxNzhkM2E1NWExMTNjNjYyNTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGUwZjExZmFhZDg3NDM0NjhlNGIyNTdjZGRmYzEzOGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjE1NDc0LC0wLjE2ODI1NjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzAwNDk5M2YxMmMxNGY5NWFhNDlmYjYzM2RmN2UwNDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWY1ZmQ5N2JjZGQ4NGNmNTg0ODM5YjcxMmE5Zjk5ZDcgPSAkKCc8ZGl2IGlkPSJodG1sXzFmNWZkOTdiY2RkODRjZjU4NDgzOWI3MTJhOWY5OWQ3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MSVNTT04gU1RSRUVUIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzAwNDk5M2YxMmMxNGY5NWFhNDlmYjYzM2RmN2UwNDYuc2V0Q29udGVudChodG1sXzFmNWZkOTdiY2RkODRjZjU4NDgzOWI3MTJhOWY5OWQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlMGYxMWZhYWQ4NzQzNDY4ZTRiMjU3Y2RkZmMxMzhkLmJpbmRQb3B1cChwb3B1cF9jMDA0OTkzZjEyYzE0Zjk1YWE0OWZiNjMzZGY3ZTA0Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MWFhYzliNjJkMzM0Y2M5ODNiNjc1YzRlMTc3MmUxMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NjM1MTMsLTAuMDkyMDEyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMDI1NWM1MTBjYTc0MmYzYjRlMGJlZjkzOTI1MzliNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MmM3MGQ1YzdhOTk0YWRiYjMyNzRlOWI0OWY5ZDUxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjJjNzBkNWM3YTk5NGFkYmIzMjc0ZTliNDlmOWQ1MWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxJVkVSUE9PTCBHUk9WRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwMjU1YzUxMGNhNzQyZjNiNGUwYmVmOTM5MjUzOWI1LnNldENvbnRlbnQoaHRtbF82MmM3MGQ1YzdhOTk0YWRiYjMyNzRlOWI0OWY5ZDUxYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MWFhYzliNjJkMzM0Y2M5ODNiNjc1YzRlMTc3MmUxMS5iaW5kUG9wdXAocG9wdXBfZDAyNTVjNTEwY2E3NDJmM2I0ZTBiZWY5MzkyNTM5YjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTgzYzQ0MmRjMGUyNGM1YjljMmQ0YzMzODdlYzhjOGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NTE2MDIxLC0wLjIzODI3NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjQ0NjA2MTUyMjI2NDNmMDg1YmY1NjMyOWE4NTQ2Y2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWVmNjljMTBmYmE2NDU1ZjhmMzllZDk2MmMxMTIyNDggPSAkKCc8ZGl2IGlkPSJodG1sXzllZjY5YzEwZmJhNjQ1NWY4ZjM5ZWQ5NjJjMTEyMjQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MT05HV09PRCBEUklWRSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I0NDYwNjE1MjIyNjQzZjA4NWJmNTYzMjlhODU0NmNkLnNldENvbnRlbnQoaHRtbF85ZWY2OWMxMGZiYTY0NTVmOGYzOWVkOTYyYzExMjI0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hODNjNDQyZGMwZTI0YzViOWMyZDRjMzM4N2VjOGM4Zi5iaW5kUG9wdXAocG9wdXBfYjQ0NjA2MTUyMjI2NDNmMDg1YmY1NjMyOWE4NTQ2Y2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGIxMTc5ZTkyMWFlNGIzMjg2YTUxNWEwMmY3ZTA3NDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41Mzk1ODQ4OTk5OTk5OTQsLTAuMTA4MTc2NTEyMjExNjgxOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZmNhMjc1OGJiOGI0MGUxYWFiNDBiNjNmOTc1NGM2MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMTc1MGJlZTA3MWI0N2I0YTU0OGI2MzEyNmRkMGUzZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMDE3NTBiZWUwNzFiNDdiNGE1NDhiNjMxMjZkZDBlM2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxPTlNEQUxFIFNRVUFSRSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVmY2EyNzU4YmI4YjQwZTFhYWI0MGI2M2Y5NzU0YzYxLnNldENvbnRlbnQoaHRtbF8wMTc1MGJlZTA3MWI0N2I0YTU0OGI2MzEyNmRkMGUzZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kYjExNzllOTIxYWU0YjMyODZhNTE1YTAyZjdlMDc0MS5iaW5kUG9wdXAocG9wdXBfNWZjYTI3NThiYjhiNDBlMWFhYjQwYjYzZjk3NTRjNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTZjZjc0MGUzZmE5NGFkNWIyYTBmZjFkOGE2NjFkYjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS4zNTMzOTYzLC03OS4xNTAzMTc2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y1NTJiNjhlYmU0ZDQ4ZDg4NzkxODBiYjM0ZDlkNzE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q2YmFhYTg2MDQ3YjQzMjc5MmEwYjc4Y2RlOWFhNjM0ID0gJCgnPGRpdiBpZD0iaHRtbF9kNmJhYWE4NjA0N2I0MzI3OTJhMGI3OGNkZTlhYTYzNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TUFaRSBISUxMIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjU1MmI2OGViZTRkNDhkODg3OTE4MGJiMzRkOWQ3MTYuc2V0Q29udGVudChodG1sX2Q2YmFhYTg2MDQ3YjQzMjc5MmEwYjc4Y2RlOWFhNjM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk2Y2Y3NDBlM2ZhOTRhZDViMmEwZmYxZDhhNjYxZGI0LmJpbmRQb3B1cChwb3B1cF9mNTUyYjY4ZWJlNGQ0OGQ4ODc5MTgwYmIzNGQ5ZDcxNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYTEyYTVhYzM5M2E0NzNmYmJkZDY2ZWFjNTFhYjE0YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUxODI4MzEsLTAuMDk5MTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMDU3MWE4OTNjM2Y0ZDEzYjYxNzVjNTM5ZTcwOWNiNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NmFiZDc5M2RiMzA0YzQ0OGFlNWQ1ZjY0NDBiYTI2NiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjZhYmQ3OTNkYjMwNGM0NDhhZTVkNWY2NDQwYmEyNjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1JRERMRVNFWCBQQVNTQUdFIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjA1NzFhODkzYzNmNGQxM2I2MTc1YzUzOWU3MDljYjUuc2V0Q29udGVudChodG1sXzY2YWJkNzkzZGIzMDRjNDQ4YWU1ZDVmNjQ0MGJhMjY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNhMTJhNWFjMzkzYTQ3M2ZiYmRkNjZlYWM1MWFiMTRhLmJpbmRQb3B1cChwb3B1cF8yMDU3MWE4OTNjM2Y0ZDEzYjYxNzVjNTM5ZTcwOWNiNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZWM2NzE2NTQxNDY0M2YyYTMxYjllZjY1MzQ4YWI4ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjM0MTA4MjYsMS4wMjExNzQ1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRjYWE0ZGIyNmQzZDRlNmNhMDhjOGMyMWEzMjg5MjgxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhiOTI1NDQ2ZTBjZjQzZTA5NzI2YjU2NjlkMWNmMDk3ID0gJCgnPGRpdiBpZD0iaHRtbF84YjkyNTQ0NmUwY2Y0M2UwOTcyNmI1NjY5ZDFjZjA5NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TU9OVFBFTElFUiBBVkVOVUUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Y2FhNGRiMjZkM2Q0ZTZjYTA4YzhjMjFhMzI4OTI4MS5zZXRDb250ZW50KGh0bWxfOGI5MjU0NDZlMGNmNDNlMDk3MjZiNTY2OWQxY2YwOTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWVjNjcxNjU0MTQ2NDNmMmEzMWI5ZWY2NTM0OGFiOGQuYmluZFBvcHVwKHBvcHVwXzRjYWE0ZGIyNmQzZDRlNmNhMDhjOGMyMWEzMjg5MjgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlNWJiNTI3ZWJjNTQ1ZDE5ZGNiMTQ1MDVhYTZkMDQxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDk4ODkzOCwtMC4xNjcxNzM1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJmZjQ0YjgyN2Y5YzQ1NjZhZmFiNWVjYjQwMzUxOWI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ1MGZiOGMwMWZhYjQ5NWI5YWM2ODk1MWE5NDVlMjU3ID0gJCgnPGRpdiBpZD0iaHRtbF80NTBmYjhjMDFmYWI0OTViOWFjNjg5NTFhOTQ1ZTI1NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TU9OVFBFTElFUiBXQUxLIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmZmNDRiODI3ZjljNDU2NmFmYWI1ZWNiNDAzNTE5Yjkuc2V0Q29udGVudChodG1sXzQ1MGZiOGMwMWZhYjQ5NWI5YWM2ODk1MWE5NDVlMjU3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlNWJiNTI3ZWJjNTQ1ZDE5ZGNiMTQ1MDVhYTZkMDQxLmJpbmRQb3B1cChwb3B1cF8yZmY0NGI4MjdmOWM0NTY2YWZhYjVlY2I0MDM1MTliOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OTY3NTM1YmJmMDk0MzhlYmJhYWEwNjkzZWY3OTNiNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ0NjUwMzEsLTAuMTc2MTczMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZWQ0Yjk5NmY1ZDI0N2JiOTI5MTY0OThjN2ZhOGQ2OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MGYyNTJiNDBhZmM0ZTk1OTU0ZGZkN2E0ZmRhZDEyYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNjBmMjUyYjQwYWZjNGU5NTk1NGRmZDdhNGZkYWQxMmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1VTFRPTiBST0FEIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWVkNGI5OTZmNWQyNDdiYjkyOTE2NDk4YzdmYThkNjguc2V0Q29udGVudChodG1sXzYwZjI1MmI0MGFmYzRlOTU5NTRkZmQ3YTRmZGFkMTJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg5Njc1MzViYmYwOTQzOGViYmFhYTA2OTNlZjc5M2I3LmJpbmRQb3B1cChwb3B1cF85ZWQ0Yjk5NmY1ZDI0N2JiOTI5MTY0OThjN2ZhOGQ2OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mODI4OWUwZjhlODQ0ODIyYjg4MTkwOWNkMWY4MDg5NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5NDI3NTQsLTAuMjEyNDkzNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYThhZTJjNzJjYjA0YzRiYTY1YzQ4OTVmMWI4N2EzYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OThmODRjNjRhMjM0OTViYjg2ZGY0YzcwMDNjMWJjMCA9ICQoJzxkaXYgaWQ9Imh0bWxfODk4Zjg0YzY0YTIzNDk1YmI4NmRmNGM3MDAzYzFiYzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1VTkRFTiBTVFJFRVQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hYThhZTJjNzJjYjA0YzRiYTY1YzQ4OTVmMWI4N2EzYi5zZXRDb250ZW50KGh0bWxfODk4Zjg0YzY0YTIzNDk1YmI4NmRmNGM3MDAzYzFiYzApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjgyODllMGY4ZTg0NDgyMmI4ODE5MDljZDFmODA4OTYuYmluZFBvcHVwKHBvcHVwX2FhOGFlMmM3MmNiMDRjNGJhNjVjNDg5NWYxYjg3YTNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA4ZjAyMGJkZDRiNDQ1M2ViMzI1NWQxNTIwYWE5ODM3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbLTM0Ljg0NTc5NzksMTQ5LjM5MDUyNzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGRkYzFiNjFhMzRmNGRmYmJiNzgwMmQ4YjQ2MmVlODEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2JlZTVkYThkMzIzNGEzZThkMDVkMmNjNDYzNzkzYTcgPSAkKCc8ZGl2IGlkPSJodG1sXzNiZWU1ZGE4ZDMyMzRhM2U4ZDA1ZDJjYzQ2Mzc5M2E3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OT1JGT0xLIENSRVNDRU5UIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGRkYzFiNjFhMzRmNGRmYmJiNzgwMmQ4YjQ2MmVlODEuc2V0Q29udGVudChodG1sXzNiZWU1ZGE4ZDMyMzRhM2U4ZDA1ZDJjYzQ2Mzc5M2E3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA4ZjAyMGJkZDRiNDQ1M2ViMzI1NWQxNTIwYWE5ODM3LmJpbmRQb3B1cChwb3B1cF8wZGRjMWI2MWEzNGY0ZGZiYmI3ODAyZDhiNDYyZWU4MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NWU2YTBiZWM3NDk0Yzk1YTM4MTJhYWY0MzU1NDdjOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU5OTc3OTQsLTAuMDAwNTY2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MWUxZWNmNDJkZjc0NDFmYjM0ZjBjODg3ZWU5ZmEyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kYmUzZmIwMzZiYmY0MDVmODE0NjQ2YTc4MjliMGUxNyA9ICQoJzxkaXYgaWQ9Imh0bWxfZGJlM2ZiMDM2YmJmNDA1ZjgxNDY0NmE3ODI5YjBlMTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5PUlRIIENJUkNVTEFSIFJPQUQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82MWUxZWNmNDJkZjc0NDFmYjM0ZjBjODg3ZWU5ZmEyMC5zZXRDb250ZW50KGh0bWxfZGJlM2ZiMDM2YmJmNDA1ZjgxNDY0NmE3ODI5YjBlMTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzVlNmEwYmVjNzQ5NGM5NWEzODEyYWFmNDM1NTQ3YzguYmluZFBvcHVwKHBvcHVwXzYxZTFlY2Y0MmRmNzQ0MWZiMzRmMGM4ODdlZTlmYTIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI0ZWZkNTFlZTk1NzRhNjg4ZjFmZTFhMzU4YmY4MjYzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTIxMzIzMSwtMC4xNTI1MzU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q5YzQ3M2VhYmNmYjQ2Y2M5ZjA5ZDY2MjFhNjMxOWYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2NTAwZTg5NWZkMDRhYWE4NWFlNzUyY2YzZGU0MjVhID0gJCgnPGRpdiBpZD0iaHRtbF9iNjUwMGU4OTVmZDA0YWFhODVhZTc1MmNmM2RlNDI1YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tk9UVElOR0hBTSBTVFJFRVQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kOWM0NzNlYWJjZmI0NmNjOWYwOWQ2NjIxYTYzMTlmMy5zZXRDb250ZW50KGh0bWxfYjY1MDBlODk1ZmQwNGFhYTg1YWU3NTJjZjNkZTQyNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjRlZmQ1MWVlOTU3NGE2ODhmMWZlMWEzNThiZjgyNjMuYmluZFBvcHVwKHBvcHVwX2Q5YzQ3M2VhYmNmYjQ2Y2M5ZjA5ZDY2MjFhNjMxOWYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VhZmIyNzhhMmZkNzRkMTBhNWQwYzZmOWY2NjlmYjMyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDgzNjQ4MiwtMC4xNjczMjQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RkNDljN2ZiOTM2ODRkMTZhN2M0ZDYyM2UxMTM3MTI2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQxMjA0ODg1MjRiYzQzNThhNWZjMWUxMjE0M2YwMTg2ID0gJCgnPGRpdiBpZD0iaHRtbF80MTIwNDg4NTI0YmM0MzU4YTVmYzFlMTIxNDNmMDE4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0FLTEVZIFNUUkVFVCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkNDljN2ZiOTM2ODRkMTZhN2M0ZDYyM2UxMTM3MTI2LnNldENvbnRlbnQoaHRtbF80MTIwNDg4NTI0YmM0MzU4YTVmYzFlMTIxNDNmMDE4Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYWZiMjc4YTJmZDc0ZDEwYTVkMGM2ZjlmNjY5ZmIzMi5iaW5kUG9wdXAocG9wdXBfZGQ0OWM3ZmI5MzY4NGQxNmE3YzRkNjIzZTExMzcxMjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGIxNDM0MzFiNjg5NGY5ZGI3NWRiMDE5Zjg0YjYyNjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0NS41MDU1NTU2LC05Mi45Nzk0NDQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzFlMGRjNjdkYzQzOTRjYjliMjE4YWNiZTZkZjVmZmM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY0ZjFhNDNlMmQ2ZDQwZjQ5Mzk1Njg2MDcwOTM0NTFkID0gJCgnPGRpdiBpZD0iaHRtbF82NGYxYTQzZTJkNmQ0MGY0OTM5NTY4NjA3MDkzNDUxZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T0FLV09PRCBDT1VSVCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlMGRjNjdkYzQzOTRjYjliMjE4YWNiZTZkZjVmZmM2LnNldENvbnRlbnQoaHRtbF82NGYxYTQzZTJkNmQ0MGY0OTM5NTY4NjA3MDkzNDUxZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80YjE0MzQzMWI2ODk0ZjlkYjc1ZGIwMTlmODRiNjI2NC5iaW5kUG9wdXAocG9wdXBfMWUwZGM2N2RjNDM5NGNiOWIyMThhY2JlNmRmNWZmYzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWYyOGM0OGQ4OTFlNDRkOGEzM2IzM2FlOGU3YzM5ZDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MDM4NTY1LC0wLjE5NjAzNDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDJmYjE2NzA0ZjU5NDZmNDk0N2ZhZjNmMDRlYjNjY2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjhkOGJmYWM5NjZhNDRmYmFjNGExZDE1ZDEyMGNmMWYgPSAkKCc8ZGl2IGlkPSJodG1sX2Y4ZDhiZmFjOTY2YTQ0ZmJhYzRhMWQxNWQxMjBjZjFmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PQlNFUlZBVE9SWSBHQVJERU5TIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDJmYjE2NzA0ZjU5NDZmNDk0N2ZhZjNmMDRlYjNjY2Muc2V0Q29udGVudChodG1sX2Y4ZDhiZmFjOTY2YTQ0ZmJhYzRhMWQxNWQxMjBjZjFmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmMjhjNDhkODkxZTQ0ZDhhMzNiMzNhZThlN2MzOWQwLmJpbmRQb3B1cChwb3B1cF9kMmZiMTY3MDRmNTk0NmY0OTQ3ZmFmM2YwNGViM2NjYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZmQyOTVhZGY4NGQ0NWUxOGRlOTg0NWYyNDkxM2VhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUwMjg2ODUsLTAuMTkxMTcwNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZWQ2YzhkNWZjMDk0OTEwYWNhNzM5MmY5YTJmOTc0ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iODRiMjg3MjNkZWU0ZmQwYjUzNDhjYjJkMTJjZTI1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfYjg0YjI4NzIzZGVlNGZkMGI1MzQ4Y2IyZDEyY2UyNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9MRCBDT1VSVCBQTEFDRSBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JlZDZjOGQ1ZmMwOTQ5MTBhY2E3MzkyZjlhMmY5NzRlLnNldENvbnRlbnQoaHRtbF9iODRiMjg3MjNkZWU0ZmQwYjUzNDhjYjJkMTJjZTI1YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZmQyOTVhZGY4NGQ0NWUxOGRlOTg0NWYyNDkxM2VhMi5iaW5kUG9wdXAocG9wdXBfYmVkNmM4ZDVmYzA5NDkxMGFjYTczOTJmOWEyZjk3NGUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTJiZjI0M2M4ZWQzNGE0ODgyMzBhZGI0OTBjNGMxMDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTE0ODIyLC0wLjE3NzI1NTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTA5NWVhOTRkNGFiNDViZGE3MWUzMjhhMTI4MTYwMDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjYyYzMxMTkxNjAxNDQ5NjllMWQ3OGRjZDRjZWFiODUgPSAkKCc8ZGl2IGlkPSJodG1sXzI2MmMzMTE5MTYwMTQ0OTY5ZTFkNzhkY2Q0Y2VhYjg1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PTlNMT1cgTUVXUyBXRVNUIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTA5NWVhOTRkNGFiNDViZGE3MWUzMjhhMTI4MTYwMDAuc2V0Q29udGVudChodG1sXzI2MmMzMTE5MTYwMTQ0OTY5ZTFkNzhkY2Q0Y2VhYjg1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUyYmYyNDNjOGVkMzRhNDg4MjMwYWRiNDkwYzRjMTA3LmJpbmRQb3B1cChwb3B1cF81MDk1ZWE5NGQ0YWI0NWJkYTcxZTMyOGExMjgxNjAwMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MWU0ZmU3MTkzNDg0YTM4YTY4YTEzY2ZhYTZiMmEyNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQxLjcxMTY2OCwtNzAuNDc3NDEzOTk0MDcxMTRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmM2ZTkzOGQzNjY0NGEwZTg1ZjA0YmJhMmQ0ZjM4MWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzJjYWY1ZDllMzBkNGE4YmExZjcwY2NjODI3ZTJmM2QgPSAkKCc8ZGl2IGlkPSJodG1sXzMyY2FmNWQ5ZTMwZDRhOGJhMWY3MGNjYzgyN2UyZjNkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QQUxBQ0UgUExBQ0UgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYzZlOTM4ZDM2NjQ0YTBlODVmMDRiYmEyZDRmMzgxYi5zZXRDb250ZW50KGh0bWxfMzJjYWY1ZDllMzBkNGE4YmExZjcwY2NjODI3ZTJmM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTFlNGZlNzE5MzQ4NGEzOGE2OGExM2NmYWE2YjJhMjUuYmluZFBvcHVwKHBvcHVwX2ZjNmU5MzhkMzY2NDRhMGU4NWYwNGJiYTJkNGYzODFiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JhMmViMzk0ZjFhMDRmNjk4YzljNjE5NTY5ZDFlMjU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTA5MzEwNiwtMC4xMzIyMDg1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxY2JmY2IxNGRkMDQ5Zjk5MGRhOGQxMGFmMWQxNDQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVhZTU4Mjk2NTRhMDRhNGY5YzBhYWFhZDE5MGEzYmY0ID0gJCgnPGRpdiBpZD0iaHRtbF81YWU1ODI5NjU0YTA0YTRmOWMwYWFhYWQxOTBhM2JmNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFOVE9OIFNUUkVFVCBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkxY2JmY2IxNGRkMDQ5Zjk5MGRhOGQxMGFmMWQxNDQ0LnNldENvbnRlbnQoaHRtbF81YWU1ODI5NjU0YTA0YTRmOWMwYWFhYWQxOTBhM2JmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYTJlYjM5NGYxYTA0ZjY5OGM5YzYxOTU2OWQxZTI1Ni5iaW5kUG9wdXAocG9wdXBfOTFjYmZjYjE0ZGQwNDlmOTkwZGE4ZDEwYWYxZDE0NDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTY3MTMwNGQzZTdkNGIwNjg2ZGFhYTgxZWM2OWU0YTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszOS43MzA5NDUxLC03NS43MDQwOTkxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlNTJlYzUzNzk3OTQ0MWJiNGFlYzFkYzQwZDFmNTZiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlkNzllOTM2YmI0ZTQ3MmE5MzIzZDhmNDNlNjJkNGE4ID0gJCgnPGRpdiBpZD0iaHRtbF85ZDc5ZTkzNmJiNGU0NzJhOTMyM2Q4ZjQzZTYyZDRhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSSyBDUkVTQ0VOVCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJlNTJlYzUzNzk3OTQ0MWJiNGFlYzFkYzQwZDFmNTZiLnNldENvbnRlbnQoaHRtbF85ZDc5ZTkzNmJiNGU0NzJhOTMyM2Q4ZjQzZTYyZDRhOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NjcxMzA0ZDNlN2Q0YjA2ODZkYWFhODFlYzY5ZTRhOS5iaW5kUG9wdXAocG9wdXBfMmU1MmVjNTM3OTc5NDQxYmI0YWVjMWRjNDBkMWY1NmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzE5MTQ5MTkwMjBhNDVhNmE0NDZkMTEyYTRmYjFmYjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMi44NzI2MTI4NDk5OTk5OTYsLTk2Ljc2NTI1ODY3NjgxNzY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YxMDA0ZDIyY2EyNjQ5M2I4MWNhNWUzMTNiZTkyMzhlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FiNTk2Mjk5MzdhZTQ4YjZiNTRiYTdkOGQzOGRjOTczID0gJCgnPGRpdiBpZD0iaHRtbF9hYjU5NjI5OTM3YWU0OGI2YjU0YmE3ZDhkMzhkYzk3MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSSyBMQU5FIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjEwMDRkMjJjYTI2NDkzYjgxY2E1ZTMxM2JlOTIzOGUuc2V0Q29udGVudChodG1sX2FiNTk2Mjk5MzdhZTQ4YjZiNTRiYTdkOGQzOGRjOTczKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMxOTE0OTE5MDIwYTQ1YTZhNDQ2ZDExMmE0ZmIxZmI2LmJpbmRQb3B1cChwb3B1cF9mMTAwNGQyMmNhMjY0OTNiODFjYTVlMzEzYmU5MjM4ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xOGIzNWQ3NzFlNzU0ZWIwYWQxZWE2MmYzNWU4MTdjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQwMjc3MTEsLTAuNDEzNzMzMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZGEwNjBjODY3YjQ0MGJjYWE4ZTA5NzgyMzRkMjZiMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81OWQ4NWZlMGU0MWU0NGNjODMwMzFjMGMyMjk1NDM1MyA9ICQoJzxkaXYgaWQ9Imh0bWxfNTlkODVmZTBlNDFlNDRjYzgzMDMxYzBjMjI5NTQzNTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBBUktFIFJPQUQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZGEwNjBjODY3YjQ0MGJjYWE4ZTA5NzgyMzRkMjZiMi5zZXRDb250ZW50KGh0bWxfNTlkODVmZTBlNDFlNDRjYzgzMDMxYzBjMjI5NTQzNTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMThiMzVkNzcxZTc1NGViMGFkMWVhNjJmMzVlODE3YzEuYmluZFBvcHVwKHBvcHVwXzZkYTA2MGM4NjdiNDQwYmNhYThlMDk3ODIzNGQyNmIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZiMzEwYmI5NmE5ODQ4ZjFiMjBhZmJlNTA0YWNhZDYxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuMzg5Njc0NywtNi45NDczMTEwMjcwMzczNDc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JiOGVhYWJlZTFkMjRhODhhMjI3ZDI2NjU2ZmNhNzJlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc5MTkwYmYzMDgwYjQ2NjNiNmQ1MWFjY2JlNWFmMDA0ID0gJCgnPGRpdiBpZD0iaHRtbF83OTE5MGJmMzA4MGI0NjYzYjZkNTFhY2NiZTVhZjAwNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEFSS0ZJRUxEUyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JiOGVhYWJlZTFkMjRhODhhMjI3ZDI2NjU2ZmNhNzJlLnNldENvbnRlbnQoaHRtbF83OTE5MGJmMzA4MGI0NjYzYjZkNTFhY2NiZTVhZjAwNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YjMxMGJiOTZhOTg0OGYxYjIwYWZiZTUwNGFjYWQ2MS5iaW5kUG9wdXAocG9wdXBfYmI4ZWFhYmVlMWQyNGE4OGEyMjdkMjY2NTZmY2E3MmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTA0NDYwOWY5NTk1NDI3ZDkzYzE5ZmFjNDgyZWZhZDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40NzU0ODA0LC0wLjE5NjMxNjddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2IwYzUxNjdmZGYwNGI1OThiY2JjOWFhMGVjMzZlZDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzQ1NzUxOGY3OGFjNDdlY2IzYWI0NjBjMzIxOGZjOTEgPSAkKCc8ZGl2IGlkPSJodG1sXzM0NTc1MThmNzhhYzQ3ZWNiM2FiNDYwYzMyMThmYzkxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QQVJUSEVOSUEgUk9BRCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdiMGM1MTY3ZmRmMDRiNTk4YmNiYzlhYTBlYzM2ZWQ5LnNldENvbnRlbnQoaHRtbF8zNDU3NTE4Zjc4YWM0N2VjYjNhYjQ2MGMzMjE4ZmM5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDQ0NjA5Zjk1OTU0MjdkOTNjMTlmYWM0ODJlZmFkNS5iaW5kUG9wdXAocG9wdXBfN2IwYzUxNjdmZGYwNGI1OThiY2JjOWFhMGVjMzZlZDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjljN2ZkYmIwMzk0NDBkZmI0ZGQ4ZWVmYmYwYjkyMjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4wODY1OTg0LDEuMTc2NDg4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ODRmZDhlMTI1OTU0MWZjYTFmMzg1YTQ4M2NhZDM5NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNmYxMGJkMTNiYTQ0NWVkYjIxZGU1YmYxODVhNTVkNSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDZmMTBiZDEzYmE0NDVlZGIyMWRlNWJmMTg1YTU1ZDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBBVklMSU9OIFJPQUQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ODRmZDhlMTI1OTU0MWZjYTFmMzg1YTQ4M2NhZDM5Ni5zZXRDb250ZW50KGh0bWxfZDZmMTBiZDEzYmE0NDVlZGIyMWRlNWJmMTg1YTU1ZDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjljN2ZkYmIwMzk0NDBkZmI0ZGQ4ZWVmYmYwYjkyMjMuYmluZFBvcHVwKHBvcHVwXzc4NGZkOGUxMjU5NTQxZmNhMWYzODVhNDgzY2FkMzk2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA5NWRjMGNjZDAyNjQ1ZjlhMTNkNTlmMWEzZmZiMjgyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTEyMzYxOSwtMC4xOTgyOTQ3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdlYWZjNWFkNTI1ZDRmMTZiYjAwNjg1OTMwZjk5ZTM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FmYWQ4ZmVkZmFhMzQ2NTM4YmYzNTNlNTliMjAxYmE1ID0gJCgnPGRpdiBpZD0iaHRtbF9hZmFkOGZlZGZhYTM0NjUzOGJmMzUzZTU5YjIwMWJhNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEVNQlJJREdFIE1FV1MgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZWFmYzVhZDUyNWQ0ZjE2YmIwMDY4NTkzMGY5OWUzOC5zZXRDb250ZW50KGh0bWxfYWZhZDhmZWRmYWEzNDY1MzhiZjM1M2U1OWIyMDFiYTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDk1ZGMwY2NkMDI2NDVmOWExM2Q1OWYxYTNmZmIyODIuYmluZFBvcHVwKHBvcHVwXzdlYWZjNWFkNTI1ZDRmMTZiYjAwNjg1OTMwZjk5ZTM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NiYmRmNjQwNTIwOTQxNjc5NTVkODlhNzUyNmY1NDZkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTA5MTQ4MywtMC4xOTcwODYzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4ZDUwOTZhNTNhMTRkNjRiOWQ3MDAzYTNkN2UxYjk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBlYzVmMGMyNTRiNzRjY2VhNDIyMmNlMDZhOTI5YzkxID0gJCgnPGRpdiBpZD0iaHRtbF8wZWM1ZjBjMjU0Yjc0Y2NlYTQyMjJjZTA2YTkyOWM5MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEVNQlJJREdFIFJPQUQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lOGQ1MDk2YTUzYTE0ZDY0YjlkNzAwM2EzZDdlMWI5Ny5zZXRDb250ZW50KGh0bWxfMGVjNWYwYzI1NGI3NGNjZWE0MjIyY2UwNmE5MjljOTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2JiZGY2NDA1MjA5NDE2Nzk1NWQ4OWE3NTI2ZjU0NmQuYmluZFBvcHVwKHBvcHVwX2U4ZDUwOTZhNTNhMTRkNjRiOWQ3MDAzYTNkN2UxYjk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJkMmQ5YThkNjcxNTQ0YzNhZDNjYzcxOGRmMWEzNmY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbMzIuMjkzMDkxMzUsLTY0Ljc3OTMxODY2NDQ0MTA5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JlYTk5OWFiMGQ1MTQ3ODVhNTcwNDYwNTM3Mzk5ZTM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxYzIwZmUzNGJmYjQwNzNiMzNjNDhlZWNjMzg2ZDBiID0gJCgnPGRpdiBpZD0iaHRtbF8zMWMyMGZlMzRiZmI0MDczYjMzYzQ4ZWVjYzM4NmQwYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEVNQlJPS0UgU1RVRElPUyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JlYTk5OWFiMGQ1MTQ3ODVhNTcwNDYwNTM3Mzk5ZTM1LnNldENvbnRlbnQoaHRtbF8zMWMyMGZlMzRiZmI0MDczYjMzYzQ4ZWVjYzM4NmQwYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZDJkOWE4ZDY3MTU0NGMzYWQzY2M3MThkZjFhMzZmOC5iaW5kUG9wdXAocG9wdXBfYmVhOTk5YWIwZDUxNDc4NWE1NzA0NjA1MzczOTllMzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODRhNGM4NTQ1ZmFjNDM3N2FmYThlN2JmMDI2ZWZhYzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTM1NDcsLTAuMjAwMzgxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZGY4ZmZmYjhmODc0Mzc5YmM1NmFlMWU2ZDAxOWFjZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNGYyYTJhNjRmYTA0ZjUzYTgxOTEyOTFjMWJjOGUxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTRmMmEyYTY0ZmEwNGY1M2E4MTkxMjkxYzFiYzhlMWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBFTkNPTUJFIE1FV1MgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZGY4ZmZmYjhmODc0Mzc5YmM1NmFlMWU2ZDAxOWFjZi5zZXRDb250ZW50KGh0bWxfMTRmMmEyYTY0ZmEwNGY1M2E4MTkxMjkxYzFiYzhlMWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODRhNGM4NTQ1ZmFjNDM3N2FmYThlN2JmMDI2ZWZhYzQuYmluZFBvcHVwKHBvcHVwXzBkZjhmZmZiOGY4NzQzNzliYzU2YWUxZTZkMDE5YWNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IxOWY0ZTQ5MzA5ODQyNWZiZWUyMzgxODZlYjY3ZTk0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNDYwNzEyOCwtMS45MzYwNzQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRkODk4Y2MwZjZjNDRkYTliNGQyY2Q2NDIyMzNiNjAyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJlYzE1YjRiZDI1YzQzNWJhYzZkYjc0MzA2ZDI4ZjVlID0gJCgnPGRpdiBpZD0iaHRtbF8yZWMxNWI0YmQyNWM0MzViYWM2ZGI3NDMwNmQyOGY1ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEVURVJTSEFNIFBMQUNFIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGQ4OThjYzBmNmM0NGRhOWI0ZDJjZDY0MjIzM2I2MDIuc2V0Q29udGVudChodG1sXzJlYzE1YjRiZDI1YzQzNWJhYzZkYjc0MzA2ZDI4ZjVlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2IxOWY0ZTQ5MzA5ODQyNWZiZWUyMzgxODZlYjY3ZTk0LmJpbmRQb3B1cChwb3B1cF80ZDg5OGNjMGY2YzQ0ZGE5YjRkMmNkNjQyMjMzYjYwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZDU5ZDcyNzhlNTE0YWYxOTQ1MzMxNDU0MGY0NTI1YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWy0zMi4wNTQ4NDM1MDAwMDAwMDQsMTE1Ljc0Mjc5NjMwMjA4NTk0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc3YmVmZTcxZjM4MzQxNjZhMzYyMTBjMDk3OTU0NDU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZiNDFlZDNlN2E2MTRkNmU5NmE5MjhmM2JhODNlNThkID0gJCgnPGRpdiBpZD0iaHRtbF82YjQxZWQzZTdhNjE0ZDZlOTZhOTI4ZjNiYTgzZTU4ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UEhJTExJTU9SRSBHQVJERU5TIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzdiZWZlNzFmMzgzNDE2NmEzNjIxMGMwOTc5NTQ0NTQuc2V0Q29udGVudChodG1sXzZiNDFlZDNlN2E2MTRkNmU5NmE5MjhmM2JhODNlNThkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJkNTlkNzI3OGU1MTRhZjE5NDUzMzE0NTQwZjQ1MjVhLmJpbmRQb3B1cChwb3B1cF83N2JlZmU3MWYzODM0MTY2YTM2MjEwYzA5Nzk1NDQ1NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lODMxNzMzMzE3MDE0MzAxYmY4MmQ4OTg4YjIxOTQ4MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ4NTU5MjgsLTAuMTYxOTg0Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MjQ5OWNiMmQ4YjI0MjU3ODQxOWIzMWE5MmU4ZDM1OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wYThhZDI2ZGY0NjM0M2I0OGYzMDhhYTBmMDU3ZTAwMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMGE4YWQyNmRmNDYzNDNiNDhmMzA4YWEwZjA1N2UwMDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBIWVNJQyBQTEFDRSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcyNDk5Y2IyZDhiMjQyNTc4NDE5YjMxYTkyZThkMzU5LnNldENvbnRlbnQoaHRtbF8wYThhZDI2ZGY0NjM0M2I0OGYzMDhhYTBmMDU3ZTAwMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lODMxNzMzMzE3MDE0MzAxYmY4MmQ4OTg4YjIxOTQ4MS5iaW5kUG9wdXAocG9wdXBfNzI0OTljYjJkOGIyNDI1Nzg0MTliMzFhOTJlOGQzNTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjYyZGNmOWU1NWI5NDQ0MDk2NDE3MTkzODZmN2EyY2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjkyODc3LC0wLjA4MzQ1MjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWUzN2Q3MWM5NzQzNDhhNWFjODE3OGNjNmQ0YmUxMjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDE5YWUyODBjNzI1NDliMDg3YWQxZDE3Yjc1MzJlZmEgPSAkKCc8ZGl2IGlkPSJodG1sXzQxOWFlMjgwYzcyNTQ5YjA4N2FkMWQxN2I3NTMyZWZhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QSVRGSUVMRCBTVFJFRVQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lZTM3ZDcxYzk3NDM0OGE1YWM4MTc4Y2M2ZDRiZTEyMi5zZXRDb250ZW50KGh0bWxfNDE5YWUyODBjNzI1NDliMDg3YWQxZDE3Yjc1MzJlZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjYyZGNmOWU1NWI5NDQ0MDk2NDE3MTkzODZmN2EyY2EuYmluZFBvcHVwKHBvcHVwX2VlMzdkNzFjOTc0MzQ4YTVhYzgxNzhjYzZkNGJlMTIyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RjYWE4MGZjMzVhNzQ5NmRiYjhkODc2ZjU0NDkzNmJhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNzgyNDgzLC00LjcwMzk2NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNThhNzIwMDc4Y2M4NDkxZDk3OTYzZGJiMzdmYTZjZDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTAwNDM2NTE0Mjc5NDQ3Zjg4YzExZTY3ZDFmNTFhMzIgPSAkKCc8ZGl2IGlkPSJodG1sX2UwMDQzNjUxNDI3OTQ0N2Y4OGMxMWU2N2QxZjUxYTMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QUklOQ0VTIEdBVEUgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OGE3MjAwNzhjYzg0OTFkOTc5NjNkYmIzN2ZhNmNkMC5zZXRDb250ZW50KGh0bWxfZTAwNDM2NTE0Mjc5NDQ3Zjg4YzExZTY3ZDFmNTFhMzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGNhYTgwZmMzNWE3NDk2ZGJiOGQ4NzZmNTQ0OTM2YmEuYmluZFBvcHVwKHBvcHVwXzU4YTcyMDA3OGNjODQ5MWQ5Nzk2M2RiYjM3ZmE2Y2QwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFkZWJiZGY3NWVlNTRhZGU5NGM3NDVkOTEzN2YzZmE0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTMuMjY3OTQ3NTUsLTkuMDU1NTk3MzIyMTcyODRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODk4OWQ3ZDMzY2E3NGQ2MzhjMDBlMzA4ZWQ2MmRlM2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDc1NjIxMzQyZDQ4NDI1ZGIxMzAyNjY5NTNiN2QxNjkgPSAkKCc8ZGl2IGlkPSJodG1sXzA3NTYyMTM0MmQ0ODQyNWRiMTMwMjY2OTUzYjdkMTY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QUklPUlkgUk9BRCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg5ODlkN2QzM2NhNzRkNjM4YzAwZTMwOGVkNjJkZTNhLnNldENvbnRlbnQoaHRtbF8wNzU2MjEzNDJkNDg0MjVkYjEzMDI2Njk1M2I3ZDE2OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZGViYmRmNzVlZTU0YWRlOTRjNzQ1ZDkxMzdmM2ZhNC5iaW5kUG9wdXAocG9wdXBfODk4OWQ3ZDMzY2E3NGQ2MzhjMDBlMzA4ZWQ2MmRlM2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2IyMjk3OGNkYjNhNDI2OWE5OTAyMWQwODMyZmM5NDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41ODU3MzEsLTAuMjI4MDYxNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZTE2NmRjMTViMWQ0YThhYmEyZmUxZGVmZTM0OGE2NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZDI2MWNmYjcxYjc0MTcyOTBkNGZjMTcxY2VkYzMwMyA9ICQoJzxkaXYgaWQ9Imh0bWxfM2QyNjFjZmI3MWI3NDE3MjkwZDRmYzE3MWNlZGMzMDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBST1RIRVJPIEdBUkRFTlMgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mZTE2NmRjMTViMWQ0YThhYmEyZmUxZGVmZTM0OGE2Ny5zZXRDb250ZW50KGh0bWxfM2QyNjFjZmI3MWI3NDE3MjkwZDRmYzE3MWNlZGMzMDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2IyMjk3OGNkYjNhNDI2OWE5OTAyMWQwODMyZmM5NDAuYmluZFBvcHVwKHBvcHVwX2ZlMTY2ZGMxNWIxZDRhOGFiYTJmZTFkZWZlMzQ4YTY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc4ZmQ5OTIwNGY5MTQyMmM4MzNkYTFjNzQwMGQzNTU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDYwOTg2OSwtMC4yMTcwNjI0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZlZTdiMGY2ZDY3ZTRlZmQ5Y2JhNTA3MDA2YzM3OTM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU2MmY1NWMwM2EyNTQ1NDE5Njk0NmIxYjU0N2NmMDcwID0gJCgnPGRpdiBpZD0iaHRtbF81NjJmNTVjMDNhMjU0NTQxOTY5NDZiMWI1NDdjZjA3MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UFVUTkVZIEhJR0ggU1RSRUVUIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmVlN2IwZjZkNjdlNGVmZDljYmE1MDcwMDZjMzc5MzQuc2V0Q29udGVudChodG1sXzU2MmY1NWMwM2EyNTQ1NDE5Njk0NmIxYjU0N2NmMDcwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc4ZmQ5OTIwNGY5MTQyMmM4MzNkYTFjNzQwMGQzNTU5LmJpbmRQb3B1cChwb3B1cF82ZWU3YjBmNmQ2N2U0ZWZkOWNiYTUwNzAwNmMzNzkzNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNTkyYTcyOWJlMDc0ZTAxOTMxYzdjNjQzMDMzNWE5NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ3MzExMjcsLTAuMTk1ODY3NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZGNkODE0ZGY1NDU0ZjUxOGVjNTMwZWY0ODE0ODIwNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wNzkzMjFmMWE0OGQ0NmMwYWVkMWE5ZjAyMzg1ZGI1YiA9ICQoJzxkaXYgaWQ9Imh0bWxfMDc5MzIxZjFhNDhkNDZjMGFlZDFhOWYwMjM4NWRiNWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlFVQVJSRU5ET04gU1RSRUVUIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmRjZDgxNGRmNTQ1NGY1MThlYzUzMGVmNDgxNDgyMDYuc2V0Q29udGVudChodG1sXzA3OTMyMWYxYTQ4ZDQ2YzBhZWQxYTlmMDIzODVkYjViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y1OTJhNzI5YmUwNzRlMDE5MzFjN2M2NDMwMzM1YTk0LmJpbmRQb3B1cChwb3B1cF9iZGNkODE0ZGY1NDU0ZjUxOGVjNTMwZWY0ODE0ODIwNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YTExODZkZGE0NzY0ZDFhYTY5M2E5YzhkZTIwMmQ3NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzM2Ljk1Mjk2NjIsLTc2LjUyNjcyNDddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjliZmUzY2IxNjcxNDQzMWEzNjA3NDg2ODFkMmQ5YWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWZjYWZkNmZlNTA0NDk3ZGI1ZGUzMmNkMzZkZmY0MjEgPSAkKCc8ZGl2IGlkPSJodG1sXzVmY2FmZDZmZTUwNDQ5N2RiNWRlMzJjZDM2ZGZmNDIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5RVUVFTlMgR0FURSBURVJSQUNFIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjliZmUzY2IxNjcxNDQzMWEzNjA3NDg2ODFkMmQ5YWIuc2V0Q29udGVudChodG1sXzVmY2FmZDZmZTUwNDQ5N2RiNWRlMzJjZDM2ZGZmNDIxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlhMTE4NmRkYTQ3NjRkMWFhNjkzYTljOGRlMjAyZDc1LmJpbmRQb3B1cChwb3B1cF9mOWJmZTNjYjE2NzE0NDMxYTM2MDc0ODY4MWQyZDlhYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYmFlODQzMjI2MDE0MDdhOTg4ZjY2MDc0ZGNhMTY5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ3OTA5MDYsLTAuMTY5NDU0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwZDBkOGIzMDA3OTRiNjA5NDkwNDc0NjE1ZjYyMTgzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBjYmUyNzk3MzEwOTRkMWE5OTIzY2FlNTE2NjkzMTViID0gJCgnPGRpdiBpZD0iaHRtbF8wY2JlMjc5NzMxMDk0ZDFhOTkyM2NhZTUxNjY5MzE1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkFEU1RPQ0sgU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDBkMGQ4YjMwMDc5NGI2MDk0OTA0NzQ2MTVmNjIxODMuc2V0Q29udGVudChodG1sXzBjYmUyNzk3MzEwOTRkMWE5OTIzY2FlNTE2NjkzMTViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNiYWU4NDMyMjYwMTQwN2E5ODhmNjYwNzRkY2ExNjlkLmJpbmRQb3B1cChwb3B1cF9kMGQwZDhiMzAwNzk0YjYwOTQ5MDQ3NDYxNWY2MjE4Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MzdjYmRmY2Y4ZTY0ZmVjYTgwNzhlYTFiNTMwYjFhOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ2ODg5MjgsLTAuMjA2NDgyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMxMzk1ZTgxNGMyZDRlNDA4ZmExMGQ5NmVmNTE4M2MyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FmNWQ2YzFjZTgzNjQ2NTY4YjA3MGJmZmQxODE1ODc2ID0gJCgnPGRpdiBpZD0iaHRtbF9hZjVkNmMxY2U4MzY0NjU2OGIwNzBiZmZkMTgxNTg3NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkFORUxBR0ggQVZFTlVFIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzEzOTVlODE0YzJkNGU0MDhmYTEwZDk2ZWY1MTgzYzIuc2V0Q29udGVudChodG1sX2FmNWQ2YzFjZTgzNjQ2NTY4YjA3MGJmZmQxODE1ODc2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzN2NiZGZjZjhlNjRmZWNhODA3OGVhMWI1MzBiMWE4LmJpbmRQb3B1cChwb3B1cF8zMTM5NWU4MTRjMmQ0ZTQwOGZhMTBkOTZlZjUxODNjMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNTA2NWU1MTNmYWI0NTQ2OWYxNzFjZDkxZWYxNDI5NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjcyNzE3NDMsMC40NjY4MDQ1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QzMzE3NjZkZTcxNTQ3ZmU5YTNkZjA5OTlkYWRjZGNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVkY2JmNjFkNjFjODQzZWU4YTlkMWJjYzAwODNiNDM3ID0gJCgnPGRpdiBpZD0iaHRtbF81ZGNiZjYxZDYxYzg0M2VlOGE5ZDFiY2MwMDgzYjQzNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkVEQ0xJRkZFIFJPQUQgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMzMxNzY2ZGU3MTU0N2ZlOWEzZGYwOTk5ZGFkY2RjYi5zZXRDb250ZW50KGh0bWxfNWRjYmY2MWQ2MWM4NDNlZThhOWQxYmNjMDA4M2I0MzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjUwNjVlNTEzZmFiNDU0NjlmMTcxY2Q5MWVmMTQyOTUuYmluZFBvcHVwKHBvcHVwX2QzMzE3NjZkZTcxNTQ3ZmU5YTNkZjA5OTlkYWRjZGNiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNiNmU2Y2Y1ODAzMTQwZWU5MzJkMDZmYTUzZDE2ZWMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTA5NzA3NywtMC4xNTQwOTg2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU3MGY3ZWQzYmFjYTRhNWNiMDlkNDY1NDBmMTdmZDk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEwMDFmYzliOWE4NjQxOTU4MjYyOTlkYzc0NGJkZWUzID0gJCgnPGRpdiBpZD0iaHRtbF8xMDAxZmM5YjlhODY0MTk1ODI2Mjk5ZGM3NDRiZGVlMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkVFVkVTIE1FV1MgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NzBmN2VkM2JhY2E0YTVjYjA5ZDQ2NTQwZjE3ZmQ5Ny5zZXRDb250ZW50KGh0bWxfMTAwMWZjOWI5YTg2NDE5NTgyNjI5OWRjNzQ0YmRlZTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2I2ZTZjZjU4MDMxNDBlZTkzMmQwNmZhNTNkMTZlYzAuYmluZFBvcHVwKHBvcHVwXzU3MGY3ZWQzYmFjYTRhNWNiMDlkNDY1NDBmMTdmZDk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzliNDdiZDA2ODgwYTRmZjliZjE2Yzk4ZTY5MDdkMjg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTM0OTk2NSwtMC4wOTgxMzg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NlNGYxZjI1ZWFkNTQxNWI5MTYzNzBjZTQ0YmRiMTFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjZGQ2NzVhMDE3YjRjZGJhMTdkMzE1MzNhNGU1OGU0ID0gJCgnPGRpdiBpZD0iaHRtbF83Y2RkNjc1YTAxN2I0Y2RiYTE3ZDMxNTMzYTRlNThlNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UkhFSURPTCBNRVdTIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2U0ZjFmMjVlYWQ1NDE1YjkxNjM3MGNlNDRiZGIxMWQuc2V0Q29udGVudChodG1sXzdjZGQ2NzVhMDE3YjRjZGJhMTdkMzE1MzNhNGU1OGU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzliNDdiZDA2ODgwYTRmZjliZjE2Yzk4ZTY5MDdkMjg2LmJpbmRQb3B1cChwb3B1cF9jZTRmMWYyNWVhZDU0MTViOTE2MzcwY2U0NGJkYjExZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNjFhNmVkNTIxMzI0YjM2YWRkNGM0MmE2ZmQ2MWY2NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU5MzY5OTIsLTAuMTU1NTgxN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMzBjMjM1MDQ1MjY0YWIyOTc5MjQzZGNlMTJlOWYzMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xZDhmZGMwODA4YzA0MjU4YjRhMGE4ZjMyN2ZjZTQ3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMWQ4ZmRjMDgwOGMwNDI1OGI0YTBhOGYzMjdmY2U0N2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJJTkdXT09EIEFWRU5VRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAzMGMyMzUwNDUyNjRhYjI5NzkyNDNkY2UxMmU5ZjMyLnNldENvbnRlbnQoaHRtbF8xZDhmZGMwODA4YzA0MjU4YjRhMGE4ZjMyN2ZjZTQ3YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNjFhNmVkNTIxMzI0YjM2YWRkNGM0MmE2ZmQ2MWY2NS5iaW5kUG9wdXAocG9wdXBfMDMwYzIzNTA0NTI2NGFiMjk3OTI0M2RjZTEyZTlmMzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzMxYjg3NWY2ZGRkNDVjMmJlNzIyZTA0NThjNTdmODAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi40NTc5MTcsLTEuODY1MTI4NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZjRhZWVmMGE4Yjg0NjRiODE4ZDQxYjJiOGMzNzFmZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NTY5NmJiOWIzNTk0YTAzYjU0ODcyYjgxZmIzOWI5NyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDU2OTZiYjliMzU5NGEwM2I1NDg3MmI4MWZiMzliOTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJPREVSSUNLIFJPQUQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZjRhZWVmMGE4Yjg0NjRiODE4ZDQxYjJiOGMzNzFmZC5zZXRDb250ZW50KGh0bWxfNDU2OTZiYjliMzU5NGEwM2I1NDg3MmI4MWZiMzliOTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzMxYjg3NWY2ZGRkNDVjMmJlNzIyZTA0NThjNTdmODAuYmluZFBvcHVwKHBvcHVwXzVmNGFlZWYwYThiODQ2NGI4MThkNDFiMmI4YzM3MWZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M0Njc3MDk0ZGNmMjRiZTQ4OGEyMzM4ZDAzZmVlZjlmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTA5ODQzMjUsLTAuMDMyODA2NTQ1MTA0Mzc1MDRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTUyMzEyNGNmYTcyNGMxNTg3YWZiNDM4ZDAwY2ZlNmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTY2YmExNTRlYjI0NDRlNDlmZTVhZTk3YTk0MzRhNTcgPSAkKCc8ZGl2IGlkPSJodG1sXzU2NmJhMTU0ZWIyNDQ0ZTQ5ZmU1YWU5N2E5NDM0YTU3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ST1BFTUFLRVJTIEZJRUxEUyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE1MjMxMjRjZmE3MjRjMTU4N2FmYjQzOGQwMGNmZTZlLnNldENvbnRlbnQoaHRtbF81NjZiYTE1NGViMjQ0NGU0OWZlNWFlOTdhOTQzNGE1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNDY3NzA5NGRjZjI0YmU0ODhhMjMzOGQwM2ZlZWY5Zi5iaW5kUG9wdXAocG9wdXBfMTUyMzEyNGNmYTcyNGMxNTg3YWZiNDM4ZDAwY2ZlNmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmU4NDJhZTkzODc4NGJmYmI0ZTAzNGQ0NThjMjU0NTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4zODcyOTU3LC0yLjM2ODQzOTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmY1OTVlNGJjNjU3NDNhMGE0ZjNiMGMxYTJmNDE2NzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWVkN2IxZDZlZjliNGM1ODhiNzI2OTAwZThlMTI4ZjAgPSAkKCc8ZGl2IGlkPSJodG1sX2FlZDdiMWQ2ZWY5YjRjNTg4YjcyNjkwMGU4ZTEyOGYwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ST1lBTCBDUkVTQ0VOVCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJmNTk1ZTRiYzY1NzQzYTBhNGYzYjBjMWEyZjQxNjcyLnNldENvbnRlbnQoaHRtbF9hZWQ3YjFkNmVmOWI0YzU4OGI3MjY5MDBlOGUxMjhmMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZTg0MmFlOTM4Nzg0YmZiYjRlMDM0ZDQ1OGMyNTQ1OC5iaW5kUG9wdXAocG9wdXBfMmY1OTVlNGJjNjU3NDNhMGE0ZjNiMGMxYTJmNDE2NzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjU0YmIxYTY1MDgzNDYxNjg5YTQ2N2YwZDY0NmE0YjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MC41MzcwMzkzLC0zLjk1MjU2NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTg3YTM5YmRmZGFhNDQ3ZGJmMjY1YTdkOGNiMzg2MDkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTg1NmE5MzA1ZTFmNDg0NGFjODc5NTdhZDk4MGI5OGQgPSAkKCc8ZGl2IGlkPSJodG1sX2U4NTZhOTMwNWUxZjQ4NDRhYzg3OTU3YWQ5ODBiOThkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5ST1lBTCBISUxMIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTg3YTM5YmRmZGFhNDQ3ZGJmMjY1YTdkOGNiMzg2MDkuc2V0Q29udGVudChodG1sX2U4NTZhOTMwNWUxZjQ4NDRhYzg3OTU3YWQ5ODBiOThkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI1NGJiMWE2NTA4MzQ2MTY4OWE0NjdmMGQ2NDZhNGI4LmJpbmRQb3B1cChwb3B1cF9lODdhMzliZGZkYWE0NDdkYmYyNjVhN2Q4Y2IzODYwOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZjVmMTg3MjgyMDA0NTZmODhjNWViMTRmNGE2ZjFhNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5OTg2ODksLTAuMjExODkwNF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZDc5Yjc1MWExMmY0NTM2YTI5ZmQwZWQ4Zjg1OWMxYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMDNmZGQ4NmZjMTQ0OWU1OTFiOThhOWZkMThhMzQ3ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTAzZmRkODZmYzE0NDllNTkxYjk4YTlmZDE4YTM0N2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJVU1NFTEwgR0FSREVOUyBNRVdTIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGQ3OWI3NTFhMTJmNDUzNmEyOWZkMGVkOGY4NTljMWEuc2V0Q29udGVudChodG1sX2EwM2ZkZDg2ZmMxNDQ5ZTU5MWI5OGE5ZmQxOGEzNDdkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhmNWYxODcyODIwMDQ1NmY4OGM1ZWIxNGY0YTZmMWE2LmJpbmRQb3B1cChwb3B1cF80ZDc5Yjc1MWExMmY0NTM2YTI5ZmQwZWQ4Zjg1OWMxYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNjdiOGU0MDA2OWM0ZWFhYjlkYmEzZDg5Zjc1Y2RhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUxNTMyMzcsLTAuMDY0MzI2Nl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85N2M2N2IyZGY1MGE0ZTRhOGNmNjRmZmEwMWQ1NDUyMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMDQxYjg1NmQ1NzM0YTkyYjg5ZDRiMzVjZDMzODI4NCA9ICQoJzxkaXYgaWQ9Imh0bWxfZTA0MWI4NTZkNTczNGE5MmI4OWQ0YjM1Y2QzMzgyODQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNFVFRMRVMgU1RSRUVUIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTdjNjdiMmRmNTBhNGU0YThjZjY0ZmZhMDFkNTQ1MjIuc2V0Q29udGVudChodG1sX2UwNDFiODU2ZDU3MzRhOTJiODlkNGIzNWNkMzM4Mjg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y2N2I4ZTQwMDY5YzRlYWFiOWRiYTNkODlmNzVjZGExLmJpbmRQb3B1cChwb3B1cF85N2M2N2IyZGY1MGE0ZTRhOGNmNjRmZmEwMWQ1NDUyMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mZjBmMDFhMTc0ZTI0NTRmYjRjNjQ0Nzk1ZTBmMGViMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU5MjYxMywwLjA3MzE0NDldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDUyOTE4YmVhOGI0NDMyYmE3NDQ2YTlmOGRkNTIzOTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWY0N2UxNzFiMzBiNDNlZjllMzkzNzg5Y2Y3MDkzYjYgPSAkKCc8ZGl2IGlkPSJodG1sX2VmNDdlMTcxYjMwYjQzZWY5ZTM5Mzc4OWNmNzA5M2I2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TSEVMRE9OIEFWRU5VRSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ1MjkxOGJlYThiNDQzMmJhNzQ0NmE5ZjhkZDUyMzkzLnNldENvbnRlbnQoaHRtbF9lZjQ3ZTE3MWIzMGI0M2VmOWUzOTM3ODljZjcwOTNiNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZjBmMDFhMTc0ZTI0NTRmYjRjNjQ0Nzk1ZTBmMGViMy5iaW5kUG9wdXAocG9wdXBfNDUyOTE4YmVhOGI0NDMyYmE3NDQ2YTlmOGRkNTIzOTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzE4OTA1ODc1ZDA3NDRkY2I3Y2VmNjg1OTBmYTRkNWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40OTg3NDYzLC0wLjE4OTA3ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDAxZDdhOTU5ZDRkNDc3Njk1Y2JkYTkzOGU4YjA2NjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODJlNjdiOGFkYTBkNDI2Mzg4OGFlOGQ4ZTk0ZjBjZWEgPSAkKCc8ZGl2IGlkPSJodG1sXzgyZTY3YjhhZGEwZDQyNjM4ODhhZThkOGU5NGYwY2VhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TT1VUSCBFTkQgUk9XIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDAxZDdhOTU5ZDRkNDc3Njk1Y2JkYTkzOGU4YjA2NjYuc2V0Q29udGVudChodG1sXzgyZTY3YjhhZGEwZDQyNjM4ODhhZThkOGU5NGYwY2VhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMxODkwNTg3NWQwNzQ0ZGNiN2NlZjY4NTkwZmE0ZDVhLmJpbmRQb3B1cChwb3B1cF9kMDFkN2E5NTlkNGQ0Nzc2OTVjYmRhOTM4ZThiMDY2Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZjljM2I3NmI3OWM0YWNiYTM1NmU2NWM5MmJjZGYyNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU3NDYyNjcsLTAuMTQ2MjM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk4NzI3ZmYwNTI1NzQ5MTE4NjhmYjdmNzljNmY4YWQ2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzEyMmI3YzNhZTk3YzQ2ZGFiMzgyZWNkOTk4MTY3MzI5ID0gJCgnPGRpdiBpZD0iaHRtbF8xMjJiN2MzYWU5N2M0NmRhYjM4MmVjZDk5ODE2NzMyOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U09VVEhXT09EIExBV04gUk9BRCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk4NzI3ZmYwNTI1NzQ5MTE4NjhmYjdmNzljNmY4YWQ2LnNldENvbnRlbnQoaHRtbF8xMjJiN2MzYWU5N2M0NmRhYjM4MmVjZDk5ODE2NzMyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZjljM2I3NmI3OWM0YWNiYTM1NmU2NWM5MmJjZGYyNy5iaW5kUG9wdXAocG9wdXBfOTg3MjdmZjA1MjU3NDkxMTg2OGZiN2Y3OWM2ZjhhZDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmRmMDIwOTMzYmY5NDE5NmE5NTJlODEzY2IyZmY3YTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi42ODgwOTQxLC0yLjcyNDU2NzI1MTE0Mzg0NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNDExNDY1YWExMzU0ZjYxODM0ZDI1MjI3N2JhZGU3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MzI0OTg3MzJhMDg0ZjAwYWVhOGVmNTg5NTU5YzQ0ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzMyNDk4NzMyYTA4NGYwMGFlYThlZjU4OTU1OWM0NGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNPVkVSRUlHTiBQQVJLIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTQxMTQ2NWFhMTM1NGY2MTgzNGQyNTIyNzdiYWRlNzcuc2V0Q29udGVudChodG1sXzczMjQ5ODczMmEwODRmMDBhZWE4ZWY1ODk1NTljNDRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JkZjAyMDkzM2JmOTQxOTZhOTUyZTgxM2NiMmZmN2ExLmJpbmRQb3B1cChwb3B1cF9hNDExNDY1YWExMzU0ZjYxODM0ZDI1MjI3N2JhZGU3Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYTU5NjhjZTVlNzk0NmQ5YTJmMTA2ZDlhMzAwNTE1NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzE3LjA5NzUxMzUsLTg4LjYxNjExNTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDM4MTg4YzI0ZDFiNGI2ZmExN2VlZDFhZWEzNmZlOTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTUxZmIyYzNjM2ZlNDllMjlmZTZjNTgxZGY1OTljYmUgPSAkKCc8ZGl2IGlkPSJodG1sXzk1MWZiMmMzYzNmZTQ5ZTI5ZmU2YzU4MWRmNTk5Y2JlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TVCBNQVJHQVJFVFMgQ1JFU0NFTlQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wMzgxODhjMjRkMWI0YjZmYTE3ZWVkMWFlYTM2ZmU5Mi5zZXRDb250ZW50KGh0bWxfOTUxZmIyYzNjM2ZlNDllMjlmZTZjNTgxZGY1OTljYmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2E1OTY4Y2U1ZTc5NDZkOWEyZjEwNmQ5YTMwMDUxNTYuYmluZFBvcHVwKHBvcHVwXzAzODE4OGMyNGQxYjRiNmZhMTdlZWQxYWVhMzZmZTkyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUwOGVhZWY2ZGUyZjQ2MGE4YmZlN2ZkMWFhNTBiNWQzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNDg3MjA3MSwtMC4xMTg1MzQxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y4NWY4NmQyMTgxZjQ3NjQ5NjZmZjAxODdjNDE5MjZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk0NjliOWI2ZDdhOTRlMzRiZWEzOWZlNWJhMmYxNzg4ID0gJCgnPGRpdiBpZD0iaHRtbF85NDY5YjliNmQ3YTk0ZTM0YmVhMzlmZTViYTJmMTc4OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U1QgT1NXQUxEUyBQTEFDRSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y4NWY4NmQyMTgxZjQ3NjQ5NjZmZjAxODdjNDE5MjZmLnNldENvbnRlbnQoaHRtbF85NDY5YjliNmQ3YTk0ZTM0YmVhMzlmZTViYTJmMTc4OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MDhlYWVmNmRlMmY0NjBhOGJmZTdmZDFhYTUwYjVkMy5iaW5kUG9wdXAocG9wdXBfZjg1Zjg2ZDIxODFmNDc2NDk2NmZmMDE4N2M0MTkyNmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzMyZDQyODUxYWVkNDE1NTljMGI4YTkzYTE1NjExMDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MS45MDIyMzUzLDEyLjQ1NzM1NzMxMDI5ODAwOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODAwMGZmIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwMDBmZiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mYjQ3ZjE5OGMyMzI0Mzg3OTgwMThhZWQ3Yjk3NzY1YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jYzVlZTMzYjBhOGU0NzliOWI2OWI2NzIzMjM4YzlmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfY2M1ZWUzM2IwYThlNDc5YjliNjliNjcyMzIzOGM5ZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNUIFBFVEVSUyBTUVVBUkUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYjQ3ZjE5OGMyMzI0Mzg3OTgwMThhZWQ3Yjk3NzY1YS5zZXRDb250ZW50KGh0bWxfY2M1ZWUzM2IwYThlNDc5YjliNjliNjcyMzIzOGM5ZmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzMyZDQyODUxYWVkNDE1NTljMGI4YTkzYTE1NjExMDEuYmluZFBvcHVwKHBvcHVwX2ZiNDdmMTk4YzIzMjQzODc5ODAxOGFlZDdiOTc3NjVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzMTc5MzBiNDQyMzQ4ZTU4MjhjZWNkMzQzZGFmZDI4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTAwOTM3OSwtMC4xOTYwNDkyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzllMjBjOWRmNDc3MTQxYTM5Y2Q0YmFkYTkyZDE2NjUwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlMTc4YmI4YmE3MDRmNDViOGUwZmI5ZGVkMTAwNjA4ID0gJCgnPGRpdiBpZD0iaHRtbF83ZTE3OGJiOGJhNzA0ZjQ1YjhlMGZiOWRlZDEwMDYwOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U1RBRkZPUkQgVEVSUkFDRSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllMjBjOWRmNDc3MTQxYTM5Y2Q0YmFkYTkyZDE2NjUwLnNldENvbnRlbnQoaHRtbF83ZTE3OGJiOGJhNzA0ZjQ1YjhlMGZiOWRlZDEwMDYwOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MzE3OTMwYjQ0MjM0OGU1ODI4Y2VjZDM0M2RhZmQyOC5iaW5kUG9wdXAocG9wdXBfOWUyMGM5ZGY0NzcxNDFhMzljZDRiYWRhOTJkMTY2NTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTZlMWU4ZDdhOWM5NDAzMjk1MmY0NDVmYTM5MWFlNWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MTY2MTY2LC0wLjE5NzI3NjJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDE1N2ZhMjYxYjU2NGJjMDkxMTM3MzE0MzAxZWViY2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzQ4ZjhmMDU2MmI2NGU1MTkxYTAyNTg0YzEzYWQ5OWMgPSAkKCc8ZGl2IGlkPSJodG1sXzc0OGY4ZjA1NjJiNjRlNTE5MWEwMjU4NGMxM2FkOTljIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TVVRIRVJMQU5EIFBMQUNFIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDE1N2ZhMjYxYjU2NGJjMDkxMTM3MzE0MzAxZWViY2Quc2V0Q29udGVudChodG1sXzc0OGY4ZjA1NjJiNjRlNTE5MWEwMjU4NGMxM2FkOTljKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE2ZTFlOGQ3YTljOTQwMzI5NTJmNDQ1ZmEzOTFhZTVjLmJpbmRQb3B1cChwb3B1cF9kMTU3ZmEyNjFiNTY0YmMwOTExMzczMTQzMDFlZWJjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NWM5N2Q1YzFlZDk0NDI3OTJmMTVlMTJmMTA0MzRhMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjgwNzA4MiwxLjAyMzk2MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmRiYTJiODdhODRiNDk1NzgzYThjZWE1ZDk1ZjUyNzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODZmNTE0ZDk2OWU0NDBjYzljMjMxODAzNjM1MTE2MWIgPSAkKCc8ZGl2IGlkPSJodG1sXzg2ZjUxNGQ5NjllNDQwY2M5YzIzMTgwMzYzNTExNjFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TWURORVkgU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmRiYTJiODdhODRiNDk1NzgzYThjZWE1ZDk1ZjUyNzEuc2V0Q29udGVudChodG1sXzg2ZjUxNGQ5NjllNDQwY2M5YzIzMTgwMzYzNTExNjFiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY1Yzk3ZDVjMWVkOTQ0Mjc5MmYxNWUxMmYxMDQzNGEwLmJpbmRQb3B1cChwb3B1cF82ZGJhMmI4N2E4NGI0OTU3ODNhOGNlYTVkOTVmNTI3MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNjlmZmQ0MTljNzM0NWI3YTM0MDNkZjZiODkwMTAyNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjA4NTY2ODksLTAuMjQzMjc2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMjBjNDQ1ZjVhYWY0NmIzYjJmNWJhODYwOTI1MGQzMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83ZDNjMGZlMzQ1ZDQ0ZTc3ODJiNzViNjhiM2U0OWRkYiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2QzYzBmZTM0NWQ0NGU3NzgyYjc1YjY4YjNlNDlkZGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRIQU1FUyBCQU5LIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTIwYzQ0NWY1YWFmNDZiM2IyZjViYTg2MDkyNTBkMzEuc2V0Q29udGVudChodG1sXzdkM2MwZmUzNDVkNDRlNzc4MmI3NWI2OGIzZTQ5ZGRiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q2OWZmZDQxOWM3MzQ1YjdhMzQwM2RmNmI4OTAxMDI2LmJpbmRQb3B1cChwb3B1cF9lMjBjNDQ1ZjVhYWY0NmIzYjJmNWJhODYwOTI1MGQzMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNTZjNDNmOTgwNDY0MDNhODgyYjgwNzkwYzhmMTVlNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ1Mzg4MjM1LC0wLjk3NzgzNDIxODEwNTI5OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWNkYjczYTAyYTQ2NGI4MTk4NzIyNDNlYjllNzgxMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDhjYmVjZWU0NWZmNGQxOGI0MTc0ZDc3ZWViNmUyMWIgPSAkKCc8ZGl2IGlkPSJodG1sXzA4Y2JlY2VlNDVmZjRkMThiNDE3NGQ3N2VlYjZlMjFiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5USEUgSEVYQUdPTiBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFjZGI3M2EwMmE0NjRiODE5ODcyMjQzZWI5ZTc4MTAzLnNldENvbnRlbnQoaHRtbF8wOGNiZWNlZTQ1ZmY0ZDE4YjQxNzRkNzdlZWI2ZTIxYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNTZjNDNmOTgwNDY0MDNhODgyYjgwNzkwYzhmMTVlNC5iaW5kUG9wdXAocG9wdXBfMWNkYjczYTAyYTQ2NGI4MTk4NzIyNDNlYjllNzgxMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDhlNDI2YmYzMTNhNDUzYmFlMzk3OTAxYmViODU1OGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjcwOTM1LC0wLjAzMjE4NDIzMzI1OTY2MDYzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcxOTVkYzdjZTNhYzQ5ZmU4YTU2NTY1NjIyMGQ3ZGZlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ2OWZkMjFkNTU0YTQxNGJhMGRjMGRkNWQzN2JmN2RjID0gJCgnPGRpdiBpZD0iaHRtbF80NjlmZDIxZDU1NGE0MTRiYTBkYzBkZDVkMzdiZjdkYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VFJFREVHQVIgU1FVQVJFIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzE5NWRjN2NlM2FjNDlmZThhNTY1NjU2MjIwZDdkZmUuc2V0Q29udGVudChodG1sXzQ2OWZkMjFkNTU0YTQxNGJhMGRjMGRkNWQzN2JmN2RjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q4ZTQyNmJmMzEzYTQ1M2JhZTM5NzkwMWJlYjg1NThiLmJpbmRQb3B1cChwb3B1cF83MTk1ZGM3Y2UzYWM0OWZlOGE1NjU2NTYyMjBkN2RmZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NDk4ZDIyYjY2ODE0NDg5YmJiNjEyNzEzNWY0ODY0MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUyLjYyMzkzNDksMS4yODIxMzc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU0ZDYyZGIzZTIyYjRiNzM5M2IxNjA5YjIyMjM4M2FjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgwMjgzYjE3NDQ5OTQ4YTk5YTQyOWRlNDJlYmY0NGI5ID0gJCgnPGRpdiBpZD0iaHRtbF84MDI4M2IxNzQ0OTk0OGE5OWE0MjlkZTQyZWJmNDRiOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VFJJTklUWSBTVFJFRVQgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NGQ2MmRiM2UyMmI0YjczOTNiMTYwOWIyMjIzODNhYy5zZXRDb250ZW50KGh0bWxfODAyODNiMTc0NDk5NDhhOTlhNDI5ZGU0MmViZjQ0YjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzQ5OGQyMmI2NjgxNDQ4OWJiYjYxMjcxMzVmNDg2NDMuYmluZFBvcHVwKHBvcHVwXzU0ZDYyZGIzZTIyYjRiNzM5M2IxNjA5YjIyMjM4M2FjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzYmIwNTQ0MWQ2YTQ1NmI5MTIxNjk3MTY1OGEwZGNiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTEuNTU4NDY3LC0wLjE3NzQ1MjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWQ4MmU1Y2MwYmY2NDQ3NDkzY2E1NjY3ZmIxMjI0YzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzY3ODdiNTFiYjU4NDc5NWJmZDgxZDQyNzE3MTQ3ZjkgPSAkKCc8ZGl2IGlkPSJodG1sXzM2Nzg3YjUxYmI1ODQ3OTViZmQ4MWQ0MjcxNzE0N2Y5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5VUFBFUiBIQU1QU1RFQUQgV0FMSyBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlkODJlNWNjMGJmNjQ0NzQ5M2NhNTY2N2ZiMTIyNGM4LnNldENvbnRlbnQoaHRtbF8zNjc4N2I1MWJiNTg0Nzk1YmZkODFkNDI3MTcxNDdmOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81M2JiMDU0NDFkNmE0NTZiOTEyMTY5NzE2NThhMGRjYi5iaW5kUG9wdXAocG9wdXBfOWQ4MmU1Y2MwYmY2NDQ3NDkzY2E1NjY3ZmIxMjI0YzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjk5YzlmMTc4YjZjNGVkYTlkZjNiM2NlMWFkYjg4NzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS40Mzk3NDM1LC0wLjM0MTI2MzRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWU5YTBmZjQzNTMwNGZkOThmYjQwOWE2MTk5OWM5Y2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzk0NTAwZmMyNzJlNGIxYmEzZTc4ODk1ODMyYjYyZTIgPSAkKCc8ZGl2IGlkPSJodG1sX2M5NDUwMGZjMjcyZTRiMWJhM2U3ODg5NTgzMmI2MmUyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XQUxQT0xFIEdBUkRFTlMgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZTlhMGZmNDM1MzA0ZmQ5OGZiNDA5YTYxOTk5YzljZi5zZXRDb250ZW50KGh0bWxfYzk0NTAwZmMyNzJlNGIxYmEzZTc4ODk1ODMyYjYyZTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjk5YzlmMTc4YjZjNGVkYTlkZjNiM2NlMWFkYjg4NzQuYmluZFBvcHVwKHBvcHVwXzllOWEwZmY0MzUzMDRmZDk4ZmI0MDlhNjE5OTljOWNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU5MTRlYWI1ZDAyMjQ4YTViNjZiMDI1YjhiZTYzZjEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNTIuNjI2MTgxMSwxLjI4NTkzMTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWRmZjVhNWI2ODE1NDFjOWJmMmFiMDQwYjFlYjA5MjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2NhNTc1Y2U1MTQ1NGEzNWFjNDMwYjQzOTYwMTc5OGQgPSAkKCc8ZGl2IGlkPSJodG1sXzdjYTU3NWNlNTE0NTRhMzVhYzQzMGI0Mzk2MDE3OThkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XQUxQT0xFIFNUUkVFVCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVkZmY1YTViNjgxNTQxYzliZjJhYjA0MGIxZWIwOTIwLnNldENvbnRlbnQoaHRtbF83Y2E1NzVjZTUxNDU0YTM1YWM0MzBiNDM5NjAxNzk4ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81OTE0ZWFiNWQwMjI0OGE1YjY2YjAyNWI4YmU2M2YxMC5iaW5kUG9wdXAocG9wdXBfNWRmZjVhNWI2ODE1NDFjOWJmMmFiMDQwYjFlYjA5MjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI1YTVjNDlmM2Q5NDYzM2E3NTRhZTdjMjY4OTE3ODIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMy43Mzg1NjMyLC0xMTcuODQ2NTA1Njc3NDY0ODJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwZmZiNCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MGZmYjQiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGM1YTkyODVkNjMzNDQxNmI4MmJkYmI2ZjdmZTdjZDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGUzOGE1ZTliMTU5NDcwMDlkOWY0MmY3YWQxZmE5NTUgPSAkKCc8ZGl2IGlkPSJodG1sXzhlMzhhNWU5YjE1OTQ3MDA5ZDlmNDJmN2FkMWZhOTU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XQVJXSUNLIFNRVUFSRSBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RjNWE5Mjg1ZDYzMzQ0MTZiODJiZGJiNmY3ZmU3Y2QxLnNldENvbnRlbnQoaHRtbF84ZTM4YTVlOWIxNTk0NzAwOWQ5ZjQyZjdhZDFmYTk1NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMjVhNWM0OWYzZDk0NjMzYTc1NGFlN2MyNjg5MTc4Mi5iaW5kUG9wdXAocG9wdXBfZGM1YTkyODVkNjMzNDQxNmI4MmJkYmI2ZjdmZTdjZDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWYyNmVhN2I1NDA1NDQ0NDg3Nzg3MWY4ZjIxZGY5YTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1Mi41NTgwMDUzLC0wLjI2MjMwOV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OGVlNjk3NDRjYjk0YzAwYTExYTE2NWRhMjE3OGYzZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NTk2ZjYxYzA4YmM0ZDA3ODEyZTk5YTdlMDEyNzY1YyA9ICQoJzxkaXYgaWQ9Imh0bWxfNjU5NmY2MWMwOGJjNGQwNzgxMmU5OWE3ZTAxMjc2NWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFTEJFQ0sgV0FZIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNThlZTY5NzQ0Y2I5NGMwMGExMWExNjVkYTIxNzhmM2Uuc2V0Q29udGVudChodG1sXzY1OTZmNjFjMDhiYzRkMDc4MTJlOTlhN2UwMTI3NjVjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmMjZlYTdiNTQwNTQ0NDQ4Nzc4NzFmOGYyMWRmOWE2LmJpbmRQb3B1cChwb3B1cF81OGVlNjk3NDRjYjk0YzAwYTExYTE2NWRhMjE3OGYzZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYmM2NDkzMWEzNmU0NjhlOTMxZjJiMmNjZWJkY2E0ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjUyOTIxNDIsLTAuMDkyOTMyN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZGM0NzM5ZjY0MTA0NWY2OWFhODdlYzRlMzU4ZjYwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MDdkMWNiYTAxZGY0ZGYwYTZmMjNkZDVhODIyMzMwNCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzA3ZDFjYmEwMWRmNGRmMGE2ZjIzZGQ1YTgyMjMzMDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFTExFU0xFWSBURVJSQUNFIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2RjNDczOWY2NDEwNDVmNjlhYTg3ZWM0ZTM1OGY2MDIuc2V0Q29udGVudChodG1sXzcwN2QxY2JhMDFkZjRkZjBhNmYyM2RkNWE4MjIzMzA0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RiYzY0OTMxYTM2ZTQ2OGU5MzFmMmIyY2NlYmRjYTRkLmJpbmRQb3B1cChwb3B1cF8zZGM0NzM5ZjY0MTA0NWY2OWFhODdlYzRlMzU4ZjYwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMmZlMjcyNGU1NmM0NGNiYTFmNmFiZTc0NmMyODk5NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQ1LjQyMzQ0OTUsLTc1LjY5ODA1NzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODA5M2M3MWY5NmQwNGMzNzk0MTFlNWFiNWE3Mzc1ZTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTMyZTQ1NTA2OGE4NDc3YjhiOTA4MDRhMmU5NzhmM2EgPSAkKCc8ZGl2IGlkPSJodG1sX2UzMmU0NTUwNjhhODQ3N2I4YjkwODA0YTJlOTc4ZjNhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XRUxMSU5HVE9OIFNUUkVFVCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgwOTNjNzFmOTZkMDRjMzc5NDExZTVhYjVhNzM3NWU1LnNldENvbnRlbnQoaHRtbF9lMzJlNDU1MDY4YTg0NzdiOGI5MDgwNGEyZTk3OGYzYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMmZlMjcyNGU1NmM0NGNiYTFmNmFiZTc0NmMyODk5Ny5iaW5kUG9wdXAocG9wdXBfODA5M2M3MWY5NmQwNGMzNzk0MTFlNWFiNWE3Mzc1ZTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzY5NzNiZTM1ZTYxNDFlNjhkOWM1ODVhYjFjMzEzZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0MC43NDAyMjQyLC0xMTEuODQ2MzIzNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZTIzYjYxYzE0ZDU0NzM3YjI3ZGQ5NWU4MzVmZjhkNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNWZhNTQ1ZDMxN2I0NDQ1OGM0MmUyMzJiYTI0ZWUxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzVmYTU0NWQzMTdiNDQ0NThjNDJlMjMyYmEyNGVlMWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldFU1RNT1JFTEFORCBQTEFDRSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZlMjNiNjFjMTRkNTQ3MzdiMjdkZDk1ZTgzNWZmOGQ3LnNldENvbnRlbnQoaHRtbF8zNWZhNTQ1ZDMxN2I0NDQ1OGM0MmUyMzJiYTI0ZWUxYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83Njk3M2JlMzVlNjE0MWU2OGQ5YzU4NWFiMWMzMTNmMi5iaW5kUG9wdXAocG9wdXBfZmUyM2I2MWMxNGQ1NDczN2IyN2RkOTVlODM1ZmY4ZDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzhlZTZhNThlYWYwNGQ0NzgwNzQ5ZTdjNmQ1YjMwZjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS41MjI5NDM3LC0wLjEzNzkzMTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjdiY2Q2ZWFhNTQxNGE0Y2IyNWNlM2Q3YTFmY2IyZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGJlYzNjZDIxMDM5NDE0YmI4MjgxZGMxN2MzNTFiNjkgPSAkKCc8ZGl2IGlkPSJodG1sXzRiZWMzY2QyMTAzOTQxNGJiODI4MWRjMTdjMzUxYjY5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XSElURklFTEQgU1RSRUVUIENsdXN0ZXIgMTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjdiY2Q2ZWFhNTQxNGE0Y2IyNWNlM2Q3YTFmY2IyZWUuc2V0Q29udGVudChodG1sXzRiZWMzY2QyMTAzOTQxNGJiODI4MWRjMTdjMzUxYjY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M4ZWU2YTU4ZWFmMDRkNDc4MDc0OWU3YzZkNWIzMGY0LmJpbmRQb3B1cChwb3B1cF9iN2JjZDZlYWE1NDE0YTRjYjI1Y2UzZDdhMWZjYjJlZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iN2YxZDU3MTdmNzU0ODU1OGVhNzk0ZDAxODAwNmQwYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQ5ODgzOTcsLTAuMTM5Mjk0OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wOTFlZWM5ZjVmNDY0ZmNiOTFhZDMzMGRlNDVmZDA4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjIzNjhhMGM2MmU0Yjc4OGVkZTU0MDA0YjA4NmJjMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzYyMzY4YTBjNjJlNGI3ODhlZGU1NDAwNGIwODZiYzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldJTEZSRUQgU1RSRUVUIENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDkxZWVjOWY1ZjQ2NGZjYjkxYWQzMzBkZTQ1ZmQwODIuc2V0Q29udGVudChodG1sXzM2MjM2OGEwYzYyZTRiNzg4ZWRlNTQwMDRiMDg2YmMwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3ZjFkNTcxN2Y3NTQ4NTU4ZWE3OTRkMDE4MDA2ZDBjLmJpbmRQb3B1cChwb3B1cF8wOTFlZWM5ZjVmNDY0ZmNiOTFhZDMzMGRlNDVmZDA4Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jYzUxODZiMzVlZDE0MmM1YWRmZjFhNThmMjVmZjIwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjU0MzEwODgsLTAuMDk1NTA3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZjI0MTRlMzYwMGE0OGY0OTVmNGE4ZGU2NzA1MWNjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZWFiZWY1OTYwYTQ0NTUyODI4YmFmNjZiNjZlN2FkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmVhYmVmNTk2MGE0NDU1MjgyOGJhZjY2YjY2ZTdhZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldJTExPVyBCUklER0UgUk9BRCBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVmMjQxNGUzNjAwYTQ4ZjQ5NWY0YThkZTY3MDUxY2NkLnNldENvbnRlbnQoaHRtbF8yZWFiZWY1OTYwYTQ0NTUyODI4YmFmNjZiNjZlN2FkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jYzUxODZiMzVlZDE0MmM1YWRmZjFhNThmMjVmZjIwMC5iaW5kUG9wdXAocG9wdXBfNWYyNDE0ZTM2MDBhNDhmNDk1ZjRhOGRlNjcwNTFjY2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOThhZDU2MTRhMzdiNDcyNDhlZGI0OGUzMjdiZDFiMzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFszMC41OTc3OTczLC04MS41OTU3NTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTI0ZGQxZDFmYTk4NDBmZjk5NzFjOWE2M2RhNjMxNmMpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzQ4Y2Q1N2FmMGVhNDMwZjgyOWU3OWY2ZDZkMTMyNzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmQ3MjZmYmU2Mzk4NGYxZWExYWJhNWFkMmEyMmE4NjYgPSAkKCc8ZGl2IGlkPSJodG1sX2JkNzI2ZmJlNjM5ODRmMWVhMWFiYTVhZDJhMjJhODY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XSUxTT04gU1RSRUVUIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzQ4Y2Q1N2FmMGVhNDMwZjgyOWU3OWY2ZDZkMTMyNzIuc2V0Q29udGVudChodG1sX2JkNzI2ZmJlNjM5ODRmMWVhMWFiYTVhZDJhMjJhODY2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk4YWQ1NjE0YTM3YjQ3MjQ4ZWRiNDhlMzI3YmQxYjM4LmJpbmRQb3B1cChwb3B1cF8zNDhjZDU3YWYwZWE0MzBmODI5ZTc5ZjZkNmQxMzI3Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82M2RkZjA2NjUxNzc0NGU4ODFhZGExZTg1YWU5YTc4NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzUxLjQzMjkwNzQsLTAuMzQ4NDU0N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMjRkZDFkMWZhOTg0MGZmOTk3MWM5YTYzZGE2MzE2Yyk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMDQyOTQ3NWI1YWY0NTAyOTQ1MWYwZTQ3NDZkYWMyZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZTUyZjdjNTdmYzE0NjY1OTI0OGIwMGUyMzcwYzkwYiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmU1MmY3YzU3ZmMxNDY2NTkyNDhiMDBlMjM3MGM5MGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldJTkNIRU5ET04gUk9BRCBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IwNDI5NDc1YjVhZjQ1MDI5NDUxZjBlNDc0NmRhYzJkLnNldENvbnRlbnQoaHRtbF9iZTUyZjdjNTdmYzE0NjY1OTI0OGIwMGUyMzcwYzkwYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82M2RkZjA2NjUxNzc0NGU4ODFhZGExZTg1YWU5YTc4NS5iaW5kUG9wdXAocG9wdXBfYjA0Mjk0NzViNWFmNDUwMjk0NTFmMGU0NzQ2ZGFjMmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTczZWY0YWRiYWU0NGJmM2IyNzcwZTMyY2VhOTcyZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs1MS4wOTI1NTcsMS4xNzk0NTU0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEyNGRkMWQxZmE5ODQwZmY5OTcxYzlhNjNkYTYzMTZjKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA2MDMzODMzNTE4NDQ0NDNhYjc5Zjc5ODBmNDEyZTZkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MzZmU0Yzc4MjI4YTRjOTdhNDhjMzgxZGYwNzY3NzY5ID0gJCgnPGRpdiBpZD0iaHRtbF9jM2ZlNGM3ODIyOGE0Yzk3YTQ4YzM4MWRmMDc2Nzc2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V0lOR0FURSBST0FEIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDYwMzM4MzM1MTg0NDQ0M2FiNzlmNzk4MGY0MTJlNmQuc2V0Q29udGVudChodG1sX2MzZmU0Yzc4MjI4YTRjOTdhNDhjMzgxZGYwNzY3NzY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E3M2VmNGFkYmFlNDRiZjNiMjc3MGUzMmNlYTk3MmYxLmJpbmRQb3B1cChwb3B1cF8wNjAzMzgzMzUxODQ0NDQzYWI3OWY3OTgwZjQxMmU2ZCk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
london_grouped_clustering.loc[london_grouped_clustering['Cluster Labels'] == 0, london_grouped_clustering.columns[[1] + list(range(5, london_grouped_clustering.shape[1]))]].head()
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
      <th>Avg_Price</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>406</th>
      <td>2250000.0</td>
      <td>Hotel</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Indian Restaurant</td>
      <td>Bakery</td>
      <td>English Restaurant</td>
      <td>French Restaurant</td>
      <td>Juice Bar</td>
    </tr>
    <tr>
      <th>1215</th>
      <td>2208500.0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Chinese Restaurant</td>
      <td>Grocery Store</td>
      <td>Gastropub</td>
      <td>Indian Restaurant</td>
      <td>Movie Theater</td>
      <td>Garden</td>
    </tr>
    <tr>
      <th>1253</th>
      <td>2217000.0</td>
      <td>Pub</td>
      <td>Breakfast Spot</td>
      <td>Coffee Shop</td>
      <td>Brewery</td>
      <td>Chinese Restaurant</td>
      <td>Train Station</td>
      <td>Lake</td>
      <td>Bakery</td>
      <td>French Restaurant</td>
      <td>Flower Shop</td>
    </tr>
    <tr>
      <th>2225</th>
      <td>2200000.0</td>
      <td>Indian Restaurant</td>
      <td>Pharmacy</td>
      <td>Café</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Liquor Store</td>
      <td>Sandwich Place</td>
      <td>Bakery</td>
      <td>Convenience Store</td>
      <td>Pizza Place</td>
      <td>Design Studio</td>
    </tr>
    <tr>
      <th>2637</th>
      <td>2250000.0</td>
      <td>Construction &amp; Landscaping</td>
      <td>Gastropub</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
london_grouped_clustering.loc[london_grouped_clustering['Cluster Labels'] == 1, london_grouped_clustering.columns[[1] + list(range(5, london_grouped_clustering.shape[1]))]].head()
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
      <th>Avg_Price</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>196</th>
      <td>2450000.0</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Indian Restaurant</td>
      <td>Beer Garden</td>
      <td>Park</td>
      <td>Burger Joint</td>
      <td>Museum</td>
    </tr>
    <tr>
      <th>981</th>
      <td>2480000.0</td>
      <td>Pub</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Grocery Store</td>
      <td>Thai Restaurant</td>
      <td>Gym</td>
      <td>Latin American Restaurant</td>
      <td>Gastropub</td>
      <td>French Restaurant</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>2456875.0</td>
      <td>Food Service</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>2492500.0</td>
      <td>Supermarket</td>
      <td>English Restaurant</td>
      <td>Café</td>
      <td>Gym</td>
      <td>Dry Cleaner</td>
      <td>Hardware Store</td>
      <td>Fast Food Restaurant</td>
      <td>Park</td>
      <td>Pub</td>
      <td>Rental Car Location</td>
    </tr>
    <tr>
      <th>2136</th>
      <td>2461000.0</td>
      <td>Soccer Field</td>
      <td>Windmill</td>
      <td>Spa</td>
      <td>Zoo</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
london_grouped_clustering.loc[london_grouped_clustering['Cluster Labels'] == 2, london_grouped_clustering.columns[[1] + list(range(5, london_grouped_clustering.shape[1]))]].head()
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
      <th>Avg_Price</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>855</th>
      <td>2375000.0</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Food &amp; Drink Shop</td>
      <td>Nature Preserve</td>
      <td>Bookstore</td>
      <td>Café</td>
      <td>Farmers Market</td>
      <td>Thai Restaurant</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>2340000.0</td>
      <td>Pub</td>
      <td>Seafood Restaurant</td>
      <td>Campground</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Vacation Rental</td>
      <td>Fast Food Restaurant</td>
      <td>Exhibit</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
    </tr>
    <tr>
      <th>2068</th>
      <td>2375000.0</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Grocery Store</td>
      <td>Pizza Place</td>
      <td>Hotel</td>
      <td>Indian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Greek Restaurant</td>
      <td>Gastropub</td>
    </tr>
    <tr>
      <th>2129</th>
      <td>2379652.7</td>
      <td>Café</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Hotel</td>
      <td>Bakery</td>
      <td>French Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Juice Bar</td>
      <td>Burger Joint</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>2943</th>
      <td>2367500.0</td>
      <td>Hotel</td>
      <td>Pub</td>
      <td>Garden</td>
      <td>Italian Restaurant</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Chinese Restaurant</td>
      <td>Bar</td>
      <td>Mediterranean Restaurant</td>
      <td>Indian Restaurant</td>
    </tr>
  </tbody>
</table>
</div>




```python
london_grouped_clustering.loc[london_grouped_clustering['Cluster Labels'] == 3, london_grouped_clustering.columns[[1] + list(range(5, london_grouped_clustering.shape[1]))]].head()
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
      <th>Avg_Price</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>391</th>
      <td>2435000.0</td>
      <td>Pub</td>
      <td>Grocery Store</td>
      <td>Diner</td>
      <td>French Restaurant</td>
      <td>Garden</td>
      <td>English Restaurant</td>
      <td>Chinese Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Gym / Fitness Center</td>
      <td>Plaza</td>
    </tr>
    <tr>
      <th>422</th>
      <td>2400000.0</td>
      <td>Pub</td>
      <td>Casino</td>
      <td>Nightclub</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
    </tr>
    <tr>
      <th>1632</th>
      <td>2397132.0</td>
      <td>Grocery Store</td>
      <td>Other Great Outdoors</td>
      <td>Indian Restaurant</td>
      <td>Seafood Restaurant</td>
      <td>Convenience Store</td>
      <td>Coffee Shop</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>1797</th>
      <td>2400000.0</td>
      <td>Art Gallery</td>
      <td>Asian Restaurant</td>
      <td>Zoo</td>
      <td>Flea Market</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
    <tr>
      <th>1914</th>
      <td>2445000.0</td>
      <td>Gastropub</td>
      <td>Athletics &amp; Sports</td>
      <td>Pub</td>
      <td>Greek Restaurant</td>
      <td>Food &amp; Drink Shop</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Cricket Ground</td>
      <td>Art Gallery</td>
      <td>Furniture / Home Store</td>
    </tr>
  </tbody>
</table>
</div>




```python
london_grouped_clustering.loc[london_grouped_clustering['Cluster Labels'] == 4, london_grouped_clustering.columns[[1] + list(range(5, london_grouped_clustering.shape[1]))]].head()
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
      <th>Avg_Price</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2242</th>
      <td>2.300000e+06</td>
      <td>Farm</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
      <td>Fish &amp; Chips Shop</td>
    </tr>
    <tr>
      <th>2405</th>
      <td>2.286679e+06</td>
      <td>Café</td>
      <td>Pub</td>
      <td>Bar</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>French Restaurant</td>
      <td>Park</td>
      <td>Convenience Store</td>
      <td>Belgian Restaurant</td>
      <td>Supermarket</td>
    </tr>
    <tr>
      <th>2685</th>
      <td>2.287500e+06</td>
      <td>Pub</td>
      <td>Art Museum</td>
      <td>Reservoir</td>
      <td>Gift Shop</td>
      <td>Brewery</td>
      <td>Hunan Restaurant</td>
      <td>English Restaurant</td>
      <td>Event Space</td>
      <td>Exhibit</td>
      <td>Factory</td>
    </tr>
    <tr>
      <th>3376</th>
      <td>2.298000e+06</td>
      <td>Hotel</td>
      <td>Zoo</td>
      <td>Fish Market</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
    <tr>
      <th>4284</th>
      <td>2.265000e+06</td>
      <td>Pub</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Exhibit</td>
      <td>Factory</td>
      <td>Falafel Restaurant</td>
      <td>Farm</td>
      <td>Farmers Market</td>
      <td>Fast Food Restaurant</td>
      <td>Filipino Restaurant</td>
      <td>Film Studio</td>
    </tr>
  </tbody>
</table>
</div>



### Results and Discussion section

First of all, even though the London Housing Market may be in a rut, it is still an "ever-green" for business affairs.

We may discuss our results under two main perspectives.

First, we may examine them according to neighborhoods/London areas. It is interesting to note that, although West London (Notting Hill, Kensington, Chelsea, Marylebone) and North-West London (Hampsted) might be considered highly profitable venues to purchase a real estate according to amenities and essential facilities surrounding such venues i.e. elementary schools, high schools, hospitals & grocery stores, South-West London (Wandsworth, Balham) and North-West London (Isliington) are arising as next future elite venues with a wide range of amenities and facilities. Accordingly, one might target under-priced real estates in these areas of London in order to make a business affair.

Second, we may analyze our results according to the five clusters we have produced. Even though, all clusters could praise an optimal range of facilities and amenities, we have found two main patterns. The first pattern we are referring to, i.e. Clusters 0, 2 and 4, may target home buyers prone to live in 'green' areas with parks, waterfronts. Instead, the second pattern we are referring to, i.e. Clusters 1 and 3, may target individuals who love pubs, theatres and soccer.

### Conclusion

Finally, we drew the conclusion that even though the London Housing Market may be in a rut, it is still an "ever-green" for business affairs. We discussed our results under two main perspectives. First, we examined them according to neighborhoods/London areas. although West London (Notting Hill, Kensington, Chelsea, Marylebone) and North-West London (Hampsted) might be considered highly profitable venues to purchase a real estate according to amenities and essential facilities surrounding such venues i.e. elementary schools, high schools, hospitals & grocery stores, South-West London (Wandsworth, Balham) and North-West London (Isliington) are arising as next future elite venues with a wide range of amenities and facilities. Accordingly, one might target under-priced real estates in these areas of London in order to make a business affair. Second, we analyzed our results according to the five clusters we produced. While Clusters 0, 2 and 4 may target home buyers prone to live in 'green' areas with parks, waterfronts, Clusters 1 and 3 may target individuals who love pubs, theatres and soccer.


```python

```
