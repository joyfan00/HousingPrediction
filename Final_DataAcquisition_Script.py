import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
import glob
import zipfile

#unzip files
filenames = glob.glob('*zip')
#print(filenames)
for file in filenames:
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall("apartments_pl")

#load files 
filenames = glob.glob('apartments_pl/*csv')
df = pd.DataFrame()
for file in filenames:
    df_temp = pd.read_csv(file)
    #make date value
    date = pd.to_datetime(file.split('_')[3] + file.split('_')[4].split('.')[0] + '01')
    #add date column and date value for each row #what should the type be?
    df_temp["date"] = date
    df = pd.concat([df, df_temp])

#df.shape[0]

#########
#The following is not executed


# #drop observations
# df = df.dropna()

# #convert to numerical data
# df['condition'] = df['condition'].map({'low': 0, 'premium': 1})
# df['type'] = df['type'].map({'apartmentBuilding': 1, 'blockOfFlats': 2, 'tenement': 3})
# for col in ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']:
#     df[col] = df[col].map({'no': 0, 'yes': 1})

# for col in ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom', 'condition', 'type']:
#     df[col] = df[col].astype(np.int8)

# #drop features
# toDrop = ['id', 'ownership'] #buildMaterial, city
# df = df.drop(toDrop, axis=1)

# #drop low variance featurse
# sel = VarianceThreshold(threshold=.05)
# sel.fit(df/df.mean())
# mask = sel.get_support() 
# #Still need longtude, latitude, and buildYear
# df = df.loc[:, mask]

# df.info()
# df.describe()
# df.columns

# df.to_csv('full_dataset.csv', index=False)


#########


##Remove Duplicates
df = pd.read_csv('full_dataset.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

id_counts = df['id'].value_counts() #id_counts is a list of the ids that appear 7 times. There are 1460 apartments.

result = df[df['id'].isin(id_counts.index[id_counts == 7])].copy()

#result.head() #result is the dataframe that has all the apartments that appear seven times
#print(id_counts)
#print(id_counts.index[id_counts == 7].shape)

feature_columns = df.columns.difference(['price', 'date']) #features to compare to in results dataframe
#print(feature_columns)

def check_same_features(group):
    return (group[feature_columns].nunique() == 1).all()

filtered_apartments = result.groupby('id').filter(check_same_features)['id'].unique()

#print(filtered_apartments)
#filtered_apartments.shape #There are 147 apartments out of 1460 that have the exact same values for all the features except for date and price 

result_df = result[result['id'].isin(filtered_apartments)].copy() #result_df is the final dataframe with apartments that appear 7 times, are the same for all features except for date and price.

#result = df[df['id'].isin(id_counts.index[id_counts == 7])].copy()
#result_df.to_csv('dupes.csv', index=False)

#result_df.head()
result_df.to_csv('Dataset_to_merge.csv', index=False)

##Merge
df = pd.read_csv('Dataset_to_merge.csv')
df_sorted = df.sort_values(by='date')
#grouped = df_sorted.groupby('id')['price'].agg(lambda x: sorted(x.tolist())).reset_index()
grouped = df_sorted.groupby('id').apply(
    lambda x: sorted(x['price'].tolist(), 
                     key=lambda y: df_sorted.loc[df_sorted['price'] == y, 'date'].iloc[0])
                                        ).reset_index(name='sorted_prices')
#print(grouped)
#type(grouped)
#grouped.head(10)
#grouped['sorted_prices']

#drop all duplicates from Dataset_to_merge
dropped_dup_df = df.drop_duplicates('id')
#dropped_dup_df.info()

#sort Dataset_to_merge and grouped by "id"
dropped_dup_df_sorted = dropped_dup_df.sort_values('id')
grouped_sorted = grouped.sort_values('id')

#drop price and/or date from Dataset_to_merge
#concat Dataset_to_merge and grouped by "id"
final = pd.merge(dropped_dup_df, grouped, on='id')
#final.head()

final = final.drop(labels = ['price', 'date'], axis=1)

final.to_csv('Dataset_to_clean.csv', index=False)

##Clean 
#df = pd.read_csv('Dataset_to_clean.csv', converters={"sorted_prices": lambda x: x.strip("[]").split(", ")})
df = pd.read_csv('Dataset_to_clean.csv')
df["sorted_prices"] = df["sorted_prices"].fillna("[]").apply(lambda x: eval(x))

#drop observations
df = df.dropna()

#convert to numerical data
df['condition'] = df['condition'].map({'low': 0, 'premium': 1})
df['type'] = df['type'].map({'apartmentBuilding': 1, 'blockOfFlats': 2, 'tenement': 3})
for col in ['hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom']:
    df[col] = df[col].map({'no': 0, 'yes': 1})

for col in ['rooms','floor','floorCount','hasParkingSpace', 'hasBalcony', 'hasElevator', 'hasSecurity', 'hasStorageRoom', 'condition', 'type']:
    df[col] = df[col].astype(np.int8)

for col in ['buildYear', 'floorCount', 'poiCount']:
    df[col] = df[col].astype(np.int16)

#drop features
#toDrop = ['ownership'] #buildMaterial, city
#df = df.drop(toDrop, axis=1)

df['city'] = df['city'].map({'szczecin':1,
                             'gdynia':2,
                             'krakow':3,
                             'poznan':4,
                             'bialystok':5,
                             'gdansk':6,
                             'wroclaw':7,
                             'radom':8,
                             'rzeszow':9,
                             'katowice':10,
                             'lublin':11,
                             'czestochowa':12,
                             'warszawa':13,
                             'bydgoszcz':14
                             })
df['ownership'] = df['ownership'].map({'condominium': 0, 'cooperative': 1})
df['buildingMaterial'] = df['buildingMaterial'].map({'brick': 0, 'concreteSlab': 1})

for col in ['city', 'ownership', 'buildingMaterial']:
    df[col] = df[col].astype(np.int8)

###This code was not ran
# #drop low variance features
# sel = VarianceThreshold(threshold=.05)
# #need all to be numerical data
# sel.fit(df/df.mean())
# mask = sel.get_support() 
# #Still need longtude, latitude, and buildYear
# df_dropped_low_variance = df.loc[:, mask]
###End

df.to_csv('Cleaned_Apartments.csv', index=False)