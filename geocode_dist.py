import pgeocode
import pandas as pd
import numpy as np
df = pd.read_excel("C:/Users/ecriggins/Downloads/geocode.xlsx")
print(df.head())

gdata = pgeocode.GeoDistance('US')
distance = []
for zipcode in df['ZIP']:
    tempdist = gdata.query_postal_code(24515, zipcode)
    distance.append(tempdist)
df['Distance'] = distance
print(f'distance: {distance}')
print(f'--------------------')
print(df.head())

df['Distance2'] = [gdata.query_postal_code(24515, zipcode) for zipcode in df['ZIP']]
print(df.head())

# to create a new column pulling only the lat, long coordinates into a tuple
#df['new_column'] = list(zip(df.latitude, df.longitude))
# or
#df['new_column'] = df[['latitude', 'longitude']].apply(tuple, axis = 1)

df = pd.read_csv("C:/Users/ecriggins/Downloads/data.csv")
df = df[['AACOMAS_ID', 'ZIP', 'COUNTRY']]
print(df.head(25).to_string())

# set libraries
ca_zip = pgeocode.Nominatim('CA')
us_zip = pgeocode.Nominatim('US')

# cast zip column to string type
df['ZIP'] = df['ZIP'].astype(str)

# create a list of all zipcodes
zip_list = df.ZIP.values.tolist()

# loop through zip codes to produce lat long coordinates
point = []
for z in zip_list:
    lat = np.where(len(z) == 7, ca_zip.query_postal_code(z).latitude, us_zip.query_postal_code(z[0:5]).latitude)
    long = np.where(len(z) == 7, ca_zip.query_postal_code(z).longitude, us_zip.query_postal_code(z[0:5]).longitude)
    point.append((lat, long))

# add column for point coordinates
df['POINT'] = point
print(df.head(25).to_string())
