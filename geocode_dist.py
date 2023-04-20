import pgeocode
import pandas as pd
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
df['new_column'] = list(zip(df.latitude, df.longitude))
# or
df['new_column'] = df[['latitude', 'longitude']].apply(tuple, axis = 1)