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

print(df.head())

df['Distance2'] = [gdata.query_postal_code(24515, zipcode) for zipcode in df['ZIP']]
print(df.head())