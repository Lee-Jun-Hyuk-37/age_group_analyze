import pandas as pd
from sklearn.preprocessing import StandardScaler


# read data
fn = "Combined_Right_Files_Columns.xlsx"

df_r = pd.read_excel(fn, sheet_name="RIGHT2")
df_l = pd.read_excel(fn, sheet_name="LEFT2")

# resolve duplicated column problem
df_r.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)
df_l.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)

# drop NAs
df_r.dropna(inplace=True)
df_l.dropna(inplace=True)

# Normalization (Except Age, Age group, Gender)
columns_to_exclude = ['Age', 'AGEGROUP', 'GENDER']
columns_to_normalize = df_r.columns.difference(columns_to_exclude)

scaler =  StandardScaler()
df_r[columns_to_normalize] = scaler.fit_transform(df_r[columns_to_normalize])
scaler =  StandardScaler()
df_l[columns_to_normalize] = scaler.fit_transform(df_l[columns_to_normalize])

# PCA



# visualization



