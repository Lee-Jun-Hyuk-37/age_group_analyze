import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# read data
fn = "Combined_Right_Files_Columns.xlsx"

df_r = pd.read_excel(fn, sheet_name="RIGHT2")
df_l = pd.read_excel(fn, sheet_name="LEFT2")

# resolve duplicated column problem
df_r.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)
df_l.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)

# drop NAs
df_r.dropna(inplace=True)
df_r.reset_index(drop=True, inplace=True)
df_l.dropna(inplace=True)
df_l.reset_index(drop=True, inplace=True)

# Normalization (Except Age, Age group, Gender)
columns_to_exclude = ['Age', 'AGEGROUP', 'GENDER']
columns_to_cal = df_r.columns.difference(columns_to_exclude)

scaler =  StandardScaler()
df_r[columns_to_cal] = scaler.fit_transform(df_r[columns_to_cal])
scaler =  StandardScaler()
df_l[columns_to_cal] = scaler.fit_transform(df_l[columns_to_cal])

# PCA
pca = PCA()
printcipalComponents = pca.fit_transform(df_r[columns_to_cal])
principalDf = pd.DataFrame(data=printcipalComponents)

principalDf.head()
pca.explained_variance_ratio_[:3].sum()
"""only about 30% of variance can be explained by three PCs..."""


# PCA visualization
plt.ion()

# by age_group
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for age_group in df_r["AGEGROUP"].value_counts().index:
    ax.scatter(principalDf[0][df_r["AGEGROUP"] == age_group], principalDf[1][df_r["AGEGROUP"] == age_group], principalDf[2][df_r["AGEGROUP"] == age_group], marker='o', label=age_group)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
# ax.view_init(elev=30, azim=60)
plt.show()
"""age group cannot be well classified by those three PCs"""

# by gender
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for gender in df_r["GENDER"].value_counts().index:
    ax.scatter(principalDf[0][df_r["GENDER"] == gender], principalDf[1][df_r["GENDER"] == gender], principalDf[2][df_r["GENDER"] == gender], marker='o', label=gender)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.legend()
# ax.view_init(elev=30, azim=60)
plt.show()
"""
gender can be well classified by PC2
Then let's see what PC2 is
"""

columns = df_r[columns_to_cal].columns
loadings = pd.DataFrame(pca.components_.T, index=columns, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
largest_influence_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(10).index
print("Top 10 features that has largest influence to PC2")
for feature in largest_influence_pc2:
    print(feature)


# feature importance of Random Forest
x = df_r[columns_to_cal]
y = df_r['AGEGROUP']

rf = RandomForestClassifier()
rf.fit(x, y)

important_features_rf = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False).head(10).index
print("Top 10 features which is critical to classify by age group")
for feature in important_features_rf:
    print(feature)
"""
Features related to Hip Flexion and Knee Flexion, Height, Weight and TempStrideLength
are important features to classify by age group
"""

# analyze with shap values


# UMAP
"""
Let's visualize how those ten features can well classify people by age group
"""