import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import shap
import numpy as np
import umap


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
    ax.scatter(principalDf[0][df_r["AGEGROUP"] == age_group], principalDf[1][df_r["AGEGROUP"] == age_group], principalDf[2][df_r["AGEGROUP"] == age_group], marker='o', label=age_group, alpha=0.7)

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
    ax.scatter(principalDf[0][df_r["GENDER"] == gender], principalDf[1][df_r["GENDER"] == gender], principalDf[2][df_r["GENDER"] == gender], marker='o', label=gender, alpha=0.7)

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
"""
Features related to Hip and Knee movement (also height) distinguish gender well
"""


# feature importance of Random Forest
x = df_r[columns_to_cal]
y = df_r['AGEGROUP']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

train_accuracy = accuracy_score(y_train, rf.predict(x_train))
test_accuracy = accuracy_score(y_test, rf.predict(x_test))

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

important_features_rf = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False).head(10).index
print("Top 10 features which is critical to classify by age group")
for feature in important_features_rf:
    print(feature)
"""
Features related to Hip Flexion and Knee Flexion, Height, Weight and TempStrideLength
are important features to classify by age group
"""


# analyze with shap values
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x)

plt.figure()
plt.suptitle("Top 10 features for age group 20", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 0], x, max_display=10)
plt.tight_layout()
plt.savefig("figures/age_group_20.png", format='png', dpi=300)
plt.figure()
plt.suptitle("Top 10 features for age group 30", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 1], x, max_display=10)
plt.tight_layout()
plt.savefig("figures/age_group_30.png", format='png', dpi=300)
plt.figure()
plt.suptitle("Top 10 features for age group 40", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 2], x, max_display=10)
plt.tight_layout()
plt.savefig("figures/age_group_40.png", format='png', dpi=300)
plt.figure()
plt.suptitle("Top 10 features for age group 50", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 3], x, max_display=10)
plt.tight_layout()
plt.savefig("figures/age_group_50.png", format='png', dpi=300)
plt.figure()
plt.suptitle("Top 10 features for age group 60", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 4], x, max_display=10)
plt.tight_layout()
plt.savefig("figures/age_group_60.png", format='png', dpi=300)


# Extract the union set of the top 10 features selected by each age group.
feature_union = set()
for i in range(shap_values.shape[2]):
    shap_values_abs_mean = np.mean(np.abs(shap_values[:, :, i]), axis=0)
    top_10_indices = np.argsort(shap_values_abs_mean)[-10:]
    top_10_features = set(x.columns[top_10_indices])
    feature_union = feature_union.union(top_10_features)

print(f"critical {len(feature_union)} features to classify by age group")
for feature in feature_union:
    print(feature)


# UMAP
"""
Let's visualize how those features can well classify people by age group.
"""
data = x[list(feature_union)]
umap_model = umap.UMAP(n_components=3)
umap_embedding = umap_model.fit_transform(data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], umap_embedding[:, 2], c=y, cmap='coolwarm', s=10, alpha=0.7)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_ticks(np.arange(20, 61, 10))
cbar.set_label('Age Group')

plt.tight_layout()
