import pandas as pd


fn = "Combined_Right_Files_Columns.xlsx"

dfs = pd.read_excel(fn, sheet_name=None)
for sheet_name, df in dfs.items():
    print(f"Sheet name: {sheet_name}")
del dfs


df_r1 = pd.read_excel(fn, sheet_name="RIGHT1").transpose()
df_r2 = pd.read_excel(fn, sheet_name="RIGHT2")
df_l1 = pd.read_excel(fn, sheet_name="LEFT1").transpose()
df_l2 = pd.read_excel(fn, sheet_name="LEFT2")

df_r1.columns = df_r1.iloc[0]
df_r1 = df_r1.drop(df_r1.index[0]).reset_index(drop=True)
df_r1.columns.name = None

df_l1.columns = df_l1.iloc[0]
df_l1 = df_l1.drop(df_l1.index[0]).reset_index(drop=True)
df_l1.columns.name = None


cols = df_r1.columns.tolist()
index_of_second_name = [i for i, x in enumerate(cols) if x == "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance"][1]
cols[index_of_second_name] = "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"
df_r1.columns = cols

cols = df_l1.columns.tolist()
index_of_second_name = [i for i, x in enumerate(cols) if x == "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance"][1]
cols[index_of_second_name] = "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"
df_l1.columns = cols

df_r2.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)
df_l2.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)


df_l1["KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance"]
df_l1["KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"]


df_r1.head()
df_r2.head()
df_l1.head()
df_l2.head()

common_columns = df_r1.columns.intersection(df_r2.columns)
df_r1.columns.difference(common_columns)
df_r2.columns.difference(common_columns)

common_columns = df_l1.columns.intersection(df_l2.columns)
df_l1.columns.difference(common_columns)
df_l2.columns.difference(common_columns)


df_r1.columns[df_r1.columns.duplicated()]
df_r2.columns[df_r2.columns.duplicated()]
df_l1.columns[df_l1.columns.duplicated()]
df_l2.columns[df_l2.columns.duplicated()]

common_columns = df_l1.columns.intersection(df_l2.columns)

df_l1.columns.difference(common_columns)
df_l2.columns.difference(common_columns)

"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance" in df_r2.columns
"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1" in df_r1.columns

df_r1.columns
df_l1.columns

df_r1.shape
df_r2.shape
df_l1.shape
df_l2.shape


# drop NAs
(df_r1[['Age', "Height",  "Weight"]] == df_l1[['Age', "Height",  "Weight"]]).all()
df_r1.columns

df_r1.isna().any()



df_r2 = df_r2[df_r1.columns]
for column in df_r1.columns:
    df_r1[column] = df_r1[column].astype(df_r2[column].dtype)

df_right = pd.merge(df_r1, df_r2, on=df_r1.columns.to_list())

df_r1["Age"].astype("Float64")

type(df_r1["Age"].iloc[0])
type(df_r1["Age"])

df_r1["Age"].astype(df_r2["Age"].dtype)
df_r1["Age"].dtype

(df_r1["Age"] == "Age").sum()
df_r1.shape
df_r2.shape

df_r1.dtypes
df_r2.dtypes


# read data
fn = "Combined_Right_Files_Columns.xlsx"

df_r = pd.read_excel(fn, sheet_name="RIGHT2")
df_l = pd.read_excel(fn, sheet_name="LEFT2")

# resolve duplicated column problem
df_r.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)
df_l.rename(columns={"KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance.1": "KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance_"}, inplace=True)

df_r.shape
df_l.shape

df_r.dropna().shape
df_l.dropna().shape


# Normalization

columns_to_exclude = ['Age', 'AGEGROUP', 'GENDER']
columns_to_normalize = df_r.columns.difference(columns_to_exclude)

scaler =  StandardScaler()
df_r[columns_to_normalize] = scaler.fit_transform(df_r[columns_to_normalize])
scaler =  StandardScaler()
df_l[columns_to_normalize] = scaler.fit_transform(df_l[columns_to_normalize])


# PCA



df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df.head()


from sklearn.preprocessing import StandardScaler  # 표준화 패키지 라이브러리 
x = df.drop(['target'], axis=1).values # 독립변인들의 value값만 추출
y = df['target'].values # 종속변인 추출

x = StandardScaler().fit_transform(x) # x객체에 x를 표준화한 데이터를 저장

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
pd.DataFrame(x, columns=features).head()


from sklearn.decomposition import PCA
pca = PCA(n_components=2) # 주성분을 몇개로 할지 결정
printcipalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2'])
# 주성분으로 이루어진 데이터 프레임 구성


principalDf.head()
pca.explained_variance_ratio_

pca = PCA(n_components=3)

printcipalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=printcipalComponents, columns = ['principal component1', 'principal component2', '3'])

pca.explained_variance_ratio_

# visualizaation

import matplotlib.pyplot as plt

# 그냥 3d interaction 됨!

plt.ion()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for age_group in df_r["AGEGROUP"].value_counts().index:
    ax.scatter(principalDf[0][df_r["AGEGROUP"] == age_group], principalDf[1][df_r["AGEGROUP"] == age_group], principalDf[2][df_r["AGEGROUP"] == age_group], marker='o', label=age_group)

plt.legend()

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

ax.view_init(elev=30, azim=60)

plt.show()


df_r["AGEGROUP"].value_counts()
df_r["AGEGROUP"].value_counts().index


# Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Agegroup을 목표 변수, 나머지를 설명 변수로 설정
X = df.drop(columns=["Agegroup"])  # 설명 변수
y = df["Agegroup"]  # 목표 변수

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 학습
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# 변수 중요도 출력
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)




from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# UMAP 결과로 나온 그룹 라벨을 포함한 데이터프레임이 있다고 가정
X = df.drop(columns=["group"])  # 설명 변수
y = df["group"]  # UMAP으로 구분된 그룹

# 랜덤 포레스트 모델 학습
rf = RandomForestClassifier()
rf.fit(X, y)

# 변수 중요도 출력
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_importances)


from sklearn.ensemble import RandomForestClassifier

x = df_r[columns_to_cal]
y = df_r['AGEGROUP']

rf = RandomForestClassifier()
rf.fit(x, y)

feature_importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
print(feature_importances)


from sklearn.metrics import accuracy_score

y_pred = rf.predict(x)
accuracy = accuracy_score(y, y_pred)


import shap

# 랜덤 포레스트 모델 학습 후 SHAP 값 계산
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(x)

x.shape
shap_values.shape
shap_values[:, :, 0].shape

# SHAP 값 시각화
plt.figure()
plt.suptitle("Top 10 features for age group 20", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 0], x, max_display=10)
plt.tight_layout()
plt.figure()
plt.suptitle("Top 10 features for age group 30", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 1], x, max_display=10)
plt.tight_layout()
plt.figure()
plt.suptitle("Top 10 features for age group 40", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 2], x, max_display=10)
plt.tight_layout()
plt.figure()
plt.suptitle("Top 10 features for age group 50", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 3], x, max_display=10)
plt.tight_layout()
plt.figure()
plt.suptitle("Top 10 features for age group 60", fontsize=16, fontweight='bold')
shap.summary_plot(shap_values[:, :, 4], x, max_display=10)
plt.tight_layout()


import numpy as np

shap_values_abs_mean = np.mean(np.abs(shap_values[:, :, 0]), axis=0)
top_10_indices = np.argsort(shap_values_abs_mean)[-10:]
top_10_features = x.columns[top_10_indices]


feature_union = set()
for i in range(shap_values.shape[2]):
    shap_values_abs_mean = np.mean(np.abs(shap_values[:, :, i]), axis=0)
    top_10_indices = np.argsort(shap_values_abs_mean)[-10:]
    top_10_features = set(x.columns[top_10_indices])
    feature_union = feature_union.union(top_10_features)

len(feature_union)

plt.figure()
plt.title("Top 10 features for age group 20")
shap.summary_plot(shap_values[:, :, 0], x, max_display=10)
plt.figure()
shap.summary_plot(shap_values[:, :, 1], x, max_display=10)
shap.summary_plot(shap_values[:, :, 2], x, max_display=10)
shap.summary_plot(shap_values[:, :, 3], x, max_display=10)
shap.summary_plot(shap_values[:, :, 4], x, max_display=10)




# UMAP visualization

import umap
import numpy as np
import pandas as pd

data = x[list(feature_union)]

# UMAP 모델 학습 (원본 데이터에 대해)
umap_model = umap.UMAP(n_components=2)
umap_model.fit(data)

# 새로운 데이터 (다른 데이터에 대해 동일한 매핑을 적용)
new_data = np.random.rand(50, 10)  # 50개의 샘플, 10개의 특징

# 학습된 UMAP 모델을 사용하여 새로운 데이터 변환 (차원 축소 적용)
new_data_embedding = umap_model.transform(new_data)

# 결과 확인
print(new_data_embedding.shape)  # (50, 2)


data = x[list(feature_union)]
umap_model = umap.UMAP(n_components=3)
umap_embedding = umap_model.fit_transform(data)

# UMAP 결과 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], umap_embedding[:, 2], c=y, cmap='coolwarm', s=10, alpha=0.7)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Age Group')

plt.tight_layout()
