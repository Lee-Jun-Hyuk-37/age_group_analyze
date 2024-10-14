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
columns_to_normalize = df.columns.difference(columns_to_exclude)

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

