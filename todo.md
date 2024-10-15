# Sid 쌤 부탁

- [x] 엑셀 데이터 시트가 여러개 있음. -> 다 중복되는건지 확인
- [x] Age랑 Age group 중복되는거 주의
- [x] 각각의 사람들 대응시켜보기?
    r 하고 l이 대응되는 듯? 갯수가 같음
- [x] 데이터 정리 및 전처리 해서 PCA 돌려보기 -> age group에 따라서 분포 차이 보기
- [x] PCA 시각화 하기
- [x] RandomForest 활용해보기, Shap value
- [x] train test split 하기
- [ ] XGBoost 도 활용해보기
- [ ] UMAP으로 시각화해보기
- [ ] 모든 과정 r, l 각자 따로 돌려보기

데이터 전처리에서 중복되는 컬럼 있음
KINEMATICSAnkleDorsiflexion_(footoff)MeaninStance
이 컬럼 6.5 정도 범위의 값 있는건 그대로
-14 정도 범위의 값 있는 컬럼은 뒤에 언더바(_) 붙임.

r1은 l1, r2는 l2와 대응 됨
결측값들 대응되는 것끼리 일단 보완

left right 그냥 따로 돌려보기

1, 2는 merge 시키기

PCA 돌릴 때, Age, Agegroup, gender 빼고 돌리기

아... 9개 차이.. 그냥 r1하고 r2는 똑같은 거였네... 다 헛짓거리였음.

PCA로는 일단 variance explained 자체가 높게 안나옴. 전체적인 일관된 패턴이 없다는 소리.
Age group을 잘 분간하는 변수를 찾아야 하는데...

일단 PCA로 visualization은 시켜보기

## 교훈

UMAP으로는 시각화하기에 매우 좋지만, 결국 변수 중요도를 설명하기 위해서는 random forest 같은 기법이 필요하다.

## 결과

PCA를 통해서 age group을 구분짓기는 어렵다. 단순한 linear 관계가 아닌 듯.

PCA를 통해서 gender 구분은 가능하다. PC2가 gender를 구분짓는 주요한 변수이고, PC2에 큰 영향을 미치는 feature들은 아래와 같다.
KINEMATICSHipFlexionLateSwingMax  
KINEMATICSKneeFlexionPeakinSwing  
KineticsHipFlx/ExtMomentTimetopeakflexor  
KINEMATICSHipRotationMax  
KINEMATICSHipAdductionMax  
KINEMATICSHipRotationAverage  
KINEMATICSKneeFlexionTimetoPkFlexion  
Height  
KINEMATICSHipFlexionEarlyStanceMax  
KINEMATICSKneeFlexion@Normal  

Random forest의 feature importance로 본 age group 구분에 주요한 feature들 top 10
KINEMATICSHipFlexionLateSwingMax  
BMI  
KINEMATICSHipFlexion@NormalRange  
KINEMATICSHipFlexionRange  
Weight  
KINEMATICSKneeFlexionPeakinSwing  
Height  
KINEMATICSHipFlexionEarlyStanceMax  
KINEMATICSKneeFlexionRangeduetorotation  
KineticsHipFlx/ExtMomentPeakExtensor-SW  
Random forest 학습할 때마다 조금씩 달라지긴 하지만, 거의 hip, knee 관련 변수들이 대부분이다. weight, height, BMI도 주요함. TempStepLength(cm) 도 한 번씩 top10에 포함됨.
