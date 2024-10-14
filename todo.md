# Sid 쌤 부탁

- [ ] 엑셀 데이터 시트가 여러개 있음. -> 다 중복되는건지 확인
- [ ] Age랑 Age group 중복되는거 주의
- [ ] 각각의 사람들 대응시켜보기?
    r 하고 l이 대응되는 듯? 갯수가 같음
- [ ] 데이터 정리 및 전처리 해서 PCA 돌려보기 -> age group에 따라서 분포 차이 보기

- [ ] t-SNE, UMAP? 등 non linear 방식들 확인해보기


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