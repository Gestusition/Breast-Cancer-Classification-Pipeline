# Göğüs Kanseri Sınıflandırması - Uçtan Uca Makine Öğrenmesi Projesi

## Projenin Amacı

Bu proje, **Breast Cancer Wisconsin** veri seti üzerinde uçtan uca bir makine öğrenmesi akışı uygular. Amaç; veri inceleme (EDA), veri ön işleme, öznitelik mühendisliği, model eğitimi, model karşılaştırma, çapraz doğrulama, hiperparametre ayarlama ve sonuç yorumlama adımlarını temiz ve anlaşılır bir Python projesi olarak tamamlamaktır.

**Önemli Uyarı:** Bu model yalnızca eğitim amaçlıdır. Gerçek tıbbi teşhis aracı olarak KULLANILAMAZ.

## Veri Seti

- **Kaynak:** `sklearn.datasets.load_breast_cancer`
- **Örnek sayısı:** 569
- **Orijinal öznitelik sayısı:** 30 (tümü sayısal)
- **Hedef kodlaması:** `1 = malignant` (kötü huylu), `0 = benign` (iyi huylu)
- **Sınıf dağılımı:** Malignant %37.3 (212), Benign %62.7 (357)

### Orijinal Öznitelikler

30 sayısal öznitelik; her biri tümör hücre çekirdeklerinin 10 farklı karakteristiği üzerinden hesaplanmış **mean**, **standard error** ve **worst** değerleridir:

`mean radius`, `mean texture`, `mean perimeter`, `mean area`, `mean smoothness`, `mean compactness`, `mean concavity`, `mean concave points`, `mean symmetry`, `mean fractal dimension`, `radius error`, `texture error`, `perimeter error`, `area error`, `smoothness error`, `compactness error`, `concavity error`, `concave points error`, `symmetry error`, `fractal dimension error`, `worst radius`, `worst texture`, `worst perimeter`, `worst area`, `worst smoothness`, `worst compactness`, `worst concavity`, `worst concave points`, `worst symmetry`, `worst fractal dimension`

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python main.py
```

Script çalıştırıldığında sırasıyla şu adımlar gerçekleşir:
1. Veri seti yüklenir, hedef kodlaması malignant=1 olacak şekilde çevrilir
2. EDA yapılır (`outputs/class_distribution.png`)
3. Imputation'ı göstermek amacıyla ~%1 yapay eksik değer enjekte edilir
4. Öznitelik mühendisliği (3 yeni öznitelik)
5. Aykırı değer raporlaması (betimsel)
6. Stratified train/validation/test bölme (%60/%20/%20)
7. Korelasyon ısı haritası (`outputs/correlation_heatmap.png`)
8. Pipeline + 3 model eğitimi ve validation karşılaştırması
9. GridSearchCV ile hiperparametre ayarlama
10. Final modelin train+validation ile eğitimi ve test değerlendirmesi
11. Açıklanabilirlik (feature importance + SHAP)
12. Sonuç yorumu

## Üretilen Yeni Öznitelikler

| Öznitelik | Formül | Açıklama |
|---|---|---|
| `radius_growth_ratio` | `worst radius / (mean radius + eps)` | Tümör çapındaki büyüme oranı; >1 kötüye gidişi gösterir |
| `area_growth_ratio` | `worst area / (mean area + eps)` | Tümör alanındaki büyüme oranı |
| `compactness_change` | `worst compactness - mean compactness` | Tümör şekil düzensizliğindeki değişim; pozitif tümörün kötüleştiğini gösterir |

## Yapay Kategorik Değişken

Kategorik encoding adımını göstermek amacıyla **öğrenme amaçlı** bir yapay kategorik değişken oluşturulmuştur:

- `radius_category`: `mean radius` özniteliğinin quantile tabanlı üç sınıfa (small / medium / large) ayrılması
- Bu işlem `RadiusCategoryTransformer` adlı özel sklearn transformer'ı ile pipeline içinde gerçekleşir
- Bin sınırları yalnızca train verisinden öğrenilir, validation/test verilerine yalnızca uygulanır
- `OneHotEncoder` ile sayısal forma dönüştürülür

## Yapay Eksik Değer Enjeksiyonu

Orijinal veri setinde eksik değer BULUNMAMAKTADIR. Imputation işlemini göstermek amacıyla rastgele seçilen 5 sayısal sütuna sabit `random_state=42` ile ~%1 oranında (her sütunda 5 adet) yapay eksik değer eklenmiştir:

| Sütun | Eksik Sayısı |
|---|---|
| mean perimeter | 5 |
| perimeter error | 5 |
| symmetry error | 5 |
| worst radius | 5 |
| worst fractal dimension | 5 |

Eksik değerler pipeline içinde `SimpleImputer(strategy="median")` ile yalnızca train verisine fit edilerek doldurulur.

## Preprocessing Adımları

Tüm ön işleme adımları sklearn `Pipeline` ve `ColumnTransformer` içinde yürütülmüştür. **Data leakage engellenmiştir**:
- Imputer, OutlierCapper, StandardScaler ve OneHotEncoder yalnızca train verisine **fit** edilir
- Validation ve test verilerine yalnızca **transform** uygulanır
- GridSearchCV her fold'da preprocessing'i fold'un train verisiyle yeniden fit eder

### Pipeline Yapısı

**Logistic Regression ve KNN:**
```
RadiusCategoryTransformer
  -> ColumnTransformer[
       numeric: SimpleImputer(median) -> OutlierCapper -> StandardScaler
       categorical: SimpleImputer(most_frequent) -> OneHotEncoder
     ]
  -> SelectKBest(k=15, score_func=f_classif)
  -> Classifier
```

**Random Forest:**
```
RadiusCategoryTransformer
  -> ColumnTransformer[
       numeric: SimpleImputer(median) -> OutlierCapper
       categorical: SimpleImputer(most_frequent) -> OneHotEncoder
     ]
  -> SelectKBest(k=15, score_func=f_classif)
  -> RandomForestClassifier
```

Random Forest ölçekleme gerektirmediği için numeric boruda StandardScaler bulunmaz.

### Aykırı Değer İşleme

- EDA aşamasında tüm veri setinde IQR yöntemiyle aykırı değer sayıları raporlanır (betimsel)
- Capping sınırları (`OutlierCapper`) yalnızca train verisinden öğrenilir
- IQR çarpanı: 1.5

## Train-Validation-Test Oranları

| Küme | Satır Sayısı | Oran |
|---|---|---|
| Train | 341 | %60 |
| Validation | 114 | %20 |
| Test | 114 | %20 |

Tüm bölme işlemlerinde `stratify` kullanılarak malignant/benign dağılımı korunmuştur.

## Karşılaştırılan Modeller

Üç farklı sınıflandırma modeli eğitilmiştir:

1. **Logistic Regression** (`max_iter=3000`, L2 regularization)
2. **KNN** (`n_neighbors=5`)
3. **Random Forest** (`n_estimators=100`)

## Validation Sonuçları

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9846 |
| KNN | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9728 |
| Random Forest | 0.9737 | 0.9762 | 0.9535 | 0.9647 | 0.9923 |

Üç model de validation setinde aynı Accuracy, Precision, Recall ve F1-Score değerlerine ulaşmıştır. Tüm modellerin F1 ve Recall değerleri eşit olduğu için **Logistic Regression** (en basit ve yorumlanabilir model) seçilmiştir.

Model seçim kriteri: malignant F1-Score, eşitlikte malignant Recall.

## Hiperparametre Ayarlama

En iyi model (Logistic Regression) için `GridSearchCV` uygulanmıştır:

- **CV stratejisi:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- **Scoring:** `f1` (malignant F1-score)
- **En iyi CV F1-score:** 0.9465

### En İyi Parametreler

| Parametre | Değer |
|---|---|
| `classifier__C` | 100 |
| `classifier__penalty` | l2 |
| `classifier__solver` | lbfgs |
| `selector__k` | 25 |

## Test Sonuçları (Seçilen Model: Logistic Regression)

| Metrik | Değer |
|---|---|
| Accuracy | 0.9737 |
| Precision | 0.9535 |
| Recall | 0.9762 |
| F1-Score | 0.9647 |
| ROC-AUC | 0.9954 |

### Confusion Matrix (Test)

| | Tahmin: Benign | Tahmin: Malignant |
|---|---|---|
| **Gerçek: Benign** | 70 | 2 |
| **Gerçek: Malignant** | 1 | 41 |

### Classification Report

| Sınıf | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Benign (0) | 0.99 | 0.97 | 0.98 | 72 |
| Malignant (1) | 0.95 | 0.98 | 0.96 | 42 |

## Önemli Öznitelikler (Logistic Regression Katsayı Yorumu)

| Sıra | Öznitelik | |Katsayı| |
|---|---|---|---|---|
| 1 | num__mean concave points | 6.2993 |
| 2 | num__worst concave points | 5.4623 |
| 3 | num__worst radius | 5.2437 |
| 4 | num__worst concavity | 4.2598 |
| 5 | num__worst perimeter | 4.0407 |
| 6 | num__worst area | 3.1383 |
| 7 | num__mean compactness | 2.8649 |
| 8 | num__mean concavity | 2.8464 |
| 9 | num__mean radius | 2.7537 |
| 10 | cat__radius_category_large | 1.9949 |

En önemli öznitelikler tümör şekil düzensizliği (concave points, concavity), tümör boyutu (radius, perimeter, area) ve üretilen kategorik değişken (radius_category) ile ilişkilidir. Katsayı büyüklükleri, ilgili öznitelikteki bir birimlik artışın malignant olma log-odds'una etkisini gösterir.

## Model Sınırlılıkları

1. **Küçük veri seti:** 569 örnek, daha büyük ve çeşitli verilerde genelleme performansı farklı olabilir
2. **Eğitim amaçlıdır:** Bu model gerçek tıbbi teşhis aracı olarak KULLANILAMAZ
3. **Aykırı değer sınırlandırması:** Kanserli örneklerdeki yüksek değerler gerçek biyolojik sinyal olabilir; capping dikkatli yorumlanmalıdır
4. **False negative riski:** Test setinde 3 malignant vaka gözden kaçırılmıştır (recall = %92.86). Tıbbi uygulamalarda her bir false negative hayati risk taşır

## outputs/ Klasöründeki Dosyalar

| Dosya | Açıklama |
|---|---|
| `class_distribution.png` | Hedef sınıf dağılımı bar grafiği |
| `correlation_heatmap.png` | En yüksek korelasyonlu 20 öznitelik ısı haritası |
| `model_comparison.csv` | Validation karşılaştırma sonuçları tablosu |
| `confusion_matrix.png` | Test confusion matrix görselleştirmesi |
| `feature_importance.png` | En önemli 20 öznitelik (yatay bar grafik) |
| `shap_summary.png` | SHAP beeswarm plot (TreeExplainer) |

## Sonuç Yorumu

Logistic Regression, validation setinde en yüksek F1-skoru ve Recall ile seçilmiştir. Test setinde %97.37 accuracy ve %97.62 malignant recall elde edilmiştir. 42 malignant test örneğinden 41'i doğru tespit edilmiş, yalnızca 1 false negative vaka gözden kaçırılmıştır. Bu durum, modelin malignant vakaları yakalamada oldukça başarılı olduğunu göstermektedir. Ayrıca yalnızca 2 false positive vaka bulunmaktadır (precision: %95.35).

Tümör concave points (çukur noktaları), tümör boyutu (radius, perimeter, area) ve concavity en yüksek katsayı büyüklüğüne sahip özniteliklerdir. Üretilen `radius_category_large` kategorik değişkeni de ilk 10 içinde yer alarak öznitelik mühendisliğinin katkısını göstermiştir.

Logistic Regression'ın en önemli avantajı, katsayılarının doğrudan yorumlanabilir olmasıdır. SHAP LinearExplainer ile yapılan açıklanabilirlik analizi, katsayı yorumlarını desteklemektedir.
