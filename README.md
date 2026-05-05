# IEEE-CIS Fraud Detection

## Kaggle კონკურსის მიმოხილვა

ამ დავალების მთავარი მიზანია თაღლითური ტრანზაქციების იდენტიფიცირება რეალურ სამყაროში არსებული მონაცემების საფუძველზე.მონაცემთა ბაზა მოიცავს ასობით ათას ტრანზაქციას, რომლებიც დაყოფილია ორ ნაწილად: Transaction (ტრანზაქციის დეტალები) და Identity (მომხმარებლზე ინფორმაცია). მთავარი სირთულე არის მონაცემების მასშტაბი და კლასების დისბალანსი.


## ჩემი მიდგომა

1. **Time-based split** — ვალიდაციისთვის ქრონოლოგიური გაყოფა (80/20), რადგან ეს fraud detection-ის რეალობას ასახავს
2. **MLflow + DagsHub** — ყველა ექსპერიმენტის, ჰიპერპარამეტრის და მეტრიკის ავტომატური ლოგვა
3. **Pipeline-ად შენახვა** — საბოლოო მოდელი raw test set-ზე პირდაპირ გაეშვება preprocessing-ის გარეშე
4. **Overfit/Underfit ანალიზი** — სხვადასხვა კონფიგურაციის შედარება, არა მხოლოდ საუკეთესო score-ის ძებნა
5. **სრული training set-ზე გადაწვრთნა** — ექსპერიმენტების შემდეგ final submission pipeline სრულ training data-ზე რეფიტდება

---

## რეპოზიტორიის სტრუქტურა

IEEE-CIS-Fraud-Detection:
- eda-and-observations.ipynb: მონაცემთა წინასწარი ანალიზი, ვიზუალიზაცია და კორელაციების დადგენა
- model_experiment_XGBoost.ipynb      # XGBoost ექსპერიმენტები
- model_experiment_LogisticRegression.ipynb  # Logistic Regression ექსპერიმენტები
- model_inference.ipynb               # საუკეთესო მოდელის inference + submission


## EDA ძირითადი დაკვირვებები

### Fraud Distribution
მონაცემები მკვეთრად დაუბალანსებელია, 3,5 fraud და 96.5 not fraud. ენ ნიშნავს, რომ თუ მოდელი საერთოდ არაფერს არ ისწავლის და და უბრალოდ ყველა ტრანზაქციაზე იტყვის, რომ ის არ არის fraud-ი, მისი სიზუსტე ანუ Accuracy მაინც 96.5% იქნება, თუმცა ასეთი მოდელი სრულიად უსარგებლო იქნება რეალური fraud-ის გამოსავლენად. ეს ნიშნავს, რომ მოდელები accuracy-ის ნაცვლად ROC-AUC-ით უნდა შეფასდეს და class imbalance-იც (scale_pos_weight XGBoost-ში, class_weight=balanced LR-ში) უნდა გავითვალისწინოთ. AU ROC მეტრიკა იყენებს False Positive Rate-ს, რომლის მნიშვნელობაც დამოკიდებულია True Negatives-ის რაოდენობაზე. რადგან ჩვენს მონაცემებში არათაღლითური ტრანზაქციების რაოდენობა ძალიან დიდია (569,877 შემთხვევა), FPR-ის მნიშვნელობა ყოველთვის მცირე რჩება, რაც ხშირად იწვევს AU ROC ქულის "ზედმეტად ოპტიმისტურ" ჩვენებას.
PRC: ასეთი მაღალი დისბალანსის დროს PRC უფრო ობიექტურ სურათს იძლევა, რადგან ის ყურადღებას არ აქცევს True Negatives-ს და ფოკუსირდება მხოლოდ იმაზე, თუ რამდენად ეფექტურად პოულობს მოდელი იშვიათ თაღლითურ შემთხვევებს (Recall) და რამდენად ხშირად ცდება ის (Precision).

<img src="images/output1.png" width="400"/>

### TransactionAmt
Raw distribution არის ძლიერად skewed-ია. Log-transform ნორმალურს ხდის. Fraud ტრანზაქციები ოდნავ სხვაგვარ amount distribution-ს აჩვენებს განსაკუთრებით დაბალ და ძალიან მაღალ თანხებში. Boxplot დიაგრამამ აჩვენა, რომ თაღლითური ტრანზაქციების საშუალო თანხა ოდნავ აღემატება ჩვეულებრივს.

<img src="images/output2.png" width="400"/>

### Missing Values
მონაცემები ხასიათდება NaN მნიშვნელობების მაღალი სიხშირით. 434 სვეტიდან მხოლოდ 20 სვეტს არ აკლია მონაცემები. 214 სვეტში გამოტოვებული მონაცემების წილი 50%-ზე მეტია. 12 სვეტში კი NaN მნიშვნელობების წილი 90%-ს აჭარბებს. ეს ანალიზი გახდა საფუძველი შემდგომი გასუფთავების (Cleaning) სტრატეგიისთვის

<img src="images/output3.png" width="400"/>

### Transaction Hours
Fraud-ის განაწილება დღის განმავლობაში განსხვავდება: ღამის საათებში fraud rate უფრო მაღალია.

### Categorical Features
email domain, ProductCD, DeviceType - fraud rate-ის მიხედვით ამ სვეტებს განსხვავებული პროფილი აქვთ. ეს გვეუბნება, რომ frequency encoding ან target encoding გამოსადეგია.

## Feature Engineering
EDA-ს დაკვირვებებზე დაყრდნობით შევქმენით სხვადასხვა კატეგორიის ნიშნები
კატეგორიული ცვლადების კოდირებისთვის სხვადასხვა მოდელში განსხვავებული მიდგომა გამოვიყენეთ, რადგან ერთი მეთოდი ყველა არქიტექტურისთვის ოპტიმალური არ არის. Logistic Regression-ში Label Encoding გამოვიყენე One-Hot Encoding-ის ნაცვლად მეხსიერების დაზოგვის მიზნით, ვინაიდან ზოგიერთ კატეგორიულ ნიშანს 50-ზე მეტი უნიკალური მნიშვნელობა ჰქონდა და OHE dimension explosion-ს გამოიწვევდა. LightGBM-ში კატეგორიული ნიშნები native categorical-ად მიეცა, რაც ამ ფრეიმვორქის მთავარი უპირატესობაა — ის კარგად ამუშავებს კატეგორიულ ცვლადებს OHE-ს გარეშე. XGBoost-სა და Random Forest-ში Ordinal Encoding გამოვიყენე.

### კატეგორიული ცვლადების გადაყვანა
Label Encoding: გამოყენებულ იქნა კატეგორიული ცვლადებისთვის (მაგ: ProductCD, card4, card6)

სამი მიდგომა გატესტეს:

| მიდგომა | აღწერა | გამოყენება |
|---------|--------|-----------|
| **Ordinal Encoding** | ყველა unique მნიშვნელობა → integer | baseline და Logistic Regression |
| **Frequency Encoding** | კატეგორია → მისი relative frequency training set-ში | `time_amount_freq` feature set |
| **Unseen → -1** | test-ში უცნობი კატეგორია → -1 | ორივე notebook-ში |

### NaN მნიშვნელობების დამუშავება

| სტრატეგია | აღწერა | შედეგი |
|-----------|--------|--------|
| `keep_nan` | NaN უცვლელად რჩება (XGBoost handles natively) | baseline |
| `median` | სვეტის median-ით შევსება | tested |
| `minus999` | -999-ით შევსება (explicit missing signal) | tested |
| `median_indicator` | median + ცალკე binary სვეტი "იყო NaN?" | tested |

რადგან კვლევითმა ანალიზმა (EDA) აჩვენა, რომ 434 სვეტიდან 214-ში ინფორმაციის ნახევარზე მეტი აკლია. ლოგისტიკური რეგრესიისთვის გამოვიყენე მედიანით შევსების (Median Imputation) სტანდარტული მიდგომა, რაც წრფივი მოდელებისთვის მონაცემთა სტაბილურობის გარანტიაა. XGBoost-ის შემთხვევაში კი ჩავატარე უფრო ფართო ექსპერიმენტები, სადაც ერთმანეთს შეედარა მედიანით შევსება, მუდმივი მნიშვნელობით (-999) ჩანაცვლება და ინდიკატორული ცვლადების შექმნა, რომლებიც ცალკე სვეტად აღრიცხავენ, იყო თუ არა კონკრეტული მნიშვნელობა გამოტოვებული. ინდიკატორული ცვლადების გამოყენება განსაკუთრებით ეფექტური აღმოჩნდა, რადგან თაღლითური ტრანზაქციების დროს ხშირად გარკვეული ინფორმაცია განზრახ არის დაფარული, რაც მოდელისთვის მნიშვნელოვან სიგნალს წარმოადგენს. 

### Cleaning მიდგომები

XGBoost მოდელისთვის გავტესტე სვეტების ამოგდების სამი სხვადასხვა ზღვარი, კერძოდ 50%, 75% და 90%, რამაც აჩვენა, რომ ყველაზე ოპტიმალურ შედეგს 50%-იანი ზღვარი იძლევა, როდესაც მოდელიდან 214 ყველაზე ნაკლებად შევსებული სვეტი იფილტრება. ლოგისტიკური რეგრესიის შემთხვევაში შედარებით მსუბუქი 90%-იანი ზღვარი გამოვიყენე, რათა შენარჩუნებულიყო მაქსიმალური ინფორმაცია წრფივი კავშირების დასადგენად. გამოტოვებული რიცხვითი მნიშვნელობების შესავსებად ექსპერიმენტებში გამოვიყენე მედიანით ჩანაცვლება და სპეციალური ინდიკატორული ცვლადების შექმნა, რომლებიც მოდელს მიანიშნებდნენ მონაცემის არარსებობის შესახებ, რაც თაღლითობის გამოვლენისას ხშირად მნიშვნელოვან სიგნალს წარმოადგენს. 

Missing value threshold-ის შედეგები (baseline median imputation-ით):

| Threshold | Dropped | Kept | Train AUC | Val AUC | Gap |
|-----------|---------|------|-----------|---------|-----|
| 50% | 214 | 217 | 0.9094 | 0.8887 | 0.0207 |
| 75% | 208 | 223 | 0.9095 | 0.8864 | 0.0231 |
| 90% | 12 | 419 | 0.9094 | 0.8861 | 0.0233 |

**დასკვნა:** 50% threshold-ი ოდნავ უკეთეს val AUC-ს და უფრო მცირე overfit gap-ს იძლევა. 90% threshold-ი ბევრ სვეტს ინახავს, მაგრამ ეს დამატებითი სვეტები validation-ზე ვერ გვეხმარება. ამიტომ **CHOSEN_THRESHOLD = 0.50** შეირჩა.

---

## Feature Engineering ვარიანტები

| Feature Set | Features | Train AUC | Val AUC | Gap |
|-------------|----------|-----------|---------|-----|
| `baseline` | 217 | 0.9098 | 0.8873 | 0.0226 |
| `time_amount` | 222 | 0.9123 | 0.8850 | 0.0273 |
| `time_amount_freq` | 233 | 0.9145 | 0.8859 | 0.0286 |

`time_amount` feature set-ი ამატებს: `Transaction_hour`, `Transaction_day`, `Transaction_week`, `TransactionAmt_log`, `TransactionAmt_cents`.

`time_amount_freq` ამატებს ასევე frequency encoding-ს კატეგორიულ სვეტებზე.

**დასკვნა:** baseline feature set-ი validation-ზე ოდნავ უკეთესია — engineered features train-ზე უმჯობესდება, მაგრამ validation-ზე ოდნავ მეტ overfit-ს იძლევა. `BEST_FEATURE_SET = 'baseline'`.

---

## Feature Selection
წრფივი მოდელისთვის გამოვიყენე Variance Threshold მეთოდი, რომელმაც ამოაგდო ის ცვლადები, რომელთა ვარიაცია 0.01-ზე ნაკლები იყო, რადგან ასეთი სტატიკური მახასიათებლები ლოგისტიკური რეგრესიისთვის არაინფორმატიულია. XGBoost-ისთვის კი შერჩევა დაეფუძნა მოდელის შიდა Feature Importance რეიტინგებს, სადაც გაიტესტა ტოპ 100, 200 და 300 მახასიათებლის შერჩევა. ანალიზმა აჩვენა, რომ ტოპ 200 მახასიათებლის გამოყენებით მიიღწევა ყველაზე სტაბილური შედეგი და საუკეთესო ბალანსი სიზუსტესა და მოდელის სირთულეს შორის.

| მეთოდი | Features | Train AUC | Val AUC | Gap |
|--------|----------|-----------|---------|-----|
| all features | 217 | 0.9098 | 0.8873 | 0.0226 |
| top 100 importance | 100 | 0.9089 | 0.8865 | 0.0224 |
| **top 200 importance** | **200** | **0.9097** | **0.8874** | **0.0223** |
| top 300 importance | 217 | 0.9092 | 0.8836 | 0.0256 |

top 200 importance feature-ები ოდნავ უკეთეს val AUC-ს და ნაკლებ overfit gap-ს იძლევა.
---

## Training

### ტესტირებული XGBoost მოდელები

| Config | Train AUC | Val AUC | Gap | Behavior |
|--------|-----------|---------|-----|----------|
| baseline | 0.9097 | 0.8874 | 0.0223 | generalization |
| **regularized** | **0.9154** | **0.8918** | **0.0236** | generalization |
| deeper_overfit_check | 0.9891 | 0.9018 | 0.0873 | **overfit** |
| shallow_underfit_check | 0.8616 | 0.8491 | 0.0125 | **underfit** |

**Overfit ანალიზი (`deeper_overfit_check`):**
`max_depth=8`, `min_child_weight=1` — ძალიან ღრმა ხეები training data-ს noise-ს ზეპირად ისწავლის. train_auc=0.9891 vs val_auc=0.9018 — gap=0.0873 ამ მოდელის გამოუყენებლობაზე მეტყველებს Kaggle submission-ში.

**Underfit ანალიზი (`shallow_underfit_check`):**
`max_depth=2`, `min_child_weight=10`, `reg_lambda=5.0` — ძალიან მარტივი მოდელი, ვერ სწავლობს fraud pattern-ებს. train_auc=0.8616 val_auc=0.8491 — ორივე დაბალია.

**Hyperparameter ოპტიმიზაციის მიდგომა:**
Manual grid search: სხვადასხვა `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `reg_alpha`, `reg_lambda` კომბინაციები. Final model შეირჩა conservative selection score-ით: `val_auc - penalty(gap > 0.05)`.

**საბოლოო მოდელის შერჩევის დასაბუთება:**
`deeper_overfit_check`-ს უმაღლესი raw val_auc (0.9018) აქვს, მაგრამ gap=0.0873 ძლიერი overfitting-ის ნიშანია. `regularized` კონფიგი val_auc=0.8918-ით და gap=0.0236-ით ბევრად უფრო სტაბილური generalization-ს ავლენს. **Final registered model: `regularized` XGBoost Pipeline.**

### Logistic Regression — Linear Baseline

| Config | Train AUC | Val AUC | Gap |
|--------|-----------|---------|-----|
| Underfit (C=0.0001) | ~0.78 | ~0.77 | ~0.01 |
| Overfit check (C=1000) | ~0.82 | ~0.81 | ~0.01 |
| Tuned (best C) | ~0.82 | ~0.81 | ~0.01 |

Logistic Regression მოსალოდნელად სუსტია ამ dataset-ზე — IEEE-CIS-ის V*, C*, D* feature-ები strongly non-linear interaction-ებს შეიცავს, რომელსაც linear მოდელი ვერ დაიჭერს.

### Time-based Cross Validation (XGBoost)

`TimeSeriesSplit(n_splits=3)` — სრული training set-ზე:

| Fold | Train AUC | Val AUC |
|------|-----------|---------|
| 1 | 0.9345 | 0.8782 |
| 2 | 0.9191 | 0.8864 |
| 3 | 0.9096 | 0.8877 |
| **Mean** | — | **0.8841** |
| **Std** | — | **0.0042** |

CV std=0.0042 — სტაბილური მოდელი. Fold 1-ის მაღალი train AUC (0.9345) fold size-ის განსხვავებით აიხსნება.

---

## MLflow Tracking

**DagsHub Experiments:** [ejoba22/IEEE-CIS-Fraud-Detection MLflow](https://dagshub.com/ejoba22/IEEE-CIS-Fraud-Detection.mlflow)

### Experiment სტრუქტურა

```
XGBoost_Training/
  ├── XGBoost_Cleaning_threshold_50pct
  ├── XGBoost_Cleaning_threshold_75pct
  ├── XGBoost_Cleaning_threshold_90pct
  ├── XGBoost_Feature_Engineering_baseline
  ├── XGBoost_Feature_Engineering_time_amount
  ├── XGBoost_Feature_Engineering_time_amount_freq
  ├── XGBoost_Feature_Selection_all_features
  ├── XGBoost_Feature_Selection_top_100_importance
  ├── XGBoost_Feature_Selection_top_200_importance
  ├── XGBoost_Feature_Selection_top_300_importance
  ├── XGBoost_CrossValidation_TimeSeriesSplit
  ├── XGBoost_Training_baseline
  ├── XGBoost_Training_regularized
  ├── XGBoost_Training_deeper_overfit_check
  ├── XGBoost_Training_shallow_underfit_check
  └── XGBoost_Final_Pipeline_Register

LogisticRegression_Training/
  ├── LogisticRegression_Cleaning
  ├── LogisticRegression_Feature_Engineering
  ├── LogisticRegression_Feature_Selection
  ├── LogisticRegression_Training_underfit_strong_regularization
  ├── LogisticRegression_Training_weak_regularization_overfit_check
  ├── LogisticRegression_Training_tuned_C0.01
  ├── LogisticRegression_Training_tuned_C0.1
  ├── LogisticRegression_Training_tuned_C1.0
  ├── LogisticRegression_Training_tuned_C10.0
  └── LogisticRegression_Final_Pipeline_Register
```

### ჩაწერილი მეტრიკები

| მეტრიკა | აღწერა |
|---------|--------|
| `train_auc` | Training set-ზე ROC-AUC |
| `val_auc` | Validation set-ზე ROC-AUC (time-based split) |
| `overfit_gap` | train_auc − val_auc |
| `selection_score` | val_auc − penalty(gap > 0.05 × 0.25) |
| `cv_mean_auc` | TimeSeriesSplit cross-validation mean AUC |
| `cv_std_auc` | TimeSeriesSplit cross-validation std AUC |

### Model Registry

| Registry Name | Architecture | Val AUC |
|--------------|-------------|---------|
| `IEEE_CIS_XGBoost_Best` | XGBoost Pipeline (regularized) | 0.8918 |
| `IEEE_CIS_LogisticRegression_Best` | Logistic Regression Pipeline | ~0.81 |

**საუკეთესო მოდელი:** `IEEE_CIS_XGBoost_Best` — `model_inference.ipynb`-ში Registry-დან ჩამოიტვირთება, **სრულ training set-ზე** გადაიწვრთნება და test set-ზე prediction-ს გააკეთებს Kaggle submission-ისთვის.
