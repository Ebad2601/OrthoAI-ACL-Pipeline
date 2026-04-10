# OrthoAI ACL Injury Risk Prediction Pipeline

**Author:** Muhammad Ebadur Rahman Siddiqui  
**Platform:** OrthoAI Research  
**Tech stack:** Python · pandas · numpy · scikit-learn · matplotlib · seaborn

---

## What this project does

A complete end-to-end machine learning pipeline that predicts ACL injury risk
in competitive athletes from biomechanical and training load data.

**Dataset:** 1,500 athlete-seasons across 5 sports (Football, Basketball, Rugby,
Handball, Volleyball) with 19 features derived from published sports medicine
literature (Meeuwisse et al., 2007; Gabbett, 2016; Bahr & Krosshaug, 2005).

## Project structure

```
orthoai_acl_pipeline/
├── data/
│   └── acl_injury_dataset.csv     ← 1,500 athlete-season records
├── outputs/
│   ├── eda_1_overview_dashboard.png
│   ├── eda_2_correlation_heatmap.png
│   ├── eda_3_boxplots_risk_factors.png
│   └── evaluation_model_dashboard.png
├── models/
│   └── best_model.pkl             ← Serialised best model
├── generate_dataset.py            ← Dataset generation script
├── pipeline.py                    ← Full ML pipeline (production)
├── OrthoAI_ACL_Analysis.ipynb     ← Research notebook
└── README.md
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python generate_dataset.py    # create dataset
python pipeline.py            # run full analysis
jupyter notebook OrthoAI_ACL_Analysis.ipynb   # interactive research notebook
```

## Results

| Model               | CV AUC (5-fold) | Test AUC |
|---------------------|-----------------|----------|
| Logistic Regression | 0.703 ± 0.029   | **0.709** |
| Random Forest       | 0.635 ± 0.028   | 0.626    |
| Gradient Boosting   | 0.656 ± 0.036   | 0.629    |

AUC 0.71 is within the range reported in published sports injury ML literature
(Claudino et al., 2019; Bartlett et al., 2022).

## Key clinical findings

- ACWR >1.5 is the strongest **modifiable** predictor (aligns with Gabbett 2016)
- Knee valgus angle and H:Q ratio contribute independently of training load
- Prior ACL history is the strongest non-modifiable risk factor
