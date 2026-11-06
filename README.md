# Delhivery Last-Mile Delay Intelligence Pipeline

**Program:** Artificial Intelligence and Machine Learning Certification Program  
**Institution:** E&ICT Academy, IIT Roorkee  
**Project:** Delhivery Last-Mile Delay Intelligence Pipeline  
**Technology:** Python, Pandas, NumPy, Scikit-learn, Joblib

---

## 1. Project Overview

The Delhivery Last-Mile Delay Intelligence Pipeline is an end-to-end machine learning project designed to predict shipment delivery times, identify shipments that may miss their promised delivery deadlines, and classify delivery hubs based on operational performance.

The project uses synthetic shipment data representing 24 delivery hubs. It includes data cleaning, exploratory data analysis, regression modelling, classification modelling, hub segmentation, model packaging, and predictions for new shipments.

### Project Objectives

- Clean and preprocess historical shipment data.
- Explore shipment delays and operational patterns.
- Predict delivery hours using regression models.
- Identify shipments at risk of missing delivery promises.
- Segment delivery hubs based on operational performance.
- Package the final machine learning model for reuse.
- Generate actionable recommendations for new shipments.

---

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.15.0 | Programming language |
| Pandas 3.1.1 | Data manipulation |
| NumPy 2.5.4 | Numerical operations |
| Scikit-learn 1.9.1 | Machine learning |
| Joblib 1.6.0 | Model serialization |
| Matplotlib | Data visualization |

All required package versions are pinned in `requirements.txt`.

---

## 3. Project Structure

```text
Delhivery_Delay_Intelligence/
│
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
│
├── models/
│   ├── delay_model.pkl
│   ├── delay_threshold.csv
│   ├── feature_order.csv
│   └── impute_values.csv
│
├── outputs/
│   ├── hub_profile.csv
│   ├── hub_segments.csv
│   ├── new_shipment_actions.csv
│   ├── part_a_checks.csv
│   ├── part_a_correlations.csv
│   ├── part_a_delay_by_group.csv
│   ├── part_b_checks.csv
│   ├── part_b_cv.csv
│   ├── part_b_gb_learning_rate.csv
│   ├── part_b_metrics.csv
│   ├── part_b_tree_depth.csv
│   ├── part_c_feature_scores.csv
│   ├── part_c_final_config.csv
│   ├── part_c_final_test.csv
│   ├── part_c_models.csv
│   ├── part_c_selectk.csv
│   ├── part_c_thresholds.csv
│   ├── part_c_tuning.csv
│   ├── part_d_reload_check.csv
│   ├── part_d_separation.csv
│   ├── segment_profile.csv
│   └── tier_vs_segment_mismatch.csv
│
├── src/
│   └── shipment_prep.py
│
├── generate_data.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── package_model.py
├── predict_delay.py
├── requirements.txt
└── README.md
```

---

## 4. Dataset Description

The project uses synthetic shipment data generated through `generate_data.py`.

| Dataset | Description | Records |
|---|---|---:|
| hubs.csv | Hub information | 24 |
| raw_shipments.csv | Historical shipment records | 1,224 |
| new_shipments.csv | New shipments for prediction | 48 |
| modelling_table.csv | Cleaned and engineered dataset | 1,200 |

### Delivery Promise Rules

| Service Type | Promised Delivery Time |
|---|---:|
| Express | 36 hours |
| Standard | 60 hours |
| Economy | 96 hours |

A shipment is classified as delayed when:

`delivery_hours > promised_hours`

---

## 5. How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Generate Dataset

```bash
python generate_data.py
```

Expected output:

```text
24 1224 48
```

### Step 3: Run Part A – Data Cleaning and EDA

```bash
python part_a_clean_eda.py
```

### Step 4: Run Part B – Regression

```bash
python part_b_regression.py
```

### Step 5: Run Part C – Classification

```bash
python part_c_classification.py
```

### Step 6: Run Part D – Hub Segmentation

```bash
python part_d_segments.py
```

### Step 7: Package the Final Model

```bash
python package_model.py
```

### Step 8: Predict New Shipments

```bash
python predict_delay.py
```

The final predictions are saved in:

`outputs/new_shipment_actions.csv`

---

## 6. Part A – Data Cleaning and Exploratory Data Analysis

### Data Cleaning

The historical shipment dataset initially contained 1,224 records.

The following preprocessing operations were performed:

- Removed duplicate shipment records.
- Standardized service type names.
- Handled missing payment values.
- Converted declared values into numeric INR values.
- Converted dispatch timestamps into datetime format.
- Identified invalid scanner weight values.
- Replaced missing numerical values using median imputation.
- Created delivery promise and operational features.

### Data Quality Results

| Metric | Result |
|---|---:|
| Raw rows | 1,224 |
| Duplicate rows removed | 24 |
| Rows after deduplication | 1,200 |
| Missing distance values | 36 |
| Missing parcel weight values | 24 |
| Invalid sentinel weights | 5 |
| Missing declared values | 12 |
| Missing payment values | 18 |

### Feature Engineering

The modelling dataset contains 17 model features, including:

- promised_hours
- distance_km
- parcel_weight_kg
- log_declared_value
- n_stops
- rain_flag
- sale_event_flag
- is_cod
- is_weekend
- dispatch_hour
- hub_tier
- hub_daily_capacity
- label_print_batch
- scanner_build_no
- tape_roll_no
- sticker_serial_mod
- shelf_slot_no

### Part A Results

| Metric | Result |
|---|---:|
| Modelling table rows | 1,200 |
| Modelling table columns | 28 |
| Delayed shipments | 278 |
| Delay rate | 23.17% |
| Delivery-hour outliers | 1 |

The delay rate indicates that approximately 23% of historical shipments missed their promised delivery times.

---

## 7. Part B – Delivery Time Prediction Using Regression

### Objective

The objective of Part B is to predict the number of hours required to deliver a shipment.

### Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

### Dataset Split

| Dataset | Records |
|---|---:|
| Training | 840 |
| Validation | 180 |
| Testing | 180 |

### Validation Results

| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression | 4.8912 | 0.9313 |
| Decision Tree | 6.2796 | 0.8867 |
| Random Forest | 5.4680 | 0.9141 |
| Gradient Boosting | 4.9165 | 0.9306 |

### Final Model

Linear Regression was selected because it achieved the lowest validation RMSE.

### Test Results

| Metric | Result |
|---|---:|
| MAE | 3.5671 |
| MSE | 20.4624 |
| RMSE | 4.5235 |
| R² | 0.9493 |

The model achieved an R² score of 0.9493 on the test dataset, indicating strong predictive performance.

### Hyperparameter Analysis

Decision tree depth and Gradient Boosting learning rate were evaluated.

- Tree depth 6 provided the best validation result among the tested depths.
- Unlimited tree depth showed signs of overfitting.
- Learning rate 0.1 performed best among the tested Gradient Boosting configurations.
- Cross-validation supported the model evaluation process.

---

## 8. Part C – Proactive Delay Classification

### Objective

The objective is to classify shipments as either delayed or on time before the delivery promise is missed.

### Models Evaluated

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Baseline classifier

### Validation Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| Decision Tree | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| Random Forest | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

### Feature Selection

SelectKBest was used to identify the most useful seven features.

The top features included:

1. distance_km
2. hub_tier
3. hub_daily_capacity
4. n_stops
5. promised_hours
6. rain_flag
7. sale_event_flag

### Final Classification Configuration

- Algorithm: Random Forest Classifier
- Number of estimators: 100
- Maximum depth: None
- Minimum samples per leaf: 1
- Selected features: 7
- Classification threshold: 0.50

The threshold was selected using validation recall requirements.

### Final Test Results

| Metric | Result |
|---|---:|
| Accuracy | 0.8667 |
| Precision | 0.6889 |
| Recall | 0.7561 |
| F1-score | 0.7209 |

### Confusion Matrix

| | Predicted On Time | Predicted Delayed |
|---|---:|---:|
| Actual On Time | 125 | 14 |
| Actual Delayed | 10 | 31 |

The classification model identifies shipments that may miss their promised delivery deadlines and supports proactive intervention.

---

## 9. Part D – Hub Segmentation

### Objective

The objective is to group delivery hubs according to operational performance and identify hubs that require attention.

K-Means clustering was used to segment the 24 hubs.

### Cluster Evaluation

Different cluster counts were evaluated using across-cluster and within-cluster separation.

| Number of Clusters | Separation Ratio |
|---|---:|
| 2 | 2.893964 |
| 3 | 5.392845 |
| 4 | 6.304705 |
| 5 | 6.886628 |

Three clusters were selected for operational usefulness and manageable group sizes.

### Final Hub Segments

| Segment | Hubs | Average Delivery Hours | Delay Rate |
|---|---:|---:|---:|
| Standard | 7 | 47.2396 | 2.57% |
| Watch | 8 | 56.8137 | 15.50% |
| Escalate | 9 | 64.6504 | 46.00% |

### Operational Recommendations

**Standard**
- Maintain current operational practices.
- Continue routine monitoring.
- Preserve existing delivery processes.

**Watch**
- Monitor shipment delays regularly.
- Review staffing and daily capacity.
- Investigate recurring operational issues.

**Escalate**
- Prioritize delayed shipments.
- Review hub capacity and delivery routes.
- Escalate recurring problems to hub managers.
- Consider operational improvements and resource allocation.

### Tier vs Segment Mismatches

Four hubs showed differences between their original tier and the data-driven segment:

| Hub | City | Actual Segment | Expected Segment |
|---|---|---|---|
| HUB07 | Kolkata | Watch | Standard |
| HUB13 | Nagpur | Escalate | Watch |
| HUB15 | Kochi | Escalate | Watch |
| HUB22 | Dehradun | Watch | Escalate |

These mismatches can help management identify hubs whose actual operational performance differs from their assigned tier.

---

## 10. Part E – Model Packaging and New Shipment Predictions

### Model Packaging

The final classification model was saved using Joblib.

Generated model artifacts:

- `models/delay_model.pkl`
- `models/delay_threshold.csv`
- `models/feature_order.csv`
- `models/impute_values.csv`

The packaging process stores the model, selected feature order, classification threshold, and training imputation values.

### Model Reload Verification

The saved model was reloaded and tested.

| Metric | Result |
|---|---:|
| Test rows scored | 180 |
| Maximum probability difference | 0.0000 |
| Classification threshold | 0.50 |

The model reload verification passed.

### New Shipment Prediction

The final prediction pipeline processes 48 new shipments.

| Metric | Result |
|---|---:|
| Total new shipments | 48 |
| Flagged shipments | 8 |
| Not flagged shipments | 40 |
| Unique shipment IDs | 48 |
| Missing output values | 0 |

### Recommended Actions

| Action | Shipment Count |
|---|---:|
| No action | 40 |
| Escalate to hub manager | 6 |
| Proactive customer call | 2 |

The output file is:

`outputs/new_shipment_actions.csv`

The pipeline assigns actions based on the predicted delay flag and the operational hub segment.

---

## 11. Production Deployment Plan

The pipeline can be extended into a production-ready logistics intelligence system.

### Proposed Workflow

1. Collect shipment booking and dispatch information.
2. Validate and clean incoming shipment data.
3. Generate operational and shipment features.
4. Predict delivery hours.
5. Calculate delay probability.
6. Identify shipments at risk of missing promises.
7. Assign operational actions.
8. Monitor hub performance.
9. Retrain models using updated delivery data.

### Monitoring Recommendations

The following should be monitored regularly:

- Model prediction accuracy.
- Delay classification precision and recall.
- Shipment delay rate.
- Hub-level operational performance.
- Missing and invalid incoming data.
- Changes in shipment and hub characteristics.

Models should be retrained when performance declines or when significant changes occur in operational data.

---

## 12. Limitations

- The project uses synthetic shipment data rather than actual Delhivery operational records.
- Model performance may differ when applied to real-world shipment data.
- External factors such as traffic, road closures, and unexpected weather events may affect delivery times.
- Hub segments should be reviewed periodically as operational conditions change.
- Production deployment would require integration with real-time logistics systems and monitoring infrastructure.

---

## 13. Conclusion

The Delhivery Last-Mile Delay Intelligence Pipeline demonstrates an end-to-end machine learning workflow for logistics operations.

The project successfully covers data cleaning, exploratory analysis, delivery-time regression, proactive delay classification, hub segmentation, model packaging, and new shipment predictions.

The regression model achieved a test R² score of 0.9493, while the classification model achieved a test accuracy of 0.8667.

The final pipeline processed 48 new shipments and generated actionable recommendations for operational teams.

This project demonstrates how artificial intelligence and machine learning can support better delivery planning, proactive customer communication, and improved last-mile logistics operations.

---

**Project completed as part of the Artificial Intelligence and Machine Learning Certification Program, E&ICT Academy, IIT Roorkee.**
