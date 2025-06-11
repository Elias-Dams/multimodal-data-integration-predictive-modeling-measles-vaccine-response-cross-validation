# Multimodal Data Integration for Predictive Modelling of Measles Vaccine Response with Cross-Vaccine Marker Validation

## Overview
This repository contains the work presented in the Master's thesis by Elias Dams from the University of Antwerp (2024-2025). The thesis focuses on developing a robust data-driven pipeline to identify stable immune markers predictive of individual variability in response to the Measles vaccine. It also explores the generalizability of these markers to other vaccines, specifically Hepatitis B. 

## Author & Supervision
* **Author:** Elias Dams 
* **Promotor:** Dr. Pieter Meysman 
* **Supervisor:** Fabio Affaticati 
* **Institution:** University of Antwerp, Faculty of Science 
* **Degree:** Master in Computer Science: AI & Data Science 
* **Year:** June 2025 

## Abstract
This study investigated immune markers predictive of individual variability in response to the Measles vaccine component of the Measles, Mumps, and Rubella (MMR) booster in a cohort of 40 adults. Using multimodal data, the primary aim was to identify robust features associated with a binary antibody-based vaccine response (responder vs. non-responder). The research addressed challenges inherent in small (N=40), heterogeneous, high-dimensional datasets, including class imbalance in vaccine response groups. A comprehensive feature identification pipeline was developed and applied. This involved data preprocessing, including PCA-based compression of correlated features and SMOTE-based oversampling for class imbalance on specially partitioned data. Multiple classification algorithms were trained, and feature importance was robustly assessed using aggregated SHAP values from selected models. The stability and directional consistency of these identified features were further validated through an iterative multi-split analysis. The developed pipeline effectively identified stable immune markers for measles vaccine response in a small cohort, providing a principled framework for biomarker discovery in such challenging datasets. While direct cross-vaccine validation proved challenging due to inherent biological and technical differences, partial consistency in some markers suggests conserved immunological mechanisms. The study also highlighted the critical role of SMOTE application for class imbalance, and the pipeline’s capacity to indicate when underlying signals are weak or robust, guiding interpretation in small cohorts. This work offers robust hypotheses for future large-scale validation in vaccine responsiveness. 

## Methodology & Key Features

* **Objective:** To identify stable immune markers predictive of measles vaccine response. 
* **Approach:** A robust predictive modeling pipeline utilizing multimodal data integration. 
* **Data Modalities:**
    * Cytokine concentrations 
    * Standard hematology parameters (cytometry) 
    * T-cell receptor (TCR) breadth and depth metrics (via DETECT predictions for measles-specific TCRs) 
    * Gene expression modules (Blood Gen3-PCA) 
* **Data Preprocessing:**
    * PCA-based compression for highly correlated features 
    * SMOTE-based oversampling to address class imbalance. 
* **Machine Learning Models:** Five classic machine learning classifiers were trained: Logistic Regression, Decision Trees, Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes. 
* **Feature Importance Assessment:** Robustly assessed using aggregated SHapley Additive exPlanations (SHAP) values from selected models. 
* **Model Validation:** Stability and directional consistency of identified features were validated through an iterative multi-split analysis and repeated stratified 5-fold cross-validation. 
* **Cross-Vaccine Validation:** The generalizability of identified markers was investigated using an independent Hepatitis B dataset. 

## Datasets
* **Primary Cohort (Measles):** Data from 40 adults who received an MMR booster. This includes antibody titer measurements at Day 0, 21, 150, and 365. Responders were defined by a significant antibody titer increase ( $\ge120$ mIU/mL) from Day 0 to Day 21. 
* **Validation Cohort (Hepatitis B):** An independent dataset featuring Cytometry and RNA data, mirroring the modalities available in the measles study for cross-vaccine validation. 

## Key Findings
* The developed pipeline successfully identified stable immune markers for measles vaccine response in a small cohort. 
* The pipeline can differentiate between datasets with genuine predictive signals and those driven by random variation, preventing false discoveries in "weak-signal" scenarios. 
* Promising immune features were identified, particularly in lower-dimensional datasets like Clonal Depth and Cytometry. 
* While direct cross-vaccine generalizability was challenging due to inherent biological and technical differences, partial consistency in some markers suggests conserved immunological mechanisms. 
* The study emphasized the critical role of SMOTE for managing class imbalance and the pipeline's capability to indicate the strength of underlying biological signals. 
* This work provides robust hypotheses for future large-scale validation in vaccine responsiveness research. 

## Challenges & Limitations
* **Small Sample Size:** The primary measles dataset comprised only 40 samples (27 for T-cell data), which, despite being representative of contemporary systems immunology studies, limits statistical power and increases overfitting risk. 
* **Heterogeneous Data Integration:** Combining various data types (cytokines, cytometry, TCR, RNA) with different scales and units introduced complexity in normalization and feature selection. 
* **Data Quality:** Noise or missing values in small datasets can disproportionately impact model performance. 
* **High Dimensionality:** Especially in RNA data, the high number of features relative to samples necessitated careful feature selection or dimensionality reduction. 
* **Cross-Vaccine Validation Issues:** Significant biological and technical differences, including notable batch effects in RNA data across studies, restricted direct meaningful comparisons and posed challenges for marker generalizability.  Careful methodology was required to prevent data leakage during feature selection for cross-validation. 

## Acknowledgements
I would like to express my deepest gratitude to my promotor, Dr. Pieter Meysman, and my supervisor, Fabio Affaticati, for their invaluable guidance, insightful feedback, and continued encouragement during our regular meetings. Their expertise and support were instrumental in shaping this thesis and keeping me motivated at every stage. I also acknowledge the use of generative AI tools, specifically ChatGPT and GitHub Copilot, for assistance with text generation and code development throughout the thesis writing process. This assistance was used responsibly for brainstorming, drafting, and refining content, as well as debugging code snippets. All outputs were reviewed, edited, and validated by me. 

## Disclaimer
This document is an examination document that has not been corrected for any errors identified. Without prior written permission of both the supervisor(s) and the author(s), any copying, copying, using or realizing this publication or parts thereof is prohibited. For requests for information regarding the copying and/or use and/or realisation of parts of this publication, please contact to the university at which the author is registered. Prior written permission from the supervisor(s) is also required for the use for industrial or commercial utility of the (original) methods, products, circuits and programs described in this thesis, and for the submission of this publication for participation in scientific prizes or competitions. This document is in accordance with the faculty regulations related to this examination document and the Code of Conduct. The text has been reviewed by the supervisor and the attendant.
