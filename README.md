# Task-5-Decision-Trees-and-Random-Forests

Project Summary: Comparative Classification Analysis on Health Data

This repository contains a comprehensive two-part machine learning project utilizing Python's Scikit-learn and Pandas to build and evaluate classification models on critical health datasets. The primary goal was to explore the capabilities of linear (Logistic Regression) versus non-linear, ensemble methods (Decision Trees and Random Forest) in predicting health outcomes.

Part 1: Logistic Regression for Cancer Diagnosis (Wisconsin Breast Cancer Dataset)

The first phase focused on binary classification using Logistic Regression. The workflow began with meticulous data preparation, involving a 70/30 train/test split and the crucial standardization of features using StandardScaler to ensure optimal model convergence. The trained model was then rigorously evaluated using a suite of metrics including the Confusion Matrix, Precision, Recall, and F1-Score, and the overall ROC-AUC score. A significant component of this section was the exploration of the model's theoretical foundation—the Sigmoid Function—and the practical necessity of threshold tuning to strategically adjust the decision boundary, which is vital in a medical context to minimize False Negatives (missed diagnoses).

Part 2: Tree-Based and Ensemble Methods for Heart Disease Prediction (Heart Disease Dataset)

The second phase shifted to analyzing the Heart Disease Dataset, focusing on interpretability and variance reduction through tree-based models. The project trained an initial, unconstrained Decision Tree Classifier, which was then analyzed for overfitting by comparing training and testing accuracy. This led to regularization via controlling tree depth (max_depth) to find the optimal balance. To further improve stability, an ensemble Random Forest Classifier was implemented and its accuracy compared against the single Decision Tree. The interpretability of the ensemble was highlighted by calculating and visualizing the Feature Importances, identifying the most influential factors in heart disease prediction. The entire modeling process concluded with a robust model validation technique: k-fold Cross-Validation, providing a stable and reliable estimate of the model's real-world performance.

Tools and Technologies Used

Scikit-learn:Core ML framework for models (LogisticRegression, DecisionTreeClassifier, RandomForestClassifier), preprocessing (StandardScaler, train_test_split), evaluation, and cross-validation.

Pandas:Data loading, manipulation, and handling of categorical features (pd.get_dummies).

Matplotlib:Visualization of model performance, including the ROC Curve, Precision-Recall Tradeoff, Feature Importances, and the effect of max_depth.

Graphviz:External tool for rendering and visualizing the structure of the trained Decision Tree.
