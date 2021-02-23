# hd_knn_tree
Decision Tree and K-Nearest Neighbors analysis of Heart Disease dataset using RStudio. Also compare with [Logistic Regression](https://github.com/danypark91/hd_log_reg) to figure which is better model to predict the dataset.

### Tech/Framework used
* Rstudio
* Rmarkdown

### RStudio Library used
* library(caTools)
* library(class)
* library(kknn)
* library(caret)
* library(ROCR)

### Installation of R packages
`rpack <- c("kknn", "caret", "class","caTools", "ROCR")`

`install.packages(rpack)`

### Dataset
The [original dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from UCI contained 76 attributes which represent a patient's condition. The dataset for this article is from [Kaggle - Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci). The subset of 14 attributes with every incident represents a patient.

### Project Description
This project is to apply decision tree and k-nearest neighbor to the dataset. It begins with the importation of the dataset from the local device and checks if it requires data cleansing. The cleansed data divides into train and test sets with a ratio of 3 to 1. The best-fit model gets derived by using train_df. The model undergoes statistical tests to determine scientific accuracy. The model is applied to the test_df to check the predictability of the models. Finally, the predictability of the discovered models are compared with the logistic regression model.
This project does not consist data cleansing and visualization. The [hd_log_reg](https://github.com/danypark91/hd_log_reg) notebook has two of required steps for the same dataset used for this project. 

### Reference
* [Classification of Decision Tree](https://pages.mtu.edu/~shanem/psy5220/daily/Day12/classification.html)
* [Classification of KNN](https://pages.mtu.edu/~shanem/psy5220/daily/Day13/treesforestsKNN.html)
* James, G., Witten, D., Hastie, T., & Tibshirani, R. (2017). An introduction to statistical learning with applications in R. New York, N.Y: Springer.
