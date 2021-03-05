hd\_knn\_tree
================
Dany Park
01/03/2021

# K-nearest neighbor and Decision Tree

This project is to apply K-Nearest Neighbor and Decision Tree to the
heart disease dataset and apply fitted model to predict the potential
patient. Also, the models are compared with the Logistic Regression for
their accuracy and predictability.

The
[hd\_log\_reg](https://github.com/danypark91/hd_log_reg/blob/main/hd_log_reg_rmarkdown.md)
already consists extensive explanation of the dataset. The project
includes data visualization of the same dataframe. It will walk through
the step-by-step procedure of the regression analysis and the
performance of the predicted model.

## 1. K-nearest neighbor

### 1-1. Overview of KNN

K-nearest neighbors(KNN) is considered to be one of the most simplest
and well-known non-parametric methods. The algorithm does not assume the
parametric form, which allows more flexible approach to perform
analysis. The classification method indentifies the given K points that
are closest to the training independent variable x0. Then the
conditional probability for a particular class is estimated. The largest
probability is used for KNN to apply Bayes Rule and classify the test
observations.

![knn\_example](https://i.imgur.com/XXWScgF.png)

For example, assume that the K is chosen to be 3. Within the distanced
boundary, two blue points are captured along with the orange point. The
estimate probability for the blue equals to 2/3 and orange to 1/3. Then
the algorithm predicts the boundary’s class as blue. The right-hand side
presents the decision boundary of all possible values of x0 with
applying KNN algorithm and k=3.

### 1-2. Importation and Alteration of Data

Before proceeding straight into the algorithm, I imported the project’s
dataframe. Like the previous logistic regression project, the erronous
attribute name was corrected. However this time, the `knn` function
required the only response variable as a factor(categorical variable).

Also, prior to the analysis, normalization of dependent variable was
conducted to equalize the weight and range. The `normalize` function
helped to acquire the condition. The normalized dataset was divided into
two sets: `train_df` was used to apply and train the `knn` alogorithm
and the measure of predictability utlized the `test_df`.

    ##         age sex        cp  trestbps      chol fbs restecg   thalach exang
    ## 1 0.7083333   1 1.0000000 0.4811321 0.2442922   1     0.0 0.6030534     0
    ## 2 0.1666667   1 0.6666667 0.3396226 0.2831050   0     0.5 0.8854962     0
    ## 3 0.2500000   0 0.3333333 0.3396226 0.1780822   0     0.0 0.7709924     0
    ## 4 0.5625000   1 0.3333333 0.2452830 0.2511416   0     0.5 0.8167939     0
    ## 5 0.5833333   0 0.0000000 0.2452830 0.5205479   0     0.5 0.7022901     1
    ##      oldpeak slope ca      thal
    ## 1 0.37096774     0  0 0.3333333
    ## 2 0.56451613     0  0 0.6666667
    ## 3 0.22580645     1  0 0.6666667
    ## 4 0.12903226     1  0 0.6666667
    ## 5 0.09677419     1  0 0.6666667

    ##         age sex        cp  trestbps      chol fbs restecg   thalach exang
    ## 1 0.7083333   1 1.0000000 0.4811321 0.2442922   1     0.0 0.6030534     0
    ## 2 0.1666667   1 0.6666667 0.3396226 0.2831050   0     0.5 0.8854962     0
    ## 3 0.2500000   0 0.3333333 0.3396226 0.1780822   0     0.0 0.7709924     0
    ## 4 0.5625000   1 0.3333333 0.2452830 0.2511416   0     0.5 0.8167939     0
    ## 5 0.5833333   0 0.0000000 0.2452830 0.5205479   0     0.5 0.7022901     1
    ##      oldpeak slope ca      thal target
    ## 1 0.37096774     0  0 0.3333333      1
    ## 2 0.56451613     0  0 0.6666667      1
    ## 3 0.22580645     1  0 0.6666667      1
    ## 4 0.12903226     1  0 0.6666667      1
    ## 5 0.09677419     1  0 0.6666667      1

### 1-3. Selection of K

As the algorithm is based on the distance based on the value of K, it is
extremely important to choose appropriate value. The below image
illustrates the difference betwen k=3 and k=5. The result of the choice
between those values could significantly differ from one another. In
order to choose the right K, `knn` should be performed multiple times
and choose the K that has the least errors.

![Difference](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/330px-KnnClassification.svg.png)

There are couple of points to consider:

-   K is a positive integer
-   K -&gt; 1, less stable prediction
-   As K increases, prediction becomes more stable. However, if the
    error increases, then rerun of `knn` could stop
-   If tiebreaking within the range of K, then choose odd number

The below run is the sample run of KNN with K=15. The rate of error of
prediction on the `test_df` is as low as 0.2209. The result can be
considered as an accurate model. However, as stated above, `knn` should
be performed multiple times with different K values to determine the
best-fit model.

    ##       
    ## knn_15  0  1
    ##      0 28  8
    ##      1 11 39

    ## [1] "Error:  0.2209"

To minimize the effort, the list of error based on the value of k was
created. For-loop command helped to populate the sequential list of
error for K between 1 to 15. The graph represents the list, `knn-err`,
and the value of K. As the trend suggests, error rate decreases
significantly after k=6 and bounces back at 10. 8 and 9 are the most
accurate models for the dataframe. However, as the tiebreaking rule
suggests, 9 is chosen to proceed further steps for the analysis.

![](hd_knn_tree_files/figure-gfm/For%20Loop%20KNN%20from%201%20to%2015-1.png)<!-- -->
The exact rate of error of the K=9 model is 0.1744 which lower than the
k=15 model. Of course the accuracy of the model compare to the `test_df`
reponse variable is 0.8256.

    ## [1] "Accuracy of the Model :  0.8256"

    ## [1] "Error of the Model :  0.1744"

### 1-4. Prediction and Performance Measure

As we dicovered the best-fit model of `KNN`, we should examine the
model’s predictability and its performance. The most commonly used
technics are Confusion Matrix and Receiver Operating Characteristic
Curve. A confusion matrix is a table used to exhibit the classification
result on the test dataset. It contains two dimensions, ‘actual’ and
‘predicted’, and the cells are determined by the number of categories in
a response variable. The below image explains the meaning of each cell
and significant metrics.

![Confusion
Matrix](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)

The confusion matrix states that the accuracy of the 9-nearest neighbor
for the dataframe is 0.8256 with the 95% confidence interval of 0.7287
and 0.8990. The prediction result of the model is very promising in
terms of the predictability.

The sensitivity is 0.8511, which means that out of 48 patients who
suffered the heart disease, 40 patients were correctly diagnosed. The
specificity score is 0.7949. Among the 39 patients who did not carry the
heart disease, 31 patients were successfully categorized. The error of
the model is 0.1744: 8 patients are categorized as Type I error where as
7 patients suffered Type II error. As the dataframe is related to health
issues, Type II error could cause a devastating result.

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 31  7
    ##          1  8 40
    ##                                          
    ##                Accuracy : 0.8256         
    ##                  95% CI : (0.7287, 0.899)
    ##     No Information Rate : 0.5465         
    ##     P-Value [Acc > NIR] : 4.766e-08      
    ##                                          
    ##                   Kappa : 0.6473         
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.8511         
    ##             Specificity : 0.7949         
    ##          Pos Pred Value : 0.8333         
    ##          Neg Pred Value : 0.8158         
    ##              Prevalence : 0.5465         
    ##          Detection Rate : 0.4651         
    ##    Detection Prevalence : 0.5581         
    ##       Balanced Accuracy : 0.8230         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

![ROC
Curve](https://ars.els-cdn.com/content/image/3-s2.0-B9780128030141000029-f02-01-9780128030141.jpg)

Another way to measure the predictability of the model is by deriving
the ROC curve and AUC score. It is an excellent tool to graphically
repsent the predictability of a binary classification. A ROC plots True
Positive Rate vs False Positive Rate at different classification
thresholds. Lower the classification threshold will result more items as
positive. More the curve close to the blue line, more the accurate
prediction is. The ROC curve below shows that it is close to the maximum
plot that a ROC could be.

Although ROC visualize the performance of the predicted model, it is
very diffcult to quantify. AUC provides an aggregate measure of
performance for all possible classification threshold. It measures the
quality of the model’s prediction irrespecitible to the chosen
classification thresold. For KNN, AUC score is 0.9026 hiwhc is very
close to 1.00. It is an evidence that the model’s prediciton is
statistically signifcant.

    ## 
    ## Attaching package: 'kknn'

    ## The following object is masked from 'package:caret':
    ## 
    ##     contr.dummy

![](hd_knn_tree_files/figure-gfm/KNN%20ROC%20and%20AUC-1.png)<!-- -->

    ## [1] 0.9026187

## 2. Decision Tree

    ## 'data.frame':    303 obs. of  14 variables:
    ##  $ age     : int  63 37 41 56 57 57 56 44 52 57 ...
    ##  $ sex     : Factor w/ 2 levels "0","1": 2 2 1 2 1 2 1 2 2 2 ...
    ##  $ cp      : Factor w/ 4 levels "0","1","2","3": 4 3 2 2 1 1 2 2 3 3 ...
    ##  $ trestbps: int  145 130 130 120 120 140 140 120 172 150 ...
    ##  $ chol    : int  233 250 204 236 354 192 294 263 199 168 ...
    ##  $ fbs     : Factor w/ 2 levels "0","1": 2 1 1 1 1 1 1 1 2 1 ...
    ##  $ restecg : Factor w/ 3 levels "0","1","2": 1 2 1 2 2 2 1 2 2 2 ...
    ##  $ thalach : int  150 187 172 178 163 148 153 173 162 174 ...
    ##  $ exang   : Factor w/ 2 levels "0","1": 1 1 1 1 2 1 1 1 1 1 ...
    ##  $ oldpeak : num  2.3 3.5 1.4 0.8 0.6 0.4 1.3 0 0.5 1.6 ...
    ##  $ slope   : Factor w/ 3 levels "0","1","2": 1 1 3 3 3 2 2 3 3 3 ...
    ##  $ ca      : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ thal    : Factor w/ 4 levels "0","1","2","3": 2 3 3 3 3 2 3 4 4 3 ...
    ##  $ target  : int  1 1 1 1 1 1 1 1 1 1 ...

    ## 
    ## Classification tree:
    ## rpart(formula = as.factor(target) ~ ., data = train_dt_df, method = "class")
    ## 
    ## Variables actually used in tree construction:
    ## [1] age     ca      exang   thal    thalach
    ## 
    ## Root node error: 99/217 = 0.45622
    ## 
    ## n= 217 
    ## 
    ##         CP nsplit rel error  xerror     xstd
    ## 1 0.464646      0   1.00000 1.00000 0.074113
    ## 2 0.060606      1   0.53535 0.54545 0.064332
    ## 3 0.050505      3   0.41414 0.59596 0.066205
    ## 4 0.020202      4   0.36364 0.62626 0.067220
    ## 5 0.015152      5   0.34343 0.62626 0.067220
    ## 6 0.010000      7   0.31313 0.64646 0.067853

![](hd_knn_tree_files/figure-gfm/Fitting%20the%20model-1.png)<!-- -->

    ## n= 217 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ## 1) root 217 99 1 (0.4562212 0.5437788)  
    ##   2) thal=0,1,3 98 26 0 (0.7346939 0.2653061) *
    ##   3) thal=2 119 27 1 (0.2268908 0.7731092) *

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 30  9
    ##          1  9 38
    ##                                          
    ##                Accuracy : 0.7907         
    ##                  95% CI : (0.6895, 0.871)
    ##     No Information Rate : 0.5465         
    ##     P-Value [Acc > NIR] : 2.073e-06      
    ##                                          
    ##                   Kappa : 0.5777         
    ##                                          
    ##  Mcnemar's Test P-Value : 1              
    ##                                          
    ##             Sensitivity : 0.8085         
    ##             Specificity : 0.7692         
    ##          Pos Pred Value : 0.8085         
    ##          Neg Pred Value : 0.7692         
    ##              Prevalence : 0.5465         
    ##          Detection Rate : 0.4419         
    ##    Detection Prevalence : 0.5465         
    ##       Balanced Accuracy : 0.7889         
    ##                                          
    ##        'Positive' Class : 1              
    ## 

``` r
#ROC Curve and AUC Score
df_dt_prediction <- prediction(df_dt_model_fit, as.factor(test_dt_df$target))
df_dt_performance <- performance(df_dt_prediction, measure = "tpr", x.measure = "fpr")
df_dt_roc <- plot(df_dt_performance, col="Red",
                  main="ROC Curve - Decision Tree",
                  xlab="False Positive Rate",
                  ylab="True Positive Rate")+
  abline(a=0, b=1, col="Grey", lty=2)+
  abline(v=0, h=1, col="Blue", lty=3)+
  plot(df_dt_performance, col="Red",add=TRUE)
```

![](hd_knn_tree_files/figure-gfm/DT%20ROC%20and%20AUC-1.png)<!-- -->

``` r
df_dt_auc <- performance(df_dt_prediction, measure="auc")
df_dt_auc <- df_dt_auc@y.values[[1]]
df_dt_auc
```

    ## [1] 0.7888707

## 3. Comparison with other Model

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  0  1
    ##          0 31 11
    ##          1  8 36
    ##                                           
    ##                Accuracy : 0.7791          
    ##                  95% CI : (0.6767, 0.8614)
    ##     No Information Rate : 0.5465          
    ##     P-Value [Acc > NIR] : 6.351e-06       
    ##                                           
    ##                   Kappa : 0.5572          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.6464          
    ##                                           
    ##             Sensitivity : 0.7660          
    ##             Specificity : 0.7949          
    ##          Pos Pred Value : 0.8182          
    ##          Neg Pred Value : 0.7381          
    ##              Prevalence : 0.5465          
    ##          Detection Rate : 0.4186          
    ##    Detection Prevalence : 0.5116          
    ##       Balanced Accuracy : 0.7804          
    ##                                           
    ##        'Positive' Class : 1               
    ## 

![](hd_knn_tree_files/figure-gfm/Logistic%20Regression-1.png)<!-- -->

    ## integer(0)

    ## [1] "AUC Score:  0.8696"

    ##                  type       acc     lowci      upci      sens      spec
    ## 1 Logistic Regression 0.7790698 0.6766886 0.8614339 0.7659574 0.7948718
    ## 2  9-Nearest Neighbor 0.8255814 0.7286908 0.8989624 0.8510638 0.7948718
    ## 3       Decision Tree 0.7906977 0.6895340 0.8709805 0.8085106 0.7692308
    ##          f1       auc
    ## 1 0.7912088 0.8696127
    ## 2 0.8421053 0.9026187
    ## 3 0.8085106 0.7888707

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.0 --

    ## v tibble  3.0.4     v dplyr   1.0.2
    ## v tidyr   1.1.2     v stringr 1.4.0
    ## v readr   1.4.0     v forcats 0.5.0
    ## v purrr   0.3.4

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()
    ## x purrr::lift()   masks caret::lift()
    ## x dplyr::select() masks MASS::select()

![](hd_knn_tree_files/figure-gfm/Accuracy%20Comparison-1.png)<!-- -->

![](hd_knn_tree_files/figure-gfm/F1-1.png)<!-- -->

![](hd_knn_tree_files/figure-gfm/Sensitivity-1.png)<!-- -->

![](hd_knn_tree_files/figure-gfm/Specificity-1.png)<!-- -->

    ## integer(0)

![](hd_knn_tree_files/figure-gfm/ROC%20Comparison-1.png)<!-- -->

![](hd_knn_tree_files/figure-gfm/AUC%20Score-1.png)<!-- -->
