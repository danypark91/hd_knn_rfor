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

``` r
#Import Dataset from the local device
df <- read.csv("Heart.csv", header = TRUE)

#change erronous attribute name: ï..age
colnames(df)[colnames(df)=='ï..age'] <- 'age'

#Check the type and convert the dependent variable into factors
df$target <- as.factor(df$target)
```

Also, prior to the analysis, normalization of dependent variable was
conducted to equalize the weight and range. The `normalize` function
helped to acquire the condition. The normalized dataset was divided into
two sets: `train_df` was used to apply and train the `knn` alogorithm
and the measure of predictability utlized the `test_df`.

``` r
#Normalization function
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x)))
}

norm_df <- as.data.frame(lapply(df[,1:13], normalize))
head(norm_df,5)
```

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

``` r
#Combine the normalized dataframe with the target variable
norm_df <- cbind(norm_df, df$target)
colnames(norm_df)[colnames(norm_df)=="df$target"] <- "target"
head(norm_df,5)
```

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

``` r
#Split into Train and Test Datasets
library(caTools)
set.seed(1234)

sample = sample.split(norm_df, SplitRatio = 0.75)
train_df = subset(norm_df, sample==TRUE)
test_df = subset(norm_df,sample==FALSE)
```

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

``` r
#K-Nearest Neighbor sample run, k=15
library(class)
knn_15 <- knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=15)

#Predictability of the above model
table(knn_15, test_df$target)
```

    ##       
    ## knn_15  0  1
    ##      0 28  8
    ##      1 11 39

``` r
print(paste("Error: ", round(mean(knn_15 != test_df$target),4))) #knn error rate
```

    ## [1] "Error:  0.2209"

To minimize the effort, the list of error based on the value of k was
created. For-loop command helped to populate the sequential list of
error for K between 1 to 15. The graph represents the list, `knn-err`,
and the value of K. As the trend suggests, error rate decreases
significantly after k=4 and bounces back at 10. 8, 9 and 10 are the most
accurate models for the dataframe. However, as the tiebreaking rule
suggests, 9 is chosen to proceed further steps for the analysis.

``` r
#Error vs number of neighbors
knn_err <- list() #empty list

for (i in 1:15){
  #KNN
  temp <- mean((knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=i)) != test_df$target)
  knn_err[[i]] <- temp
}

#Plot of K vs Error list
x <- seq(1, 15, by=1)
knn_errplot <- plot(x, knn_err, type="b", axes=TRUE,
                    xlab="K", ylab="Error Rate", main="K vs Error of KNN", col="Red")
```

![](hd_knn_tree_files/figure-gfm/For%20Loop%20KNN%20from%201%20to%2015-1.png)<!-- -->
The exact rate of error of the K=9 model is 0.1744 which lower than the
k=15 model. Of course the accuracy of the model compare to the `test_df`
reponse variable is 0.8256.

``` r
#K=9, Fit KNN
df_knn_model <- knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=9)
df_knn_model_acc <- mean(df_knn_model == test_df$target)
df_knn_model_err <- mean(df_knn_model != test_df$target)

print(paste("Accuracy of the Model : ", round(df_knn_model_acc,4)))
```

    ## [1] "Accuracy of the Model :  0.8256"

``` r
print(paste("Error of the Model : ", round(df_knn_model_err,4)))
```

    ## [1] "Error of the Model :  0.1744"

### 1-4. Prediction and Performance Measure

As we dicovered the best-fit model of `KNN`, we should examine the
predictability of the model.

``` r
#Confusion Matrix of the KNN
library(caret)
```

    ## Loading required package: lattice

    ## Loading required package: ggplot2

``` r
df_knn_conf <- confusionMatrix(factor(df_knn_model), factor(test_df$target), positive=as.character(1))
df_knn_conf
```

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

``` r
library(kknn)
```

    ## 
    ## Attaching package: 'kknn'

    ## The following object is masked from 'package:caret':
    ## 
    ##     contr.dummy

``` r
df_knn_model.alt <- train.kknn(as.factor(target)~., train_df, ks=9,  method="knn", scale=TRUE)
df_knn_model_fit <- predict(df_knn_model.alt, test_df, type="prob")[,2]

#ROC and AUC of the plot
library(ROCR)
df_knn_prediction <- prediction(df_knn_model_fit, test_df$target)
df_knn_performance <- performance(df_knn_prediction, measure = "tpr", x.measure = "fpr")
df_knn_roc <- plot(df_knn_performance, col="Red",
                   main="ROC Curve - 9NN",
                   xlab="False Positive Rate",
                   ylab="True Positive Rate")+
  abline(a=0, b=1, col="Grey", lty=2)+
  abline(v=0, h=1, col="Blue", lty=3)+
  plot(df_knn_performance, col="Red",add=TRUE)
```

![](hd_knn_tree_files/figure-gfm/ROC%20and%20AUC-1.png)<!-- -->

``` r
df_knn_auc <- performance(df_knn_prediction, measure = "auc")
df_knn_auc <- df_knn_auc@y.values[[1]]
df_knn_auc
```

    ## [1] 0.9026187
