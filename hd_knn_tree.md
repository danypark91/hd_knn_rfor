hd\_knn\_tree
================
Dany Park
01/03/2021

# K-Nearest Neighbor and Decision Tree

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

## 1. K-Nearest Neighbor

### Overview of KNN

``` r
#Import Dataset from the local device
df <- read.csv("Heart.csv", header = TRUE)

#change erronous attribute name: ï..age
colnames(df)[colnames(df)=='ï..age'] <- 'age'

#Check the type and convert the dependent variable into factors
df$target <- as.factor(df$target)
```

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
mean(knn_15 != test_df$target) #knn error rate
```

    ## [1] 0.2209302

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
                   main="ROC Curve - KNN",
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
