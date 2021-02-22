##------------------------------------------------------------------------------------##
##                                     FULL CODE                                      ##
##------------------------------------------------------------------------------------##

#Import Dataset from the local device
df <- read.csv("Heart.csv", header = TRUE)

#change erronous attribute name: ï..age
colnames(df)[colnames(df)=='ï..age'] <- 'age'
str(df)

#Check the type and convert the dependent variable into factors
df$target <- as.factor(df$target)

#Normalize the dataframe
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x)))
}
norm_df <- as.data.frame(lapply(df[,1:13], normalize))
head(norm_df,5)

#Combine the normalized dataframe with the target variable
norm_df <- cbind(norm_df, df$target)
colnames(norm_df)[colnames(norm_df)=="df$target"] <- "target"
head(norm_df,5)

#Split into Train and Test Datasets
library(caTools)
set.seed(1234)

sample = sample.split(norm_df, SplitRatio = 0.75)
train_df = subset(norm_df, sample==TRUE)
test_df = subset(norm_df,sample==FALSE)

#K-Nearest Neighbor sample run, k=15
library(class)
knn_15 <- knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=15)

#Predictability of the above model
table(knn_15, test_df$target)
mean(knn_15 != test_df$target) #knn error rate


#Error vs number of neighbors
knn_err <- list() #empty list

for (i in 1:15){
  #KNN
  temp <- mean((knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=i)) != test_df$target)
  knn_err[[i]] <- temp
}

#Plot of K vs Error list
x <- seq(1, 15, by=1)
plot(x,knn_err, type="b", 
     xlab="K", ylab="Error Rate", main="K vs Error Rate", col="Red")

#K=9, Fit KNN
df_knn_model <- knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=9)
df_knn_model_acc <- mean(df_knn_model == test_df$target)
df_knn_model_err <- mean(df_knn_model != test_df$target)

print(paste("Accuracy of the Model : ", round(df_knn_model_acc,4)))
print(paste("Error of the Model : ", round(df_knn_model_err,4)))


#Confusion Matrix of the LDA
library(caret)
confusionMatrix(factor(df_knn_model), factor(test_df$target), positive=as.character(1))

library(kknn)
df_knn_model.alt <- train.kknn(as.factor(target)~., train_df, ks=9,  method="knn", scale=TRUE)
df_knn_model_fit <- predict(df_knn_model.alt, test_df, type="prob")[,2]

#ROC and AUC of the plot
library(ROCR)
df_knn_prediction <- prediction(df_knn_model_fit, test_df$target)
df_lda_performance <- performance(df_knn_prediction, measure = "tpr", x.measure = "fpr")
df_knn_roc <- plot(df_lda_performance, col="Red",
                   main="ROC Curve",
                   xlab="False Positive Rate",
                   ylab="True Positive Rate")+
  abline(a=0, b=1, col="Grey", lty=2)+
  abline(v=0, h=1, col="Blue", lty=3)+
  plot(df_lda_performance, col="Red",add=TRUE)

df_knn_auc <- performance(df_knn_prediction, measure = "auc")
df_knn_auc <- df_knn_auc@y.values[[1]]
df_knn_auc

#Random Forest

