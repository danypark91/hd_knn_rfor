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


#error vs number of neighbors
knn_err <- list() #empty list

for (i in 1:15){
  #KNN
  temp<-mean((knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=i)) != test_df$target)
  knn_err[[i]] <- temp
}

#Plot of K vs Error list
x <- seq(1, 15, by=1)
plot(x,knn_err, type="l", 
     xlab="K", ylab="Error Rate", main="K vs Error Rate", col="Red")



#plot the summary
plot(df_lda_model)

#Prediction of the model to the test dataset
df_lda_model_fit <- predict(df_lda_model, newdata=test_df)
names(df_lda_model_fit) #output categories

#Confusion Matrix of the LDA
library(caret)
confusionMatrix(factor(df_lda_model_fit$class), factor(test_df$target), positive=as.character(1))

#Prediction plot
ldahist(df_lda_model_fit$x[,1], df_lda_model_fit$class)
lda_data <- data.frame(class = df_lda_model_fit$class, LD1 = df_lda_model_fit$x[,1], target = test_df$target)
ggplot(lda_data)+
  geom_point(aes(LD1, class, color=as.factor(target)))+
  labs(title="Classification Distribution",
       x="LD1",
       y="Class")+
  theme_bw()

#ROC and AUC of the plot
library(ROCR)
df_lda_prediction <- prediction(df_lda_model_fit$x[,1], test_df$target)
df_lda_performance <- performance(df_lda_prediction, measure = "tpr", x.measure = "fpr")
df_lda_roc <- plot(df_lda_performance, col="Red",
                   main="ROC Curve",
                   xlab="False Positive Rate",
                   ylab="True Positive Rate")+
  abline(a=0, b=1, col="Grey", lty=2)+
  abline(v=0, h=1, col="Blue", lty=3)+
  plot(df_lda_performance, col="Red",add=TRUE)

df_lda_auc <- performance(df_lda_prediction, measure = "auc")
df_lda_auc <- df_lda_auc@y.values[[1]]
df_lda_auc

#K-Nearest Neighbor

