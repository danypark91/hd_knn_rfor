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
print(paste("Error: ", round(mean(knn_15 != test_df$target),4))) #knn error rate

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

#K=9, Fit KNN
df_knn_model <- knn(train=train_df[1:13], test=test_df[1:13], cl=train_df$target, k=9)
df_knn_model_acc <- mean(df_knn_model == test_df$target)
df_knn_model_err <- mean(df_knn_model != test_df$target)

print(paste("Accuracy of the Model : ", round(df_knn_model_acc,4)))
print(paste("Error of the Model : ", round(df_knn_model_err,4)))

#Confusion Matrix of the KNN
library(caret)
df_knn_conf <- confusionMatrix(factor(df_knn_model), factor(test_df$target), positive=as.character(1))

library(kknn)
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

df_knn_auc <- performance(df_knn_prediction, measure = "auc")
df_knn_auc <- df_knn_auc@y.values[[1]]
df_knn_auc


#Decision Tree
#Convert categorical variable from int to factor
df <- read.csv("Heart.csv", header = TRUE)
colnames(df)[colnames(df)=='ï..age'] <- 'age'

df$sex <- as.factor(df$sex)
df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)

#Split into Train and Test Datasets
library(caTools)
set.seed(1234)

sample_dt = sample.split(df, SplitRatio = 0.75)
train_dt_df = subset(df, sample_dt==TRUE)
test_dt_df = subset(df,sample_dt==FALSE)

#Decision Tree for the train model
library(rpart)

df_dt_model <- rpart(as.factor(target)~., data=train_dt_df, method = "class", control=rpart.control(xval=10, minbucket=1, cp=0))
printcp(df_dt_model) # display the results
plotcp(df_dt_model) # visualize cross-validation results

#Prune the tree based on the result
df_dt_prune <- prune(df_dt_model, cp=df_dt_model$cptable[which.min(df_dt_model$cptable[,"xerror"]),"CP"])

#ConfusionMatrix
df_dt_model_fit <- predict(df_dt_prune, newdata=test_dt_df, type="prob")[,2]
df_dt_model_conf <- ifelse(df_dt_model_fit>0.5,1,0)

df_dt_conf <- confusionMatrix(as.factor(df_dt_model_conf), as.factor(test_dt_df$target), positive=as.character(1))

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

df_dt_auc <- performance(df_dt_prediction, measure="auc")
df_dt_auc <- df_dt_auc@y.values[[1]]
df_dt_auc


#Comparison with the other classification methods
#Logistic Regression Model
library(MASS)
df_model.part <- glm(target~sex+cp+trestbps+thalach+oldpeak+ca, data=train_dt_df, family=binomial(link="logit"))
df_model_fit <- predict(df_model.part, newdata=test_dt_df, type="response")
df_model_confmat <- ifelse(df_model_fit >0.5, 1, 0)

df_log_conf <- confusionMatrix(factor(df_model_confmat), factor(test_dt_df$target), positive=as.character(1))

df_prediction <- prediction(df_model_fit, test_dt_df$target)
df_performance <- performance(df_prediction, measure = "tpr", x.measure="fpr")

plot(df_performance, col = "Red", 
     main = "ROC Curve - Logistic Regression",
     xlab="False Postiive Rate", ylab="True Positive Rate")+
  abline(a=0, b=1, col= "Grey", lty=2)+
  abline(v=0, h=1, col= "Blue", lty=3)+
  plot(df_performance, col = "Red", 
       main = "ROC Curve - Logistic Regression",
       xlab="False Postiive Rate", ylab="True Positive Rate",add=TRUE)

df_auc <- performance(df_prediction, measure = "auc")
df_auc <- df_auc@y.values[[1]]
print(paste("AUC Score: ", lapply(df_auc,round,4)))

#Accuracy Comparison
#Model Fit summary dataframe
df_ci <- data.frame(type=c("Logistic Regression", "9-Nearest Neighbor", "Decision Tree"), 
                    acc=c(df_log_conf$overall[1], df_knn_conf$overall[1], df_dt_conf$overall[1]),
                    lowci=c(df_log_conf$overall[3], df_knn_conf$overall[3], df_dt_conf$overall[3]),
                    upci=c(df_log_conf$overall[4], df_knn_conf$overall[4], df_dt_conf$overall[4]),
                    sens=c(df_log_conf$byClass[1], df_knn_conf$byClass[1], df_dt_conf$byClass[1]),
                    spec=c(df_log_conf$byClass[2], df_knn_conf$byClass[2], df_dt_conf$byClass[2]),
                    f1=c(df_log_conf$byClass[7], df_knn_conf$byClass[7], df_dt_conf$byClass[7]),
                    auc=c(df_auc, df_knn_auc, df_dt_auc))

#Accuracy Comparison with 95% Confidence Interval
library(tidyverse)
library(ggsci)
ggplot(data=df_ci, aes(type,acc))+
  labs(title="Comparison of Classification", subtitle="Accuracy and Confidence Interval", x="Classification", y="Accuracy")+
  geom_point(size=5, aes(color=type))+
  geom_errorbar(aes(ymax=upci, ymin=lowci),width=0.2)+
  theme_bw()+
  scale_fill_npg()

#F1 Score
ggplot(data=df_ci, aes(type, f1))+
  geom_point(size=5, aes(color=type))+
  geom_text(aes(label=round(f1,4), hjust=-0.3, vjust=0))+
  labs(title="Comparison of Classification", subtitle="F1 Score", x="Classification Method", y="F1 Score")+
  theme_bw()+
  scale_fill_tron()

#Sensitivity
ggplot(data=df_ci, aes(type, sens))+
  geom_point(size=5, aes(color=type))+
  geom_text(aes(label=round(sens,4), hjust=-0.3, vjust=0))+
  labs(title="Comparison of Classification", subtitle="Sensitivity", x="Classification Method", y="Sensitivity")+
  theme_bw()+
  scale_fill_npg()

#Specificty
ggplot(data=df_ci, aes(type, spec))+
  geom_point(size=5, aes(color=type))+
  geom_text(aes(label=round(spec,4), hjust=-0.3, vjust=0))+
  labs(title="Comparison of Classification", subtitle="Specifity", x="Classification Method", y="Specificity")+
  theme_bw()+
  scale_fill_npg()

#ROC Comparison
plot(df_performance, main="ROC Curve: Comparison", col="Red")+
  abline(a=0, b=1, col= "Grey", lty=2)+
  abline(v=0, h=1, col= "Blue", lty=3)
par(new=TRUE)
plot(df_knn_performance, col="Orange")
par(new=TRUE)
plot(df_dt_performance, col="Dark Red")

#AUC
ggplot(data=df_ci, aes(type, auc))+
  geom_point(size=5, aes(color=type))+
  geom_text(aes(label=round(auc,4), hjust=-0.3, vjust=0))+
  labs(title="Comparison of Classification", subtitle="Area Under Curve", x="Classification Method", y="AUC")+
  theme_bw()+
  scale_fill_npg()