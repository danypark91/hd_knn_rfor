##------------------------------------------------------------------------------------##
##                                     FULL CODE                                      ##
##------------------------------------------------------------------------------------##

#Import Dataset from the local device
df <- read.csv("Heart.csv", header = TRUE)

#change erronous attribute name: ï..age
colnames(df)[colnames(df)=='ï..age'] <- 'age'
str(df)

#Check the type and convert the categorical variable into factors
head(df)
str(df)
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

sample = sample.split(df, SplitRatio = 0.75)
train_df = subset(df, sample==TRUE)
test_df = subset(df,sample==FALSE)

#Linear Dicriminant Analysis
#Fit the train_df with lda function
library(MASS)
df_lda_model <- lda(target~., data=train_df, prior=c(99/217,118/217)) #probability unknow, ratio of response variable
df_lda_model

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

