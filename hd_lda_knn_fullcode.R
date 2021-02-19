df <- read.csv("Heart.csv", header = TRUE)
head(df)
str(df)
df$sex <- as.factor(df$sex)
df$cp <- as.factor(df$cp)
df$fbs <- as.factor(df$fbs)
df$restecg <- as.factor(df$restecg)
df$exang <- as.factor(df$exang)
df$slope <- as.factor(df$slope)
df$thal <- as.factor(df$thal)

colnames(df)[colnames(df)=='ï..age'] <- 'age'
str(df)

