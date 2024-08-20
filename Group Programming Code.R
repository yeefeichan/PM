---------------------------------------------------------------------------------------------
Group Assignment
---------------------------------------------------------------------------------------------

1. Download the Census Income Dataset
# https://archive.ics.uci.edu/dataset/20/census+income

2. Set the working directory to the path that contains "census+income.zip"
# setwd("C:\\Users\\yeefe\\OneDrive\\Desktop\\Predictive Modelling\\Assignment")

3. Extract the data

library(ISLR2)

df1 = read.csv(unz("census+income.zip", "adult.data"), header=F)
df2 = read.csv(unz("census+income.zip", "adult.names"), header=F)
df3 = read.csv(unz("census+income.zip", "adult.test"), header=F)
df4 = read.csv(unz("census+income.zip", "Index"))
df5 = read.csv(unz("census+income.zip", "old.adult.names"), header=F)

View(df1) # df1 is training data
View(df3) # df3 is testing data

dim(df1)
dim(df3)

4. Data Preprocessing

sapply(df1,class)

df3 = df3[ -1,]
rownames(df3) = seq_len(nrow(df3))
df3$V1 = as.integer(df3$V1)
sapply(df3,class)

5. Data Cleaning

table(df1$V2)
table(df1$V4)
table(df1$V6)
table(df1$V7)
table(df1$V8)
table(df1$V9)
table(df1$V10)
table(df1$V14)
table(df1$V15)

# V2, V7, V14 has "?"

df1$V2 = replace(df1$V2, df1$V2 == " ?", NA)
df1$V7 = replace(df1$V7, df1$V7 == " ?", NA)
df1$V14 = replace(df1$V14, df1$V14 == " ?", NA)

df1_clean = na.omit(df1)
rownames(df1_clean) = seq_len(nrow(df1_clean))

ls(df1_clean) # Same as colnames(df1_clean)

table(df3$V2)
table(df3$V4)
table(df3$V6)
table(df3$V7)
table(df3$V8)
table(df3$V9)
table(df3$V10)
table(df3$V14)
table(df3$V15)

# V2, V7, V14 has "?"

df3$V2 = replace(df3$V2, df3$V2 == " ?", NA)
df3$V7 = replace(df3$V7, df3$V7 == " ?", NA)
df3$V14 = replace(df3$V14, df3$V14 == " ?", NA)

df3$V15 = gsub("\\.","",df3$V15) #gsub??? @eddieeeeee

df3_clean = na.omit(df3)
rownames(df3_clean) = seq_len(nrow(df3_clean))

ls(df3_clean) # Same as colnames(df3_clean)

6. Preprocessing Data with Numerical Features
library(caret)
process = preProcess(df1_clean, method=c("range"))
df1_clean_normalisation = data.frame(predict(process, newdata=df1_clean))

df3_clean_normalisation = data.frame(predict(process, newdata=df3_clean))

7. Preprocessing Data with Categorical Features

oneh_df1 = dummyVars( ~ ., data=df1_clean_normalisation)
final_df1_clean = data.frame(predict(oneh_df1, newdata=df1_clean_normalisation))

oneh_df3 = dummyVars( ~ ., data=df3_clean_normalisation)
final_df3_clean = data.frame(predict(oneh_df3, newdata=df3_clean_normalisation))

8. EDA: Numeric Univariate Analysis

df_clean = rbind(df1_clean,df3_clean)
df_clean$V2 = as.numeric(factor(df_clean$V2))
df_clean$V4 = as.numeric(factor(df_clean$V4))
df_clean$V6 = as.numeric(factor(df_clean$V6))
df_clean$V7 = as.numeric(factor(df_clean$V7))
df_clean$V8 = as.numeric(factor(df_clean$V8))
df_clean$V9 = as.numeric(factor(df_clean$V9))
df_clean$V10 = as.numeric(factor(df_clean$V10))
df_clean$V14 = as.numeric(factor(df_clean$V14))
df_clean$V15 = as.numeric(factor(df_clean$V15))
df_clean$V15 = df_clean$V15 - 1

colMeans(df_clean)
apply(df_clean, 2, var)
apply(df_clean, 2, sd)

# Data Visualisation => Histograms
par(mfrow=c(3,5))
cn = names(df_clean)
for (i in 1:15) { hist(df_clean[,i],col=rainbow(length(df_clean)),xlab="",main=cn[i]) }

9. EDA: Numeric Bivariate Analysis

library(corrplot)
cor_matrix=cor(df_clean[, sapply(df_clean, is.numeric)])
corrplot(cor_matrix, method="circle")

# Data Visualisation => Scatter Plot
par(mfrow=c(3,4))
plot(df_clean)

10. PCA
df_clean_scale = scale(df_clean, scale=TRUE)
PCA = prcomp(df_clean_scale)
print(PCA)
print(summary(PCA))
biplot(PCA)

11. k-Means Clustering
par(mfrow=c(1,2))
set.seed(123)
kmc=kmeans(df_clean_scale,2,nstart=500)
kmc$cluster

library(cluster)
clusplot(df_clean_scale, kmc$cluster, main=paste0("k-Means Clustering (K=2, seed=",123,")"), 
  xlab="x", ylab="y", cex=2)

12. kNN
# Set train data and test data
census.train.knn=final_df1_clean
census.test.knn=final_df3_clean

# Remove unused columns
census.train.knn$V14.Holand.Netherlands=NULL
census.train.knn$V14..=NULL

# Convert into categorical data
census.train.knn$V15...50K=as.factor(census.train.knn$V15...50K)
census.test.knn$V15...50K=as.factor(census.test.knn$V15...50K)

# Perform kNN
library(kknn)
cat("\nTraining and validation with wkNN ...\n\n")
census.kknn=kknn(V15...50K ~ ., census.train.knn, census.test.knn, k = 1)

# Evaluating k-NN Model performance
yhat.kknn=fitted(census.kknn)
yhat.kknn=factor(yhat.kknn, levels = levels(census.test.knn$V15...50K))
knn_confusion_matrix=confusionMatrix(yhat.kknn, census.test.knn$V15...50K)
print(knn_confusion_matrix)

# AUROC for kNN
library(pROC)
yhat.prob=census.kknn$prob[,2]
roc_curve_knn=roc(census.test.knn$V15...50K,yhat.prob)
plot(roc_curve_knn,col="blue",main="ROC Curve for kNN Model")
auc_value=auc(roc_curve_knn)
print(paste("AUC:",auc_value))

# Evaluation for Binary Classification using formulas
cftable.std = table(yhat.kknn, census.test.knn$V15...50K)

ACR = sum(diag(cftable.std)) / sum(cftable.std)
TPR = cftable.std[1,1] / sum(cftable.std[,1])
TNR = cftable.std[2,2] / sum(cftable.std[,2])
PPV = cftable.std[1,1] / sum(cftable.std[1,])
NPV = cftable.std[2,2] / sum(cftable.std[2,])
FPR = 1 - TNR
FNR = 1 - TPR

RandomAccuracy = (sum(cftable.std[,2]) * sum(cftable.std[2,]) + 
	sum(cftable.std[,1]) * sum(cftable.std[1,])) / (sum(cftable.std)^2)

Kappa = (ACR - RandomAccuracy) / (1 - RandomAccuracy)

# Print the confusion matrix
print(cftable.std)

# Print the metrics
cat("\n      Accuracy :", ACR, "\n")
cat("\n         Kappa :", Kappa, "\n")
cat("\n   Sensitivity :", TPR, "\n")
cat("\n   Specificity :", TNR, "\n")
cat("\nPos Pred Value :", PPV, "\n")
cat("\nNeg Pred Value :", NPV, "\n")
cat("\n           FPR :", FPR, "\n")
cat("\n           FNR :", FNR, "\n")

13. Logistic Regression

14. Naive Bayes

library(naivebayes)

# Set train data and test data
census.train.nb = final_df1_clean
census.test.nb = final_df3_clean

# Remove unwanted columns
census.train.nb$V14.Holand.Netherlands = NULL
census.train.nb$V14.. = NULL

# Convert into categorical data
census.train.nb$V15...50K = as.factor(census.train.nb$V15...50K)
census.test.nb$V15...50K = as.factor(census.test.nb$V15...50K)

# Training and validation with Naive Bayes with Laplace smoothing
cat("\nTraining and validation with Naive Bayes ...\n\n")
model.nb = naive_bayes(V15...50K ~ ., data = census.train.nb, laplace = 1)

# Get predictions
pred.nb = predict(model.nb, census.test.nb)
nb_confusion_matrix=confusionMatrix(pred.nb, census.test.nb$V15...50K)
print(nb_confusion_matrix)

# Evaluation for Binary Classification using formulas (performance?)
cfmat = table(pred.nb, census.test.nb$V15...50K)
ACR = sum(diag(cfmat)) / sum(cfmat)
TPR = cfmat[1,1] / sum(cfmat[,1])
TNR = cfmat[2,2] / sum(cfmat[,2])
PPV = cfmat[1,1] / sum(cfmat[1,])
NPV = cfmat[2,2] / sum(cfmat[2,])
FPR = 1 - TNR
FNR = 1 - TPR
RandomAccuracy = (sum(cfmat[,2]) * sum(cfmat[2,]) + 
                  sum(cfmat[,1]) * sum(cfmat[1,])) / (sum(cfmat)^2)
Kappa = (ACR - RandomAccuracy) / (1 - RandomAccuracy)

print(cfmat)
cat("\n      Accuracy :", ACR, "\n")
cat("\n         Kappa :", Kappa, "\n")
cat("\n   Sensitivity :", TPR, "\n")
cat("\n   Specificity :", TNR, "\n")
cat("\nPos Pred Value :", PPV, "\n")
cat("\nNeg Pred Value :", NPV, "\n")
cat("\n           FPR :", FPR, "\n")
cat("\n           FNR :", FNR, "\n")

15. Decision Tree
# Set train data and test data
census.train=df1_clean
census.test=df3_clean

# Add new column "Income"("Income"="Yes" if "V15" =">50K" and "No" if "V15" not equal ">50K") for both census.train and census.test while remove column "V15" in both census.train and census.test
census.test$V15=trimws(as.character(census.test$V15))
census.test$Income=factor(ifelse(census.test$V15 == ">50K", "Yes", "No"))
census.test$V15=NULL
census.train$V15=trimws(as.character(census.train$V15))
census.train$Income=factor(ifelse(census.train$V15 == ">50K", "Yes", "No"))
census.train$V15=NULL

# Perform Decision Tree Model
library(rpart)
library(rpart.plot)
rpart.census=rpart(Income~.,census.train)
rpart.plot(rpart.census)

# Evaluate Decision Tree Model performance
census.pred=predict(rpart.census, census.test, type = "class")
confusionMatrix(census.pred, census.test$Income)

# AUROC for Decision Tree Model
library(pROC)
census.probs=predict(rpart.census, census.test, type = "prob")
roc_curve_dt=roc(census.test$Income, census.probs[, 2])
plot(roc_curve_dt)
auc(roc_curve_dt)
