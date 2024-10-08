---------------------------------------------------------------------------------------------
Group Assignment
---------------------------------------------------------------------------------------------

1. Download the Census Income Dataset
# https://archive.ics.uci.edu/dataset/20/census+income

2. Set the working directory to the path that contains "census+income.zip"
# setwd()

3. Extract the data

library(ISLR2)

df1 = read.csv(unz("census+income.zip", "adult.data"), header=F)
df2 = read.csv(unz("census+income.zip", "adult.names"), header=F)
df3 = read.csv(unz("census+income.zip", "adult.test"), header=F)
df4 = read.csv(unz("census+income.zip", "Index"))
df5 = read.csv(unz("census+income.zip", "old.adult.names"), header=F)

# df1 is training data
# df3 is training data
# View(df1) 
# View(df3)
# dim(df1)
# dim(df3)

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

ls(df1_clean)

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

df3$V15 = gsub("\\.","",df3$V15)

df3_clean = na.omit(df3)
rownames(df3_clean) = seq_len(nrow(df3_clean))

ls(df3_clean)

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

# Check the mean, variance, and standard deviation for each column
colMeans(df_clean)
apply(df_clean, 2, var)
apply(df_clean, 2, sd)

# Data Visualisation => Histograms
par(mfrow=c(3,5))
cn = names(df_clean)
for (i in 1:15) { hist(df_clean[,i],col=rainbow(length(df_clean)),xlab="",main=cn[i]) }

9. EDA: Numeric Bivariate Analysis

# Data Visualisation => Correlation Plot
library(corrplot)
cor_matrix = cor(df_clean[, sapply(df_clean, is.numeric)])
corrplot(cor_matrix, method="circle")

# Data Visualisation => Scatter Plot
par(mfrow=c(3,4))
plot(df_clean)

10. PCA

df_clean_scale = scale(df_clean, scale=TRUE)
PCA = prcomp(df_clean_scale, scale=TRUE)
print(PCA)
screeplot(PCA)

pr.var = PCA$sdev^2
pve = pr.var/sum(pr.var) # proportions of variance

par(mfrow=c(1,2))
plot(pve, type="o", ylab="PVE", xlab="Principal Component", col="blue")
plot(cumsum(pve), type="o", ylab="Cumulative PVE", 
     xlab="Principal Component", col="purple")

11. k-Means Clustering

set.seed(123)  # Reduce randomness by allowing repetition

kmc = kmeans(df_clean_scale, 2, nstart=500)
print(kmc)

library(cluster) 
clusplot(df_clean_scale, kmc$cluster, main=paste0("k-Means Clustering (K=2, seed=",123,")"), 
	 xlab="x", ylab="y", cex=2)

12. k-NN

# Set the training data and testing data
census.train.knn = final_df1_clean
census.test.knn = final_df3_clean

# Remove unused columns
census.train.knn$V14.Holand.Netherlands = NULL
census.train.knn$V14.. = NULL

# Convert into categorical data
census.train.knn$V15..50K = as.factor(census.train.knn$V15..50K)
census.test.knn$V15..50K = as.factor(census.test.knn$V15..50K)

# Perform k-NN 
library(kknn)
cat("\nTraining and validation with wkNN ...\n\n")
census.kknn = kknn(V15..50K ~ ., census.train.knn, census.test.knn, k = 1)

# Evaluating k-NN model performance
yhat.kknn = fitted(census.kknn)
yhat.kknn = factor(yhat.kknn, levels=levels(census.test.knn$V15..50K))
knn_confusion_matrix = confusionMatrix(yhat.kknn, census.test.knn$V15..50K)
print(knn_confusion_matrix)

# AUROC for k-NN
library(pROC)
yhat.prob = census.kknn$prob[,2]
roc_curve_knn = roc(census.test.knn$V15..50K, yhat.prob)
plot(roc_curve_knn, col="blue", main="ROC Curve for kNN Model")
auc(roc_curve_knn)

13. Logistic Regression

#performance()
performance = function(xtab, desc=""){
    cat("\n", desc,"\n", sep="")
    print(xtab)

    ACR = sum(diag(xtab))/sum(xtab)
    CI  = binom.test(sum(diag(xtab)), sum(xtab))$conf.int
    cat("\n        Accuracy :", ACR)
    cat("\n          95% CI : (", CI[1], ",", CI[2], ")\n")

    if(nrow(xtab)>2){
        # e1071's classAgreement() in matchClasses.R
        # Ref: https://stats.stackexchange.com/questions/586342/measures-to-compare-classification-partitions
        n  = sum(xtab)
        ni = apply(xtab, 1, sum)
        nj = apply(xtab, 2, sum)
        p0 = sum(diag(xtab))/n
        pc = sum(ni * nj)/n^2
        Kappa = (p0 - pc)/(1 - pc)
        cat("\n           Kappa :", Kappa, "\n")
        cat("\nStatistics by Class:\n")
        # Levels of the actual data
        lvls = dimnames(xtab)[[2]]
        sensitivity = c()
        specificity = c()
        ppv         = c()
        npv         = c()
        for(i in 1:length(lvls)) {
            sensitivity[i] = xtab[i,i]/sum(xtab[,i])
            specificity[i] = sum(xtab[-i,-i])/sum(xtab[,-i])
            ppv[i]         = xtab[i,i]/sum(xtab[i,])
            npv[i]         = sum(xtab[-i,-i])/sum(xtab[-i,])
        }
        b = data.frame(rbind(sensitivity,specificity,ppv,npv))
        names(b) = lvls
        print(b)
    } else {
         #names(dimnames(xtab)) = c("Prediction", "Actual")
         TPR = xtab[1,1]/sum(xtab[,1]); TNR = xtab[2,2]/sum(xtab[,2])
         PPV = xtab[1,1]/sum(xtab[1,]); NPV = xtab[2,2]/sum(xtab[2,])
         FPR = 1 - TNR                ; FNR = 1 - TPR
         # https://standardwisdom.com/softwarejournal/2011/12/confusion-matrix-another-single-value-metric-kappa-statistic/
         RandomAccuracy = (sum(xtab[,2])*sum(xtab[2,]) + 
           sum(xtab[,1])*sum(xtab[1,]))/(sum(xtab)^2)
         Kappa = (ACR - RandomAccuracy)/(1 - RandomAccuracy)
         cat("\n           Kappa :", Kappa, "\n")
         cat("\n     Sensitivity :", TPR)
         cat("\n     Specificity :", TNR)
         cat("\n  Pos Pred Value :", PPV)
         cat("\n  Neg Pred Value :", NPV)
         cat("\n             FPR :", FPR)
         cat("\n             FNR :", FNR, "\n")
         cat("\n'Positive' Class :", dimnames(xtab)[[1]][1], "\n")
    }
}

# Set the training data and testing data
census.train.logreg = final_df1_clean
census.test.logreg = final_df3_clean
V15.test = census.test.logreg$V15..50K

# Preprocess training data
census.train.logreg$V15..50K = trimws(as.character(census.train.logreg$V15..50K))
census.train.logreg$Income = factor(ifelse(census.train.logreg$V15..50K == ">50K", "Yes", "No"))
census.train.logreg$V15..50K = NULL 

# Preprocess testing data
census.test.logreg$V15..50K = trimws(as.character(census.test.logreg$V15..50K))
census.test.logreg$Income = factor(ifelse(census.test.logreg$V15..50K == ">50K", "Yes", "No"))
census.test.logreg$V15..50K = NULL

# Remove unused columns
census.train.logreg$V14.Holand.Netherlands = NULL
census.train.logreg$V14.. = NULL

# Fit logistic regression model on training data
logreg.fits = glm(Income ~ ., data = census.train.logreg, family = binomial)

# Predict probabilities on test data
logreg.probs = predict(logreg.fits, newdata = census.test.logreg, type = "response")

# Convert probabilities to class predictions
logreg.predictions = ifelse(logreg.probs > 0.5, "Yes", "No")
logreg.predictions = factor(logreg.predictions, levels = c("No", "Yes"))

# Evaluating the Logistic Regression model performance
cfmat.logreg = table(logreg.predictions, V15.test)
performance(cfmat.logreg, "Performance of Logistic Regression Model on Census Income")

# AUROC for Logistic Regression
library(pROC)
roc_curve = roc(V15.test, logreg.probs, levels = c("0", "1"))
plot(roc_curve, main = "AUROC Curve for Logistic Regression Model")
auc(roc_curve)

14. Naïve Bayes

library(naivebayes)

# Set the training data and testing data
census.train.nb = final_df1_clean
census.test.nb = final_df3_clean

# Remove unused columns
census.train.nb$V14.Holand.Netherlands = NULL
census.train.nb$V14.. = NULL

# Convert into categorical data
census.train.nb$V15..50K = as.factor(census.train.nb$V15..50K)
census.test.nb$V15..50K = as.factor(census.test.nb$V15..50K)

# Training and validation with Naïve Bayes with Laplace smoothing
cat("\nTraining and validation with Naive Bayes ...\n\n")
model.nb = naive_bayes(V15..50K ~ ., data = census.train.nb, laplace = 1)

# Get predictions
pred.nb = predict(model.nb, census.test.nb)
nb_confusion_matrix = confusionMatrix(pred.nb, census.test.nb$V15..50K)
print(nb_confusion_matrix)
census.probs = predict(model.nb, census.test.nb, type = "prob")

# Inspect column names of census.probs
print(colnames(census.probs))

# Correct column name for the positive class
positive_class_col = "1"  # Use the correct column name based on the inspection

# AUROC for Naïve Bayes
library(pROC)
roc_curve_nb = roc(census.test.nb$V15...50K, census.probs[, positive_class_col])
plot(roc_curve_nb, main="ROC Curve for Naïve Bayes Model", col="blue", lwd=2)
auc(roc_curve_nb)

15. Decision Tree

# Set the training data and testing data
census.train.dt = df1_clean
census.test.dt = df3_clean

# Preprocess training data
census.train.dt$V15 = trimws(as.character(census.train.dt$V15))
census.train.dt$Income = factor(ifelse(census.train.dt$V15 == ">50K", "Yes", "No"))
census.train.dt$V15 = NULL

# Preprocess testing data
census.test.dt$V15 = trimws(as.character(census.test.dt$V15))
census.test.dt$Income = factor(ifelse(census.test.dt$V15 == ">50K", "Yes", "No"))
census.test.dt$V15 = NULL

# Perform Decision Tree
library(rpart)
library(rpart.plot)
rpart.census = rpart(Income~.,census.train.dt)
rpart.plot(rpart.census)

# Evaluate Decision Tree Model performance
census.pred=predict(rpart.census, census.test.dt, type = "class")
confusionMatrix(census.pred, census.test.dt$Income)

# AUROC for Decision Tree
library(pROC)
census.probs = predict(rpart.census, census.test.dt, type = "prob")
roc_curve_dt = roc(census.test.dt$Income, census.probs[, 2])
plot(roc_curve_dt)
auc(roc_curve_dt)
