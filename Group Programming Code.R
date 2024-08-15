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
View(df3) # df3 is training data

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
df1$V14 = replace(df1$V14, df1$V14 == " ?", NA) # https://www.digitalocean.com/community/tutorials/replace-in-r

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
df3$V14 = replace(df3$V14, df3$V14 == " ?", NA) # https://www.digitalocean.com/community/tutorials/replace-in-r

df3_clean = na.omit(df3)
rownames(df3_clean) = seq_len(nrow(df3_clean))

ls(df3_clean) # Same as colnames(df3_clean)

6. Preprocessing Data with Numerical Features
# https://towardsdatascience.com/normalization-vs-standardization-explained-209e84d0f81e#:~:text=Well%2C%20that%20depends%20on%20the,nearest%20neighbor%20and%20neural%20networks.
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

df1_clean$V2 = as.numeric(factor(df1_clean$V2))
df1_clean$V4 = as.numeric(factor(df1_clean$V4))
df1_clean$V6 = as.numeric(factor(df1_clean$V6))
df1_clean$V7 = as.numeric(factor(df1_clean$V7))
df1_clean$V8 = as.numeric(factor(df1_clean$V8))
df1_clean$V9 = as.numeric(factor(df1_clean$V9))
df1_clean$V10 = as.numeric(factor(df1_clean$V10))
df1_clean$V14 = as.numeric(factor(df1_clean$V14))
df1_clean$V15 = as.numeric(factor(df1_clean$V15))

colMeans(df1_clean)
apply(df1_clean, 2, var)
apply(df1_clean, 2, sd)

par(mfrow=c(3,5))
cn = names(df1_clean)
for (i in 1:15) { hist(df1_clean[,i],col="purple",xlab="",main=cn[i]) }

9. EDA: Numeric Bivariate Analysis
par(mfrow=c(3,4))
plot(df1_clean) # scatter plot

10.kNN
# Set train data and test data
census.train.knn=final_df1_clean
census.test.knn=final_df3_clean

# Remove unused columns and rename columns
census.train.knn$V14.Holand.Netherlands=NULL
census.train.knn$V14..=NULL
names(census.test.knn)[names(census.test.knn) == "V15..50K."]="V15..50K"
names(census.test.knn)[names(census.test.knn) == "V15...50K."]="V15...50K"

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
confusion_matrix=confusionMatrix(yhat.kknn, census.test.knn$V15...50K)
print(confusion_matrix)






