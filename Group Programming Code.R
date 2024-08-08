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

View(df1)
View(df2)
View(df3)
View(df4)
View(df5)

dim(df1)
dim(df2)
dim(df3)
dim(df4)
dim(df5)

4. Data Preprocessing
sapply(df1,class)

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

df1$V14 = replace(df1$V14, df1$V14 == " ?", NA)
# df1 = na.omit(df1)

ls(df1) # Same as colnames(df1)

6. Preprocessing Data with Categorical Features
library(caret)
oneh = dummyVars( ~ ., data=df1)
final_df1 = data.frame(predict(oneh, newdata=df1))
