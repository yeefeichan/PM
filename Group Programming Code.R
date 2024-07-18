---------------------------------------------------------------------------------------------
Group Assignment
---------------------------------------------------------------------------------------------

1. Download the Census Income Dataset
# https://archive.ics.uci.edu/dataset/20/census+income

2. Set the working directory to the path that contains "census+income.zip"
# setwd("C:\\Users\\yeefe\\OneDrive\\Desktop\\Predictive Modelling\\Assignment")

3. Extract the data

library(ISLR2)

df1 = read.csv(unz("census+income.zip", "adult.data"))
df2 = read.csv(unz("census+income.zip", "adult.names"), header=F)
df3 = read.csv(unz("census+income.zip", "adult.test"), header=F)
df4 = read.csv(unz("census+income.zip", "Index"))
df5 = read.csv(unz("census+income.zip", "old.adult.names"))

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
sapply(ls(df1),class)


4. Data Cleaning
df1_clean = na.omit(df1)
df2_clean = na.omit(df2)
df3_clean = na.omit(df3) # Has 1 row with missing value(s)
df4_clean = na.omit(df4)
df5_clean = na.omit(df5)

dim(df1_clean)
dim(df2_clean)
dim(df3_clean)
dim(df4_clean)
dim(df5_clean)







ls(df1) # Same as colnames(df1)

4. Preprocessing Data with Categorical Features
library(caret)
oneh=dummyVars(~.,data=df1)
final_df=data.frame(predict(oneh,newdata=df1))
