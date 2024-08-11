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

ls(df3) # Same as colnames(df3)

6. Preprocessing Data with Categorical Features
library(caret)
oneh_df1 = dummyVars( ~ ., data=df1_clean)
final_df1_clean = data.frame(predict(oneh_df1, newdata=df1_clean))

oneh_df3 = dummyVars( ~ ., data=df3_clean)
final_df3_clean = data.frame(predict(oneh_df3, newdata=df3_clean))

6. Preprocessing Data with Numerical Features
# https://towardsdatascience.com/normalization-vs-standardization-explained-209e84d0f81e#:~:text=Well%2C%20that%20depends%20on%20the,nearest%20neighbor%20and%20neural%20networks.
