#R code for chapter 4 of DSILT: Statistics

setwd("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction")

train <- read.csv("train.csv", header=T)
test <- read.csv("test.csv", header=T)

str(train)

#-------------------------------------------------------------------------------------------------#
#-------------------------------Dealing with Missing Data-----------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Define a function to count the nulls in every field
naCol <- function(x){
  y <- sapply(x, function(y) sum(length(which(is.na(y) | is.nan(y) | is.infinite(y) | y=='NA' | y=='NaN' | y=='Inf' | y==''))))
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

#To see the differences between the various null values, run the code below
x <- data.frame(a=c(1, NA, 3, NaN, 5, Inf, 7, ''), b=c(1,1,1,1,1,1,1,1))
x
naCol(x)

naCol(train)
naCol(test)

#Extract the deck level
train$Deck <- substr(train$Cabin, 1, 1)
test$Deck <- substr(test$Cabin, 1, 1)

#Cross tabulate class and deck - only doing this for training set, but if class and deck are related, they will be for both sets
library(gmodels)
CrossTable(train$Deck, train$Pclass, prop.chisq=F)

#Drop cabin
train$Cabin <- NULL
test$Cabin <- NULL

#Encode the missing deck values as something else
unique(train$Deck)
unique(test$Deck)  #Make sure the encoded value is not an existing category
train[train$Deck=='', 'Deck'] <- 'Z'
test[test$Deck=='', 'Deck'] <- 'Z'

#Fill in the 1 missing fare from the test set with the mean fare price for that passenger class
missing_fare_pclass <- test[is.na(test$Fare), 'Pclass']
test[is.na(test$Fare), 'Fare'] <- round(mean(c(train[train$Pclass==missing_fare_pclass, 'Fare'], test[test$Pclass==missing_fare_pclass, 'Fare']), na.rm=T), 4)

#Function for calculating the mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
#Fill in the 2 missing embarkation ports from the training set with the mode
train[is.na(train$Embarked) | train$Embarked=='', 'Embarked'] <- Mode(c(as.character(train$Embarked), as.character(test$Embarked)))
train$Embarked <- factor(train$Embarked)  #This relevels the factor to get rid of the empty string level

#Check if age is missing at random
table(train[is.na(train$Age), 'Survived'], train[is.na(train$Age), 'Pclass'])

#-------------------------------------------------------------------------------------------------#
#---------------------------------Dealing with Outliers-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Boxplots for numeric variables
for (col in 1:ncol(train[,which(sapply(train, class) == 'numeric')])) {
  boxplot(train[,which(sapply(train, class) == 'numeric')][col], 
          main=paste0("Box Plot for ", colnames(train[,which(sapply(train, class) == 'numeric')])[col]))
}

library(outliers)
grubbs.test(train$Fare)

#Check how often the outlier occurs
sum(train$Fare==max(train$Fare))
head(train[train$Fare==max(train$Fare),])

#Compute fare per person, since some passengers bought group tickets producing fares that are sums of the individual ticket prices
train$Set <- 'train'
test$Set <- 'test'
alldata <- rbind(train[,-2], test)
library(magrittr)
library(dplyr)
alldata <- alldata  %>% group_by(Fare, Ticket) %>% mutate(Group_Size = n())
alldata <- transform(alldata, Fare_Per_Person=Fare/Group_Size)

#Plot fare by passenger class
boxplot(alldata[alldata$Pclass==1,'Fare_Per_Person'], main="Box Plot for Fare Per Person - First Class")
boxplot(alldata[alldata$Pclass==2,'Fare_Per_Person'], main="Box Plot for Fare Per Person - Second Class")
boxplot(alldata[alldata$Pclass==3,'Fare_Per_Person'], main="Box Plot for Fare Per Person - Third Class")

#Inspect outliers by class
head(alldata[alldata$Fare_Per_Person>100 & alldata$Pclass==1,])
head(alldata[alldata$Fare_Per_Person>15 & alldata$Pclass==3,])

#Change fares that equal 0 to the mean fare for the class
firstclassmean <- mean(alldata[alldata$Pclass==1, 'Fare'])
firstclassmeanpp <- mean(alldata[alldata$Pclass==1, 'Fare_Per_Person'])
secclassmean <- mean(alldata[alldata$Pclass==2, 'Fare'])
secclassmeanpp <- mean(alldata[alldata$Pclass==2, 'Fare_Per_Person'])
thirdclassmean <- mean(alldata[alldata$Pclass==3, 'Fare'])
thirdclassmeanpp <- mean(alldata[alldata$Pclass==3, 'Fare_Per_Person'])
alldata[alldata$Fare==0 & alldata$Pclass==1, 'Fare'] <- firstclassmean
alldata[alldata$Fare_Per_Person==0 & alldata$Pclass==1, 'Fare_Per_Person'] <- firstclassmeanpp
alldata[alldata$Fare==0 & alldata$Pclass==2, 'Fare'] <- secclassmean
alldata[alldata$Fare_Per_Person==0 & alldata$Pclass==2, 'Fare_Per_Person'] <- secclassmeanpp
alldata[alldata$Fare==0 & alldata$Pclass==3, 'Fare'] <- thirdclassmean
alldata[alldata$Fare_Per_Person==0 & alldata$Pclass==3, 'Fare_Per_Person'] <- thirdclassmeanpp
rm('firstclassmean' ,'firstclassmeanpp', 'secclassmean', 'secclassmeanpp', 'thirdclassmean', 'thirdclassmeanpp')

#Bar plots for categorical variables
for (col in 1:ncol(alldata[,which(sapply(alldata, class) %in% c('factor', 'integer'))])) {
  barplot(table(alldata[,which(sapply(alldata, class) %in% c('factor', 'integer'))][col]), 
          main=paste0("Bar Plot for ", colnames(alldata[,which(sapply(alldata, class) %in% c('factor', 'integer'))])[col]))
}

#Inspect the 11 passengers with the same ticket
alldata[alldata$Group_Size==11,]

#Inspect passengers with same name
alldata[alldata$Name %in% names(which(table(alldata$Name)>1)),]

#-------------------------------------------------------------------------------------------------#
#---------------------------------Transforming the Data-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Check to see how many levels the character and factor variables have
for (col in 1:ncol(alldata[,which(sapply(alldata, class) %in% c('factor', 'character'))])) {
  print(paste0("Unique categories of ", colnames(alldata[,which(sapply(alldata, class) %in% c('factor', 'character'))])[col], ": ", length(unique(alldata[,which(sapply(alldata, class) %in% c('factor', 'character'))][,col]))))
}

#One-hot encode sex, embarked, and deck
library(caret)
dummies <- dummyVars( " ~ .", data=alldata[,which(colnames(alldata) %in% c('Sex', 'Embarked', 'Deck'))])
dummyenc <- data.frame(predict(dummies, newdata=alldata[,which(colnames(alldata) %in% c('Sex', 'Embarked', 'Deck'))]))
alldata <- cbind(alldata, dummyenc)
colnames(alldata) <- gsub("[.]", "_", colnames(alldata))
rm('dummies', 'dummyenc')

#Label encode ticket
alldata$Ticket_Enc <- as.integer(alldata$Ticket)

#Remove unneeded columns
alldata <- alldata[,-which(colnames(alldata) %in% c('PassengerId', 'Sex', 'Embarked', 'Deck', 'Ticket'))]

#Split back to training and test sets and save as cleaned data
train_clean <- alldata[alldata$Set=='train',]
train_clean$Set <- NULL
train_clean$Survived <- train$Survived
test_clean <- alldata[alldata$Set=='test',]
test_clean$Set <- NULL
write.csv(train_clean, file="train_clean.csv", row.names=F)
write.csv(test_clean, file="test_clean.csv", row.names=F)
