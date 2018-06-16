#R code for chapter 5 of DSILT: Statistics

setwd("/home/dsilt/Desktop/dsilt-stats-code/01-05 and 09 Titanic Survival Prediction")

train <- read.csv("train_clean.csv", header=T)
test <- read.csv("test_clean.csv", header=T)
train$Set <- 'train'
test$Set <- 'test'
alldata <- rbind(train[,-which(colnames(train) %in% 'Survived')], test)

#Covariance and correlation between 2 variables
cov(alldata$Fare_Per_Person, alldata$Age, use='complete.obs')
cor(alldata$Fare_Per_Person, alldata$Age, use='complete.obs')

plot(alldata$Fare_Per_Person, alldata$Age)

library(ltm)
biserial.cor(train$Age, train$DeckA, use='complete.obs')
