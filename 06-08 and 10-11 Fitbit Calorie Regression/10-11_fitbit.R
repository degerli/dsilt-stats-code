#R code for chapters 10 and 11 of DSILT: Statistics

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/06-08 and 10-11 Fitbit Calorie Regression")

train <- read.csv("train.csv", header=T)
test <- read.csv("test.csv", header=T)

#-------------------------------------------------------------------------------------------------#
#----------------------Chapter 10: Regression for Count or Integer Variables----------------------#
#-------------------------------------------------------------------------------------------------#

str(train)
head(train)

mean(train$Calories)
var(train$Calories)

#Negative binomial regression
library(MASS)
negb_reg <- glm.nb(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train, link=identity)
summary(negb_reg)
deviance(negb_reg)
plot(negb_reg)

#Get the Newey-West HAC estimates of the model
library(sandwich)
library(lmtest)
coeftest(negb_reg, vcov.=NeweyWest)

#-------------------------------------------------------------------------------------------------#
#----------------------Chapter 11: Regression for Count or Integer Variables----------------------#
#-------------------------------------------------------------------------------------------------#

#This gives us a baseline to compare against to see how censoring impacts regression analysis
hist(train$Calories, breaks=50, main="Freq Dist of Calories Before Censoring")
library(MASS)
negb_reg <- glm.nb(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train, link=identity)
summary(negb_reg)

#Apply censoring
train[train$Calories>2600, 'Calories'] <- 2600
test[test$Calories>2600, 'Calories'] <- 2600

#See how censoring impacts the freq dist and the regression model
hist(train$Calories, breaks=50, main="Freq Dist of Calories After Censoring")
negb_reg_cens <- glm.nb(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train, link=identity)
summary(negb_reg_cens)

#Tobit regression
library(AER)
tobit_reg <- tobit(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train, left=-Inf, right=2600)
summary(tobit_reg)

#Group outcome into intervals (this is called binning)
train$Calories <- cut(train$Calories, c(0, 2100, 2200, 2300, 2400, 2500, Inf))
test$Calories <- cut(test$Calories, c(0, 2100, 2200, 2300, 2400, 2500, Inf))
plot(train$Calories, main="Calories After Binning")

#Interval regression
library(intReg)
interval_reg <- intReg(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train, 
                       boundaries=c(0, 2100, 2200, 2300, 2400, 2500, Inf), method="probit")
summary(interval_reg)
