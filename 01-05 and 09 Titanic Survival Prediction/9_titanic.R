#R code for chapter 9 of DSILT: Statistics

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/01-05 and 09 Titanic Survival Prediction")

train <- read.csv("train_clean.csv", header=T)
test <- read.csv("test_clean.csv", header=T)
train$Set <- 'train'
test$Set <- 'test'
alldata <- rbind(train[,-which(colnames(train) %in% 'Survived')], test)

#-------------------------------------------------------------------------------------------------#
#-----------------------------------Final Feature Engineering-------------------------------------#
#-------------------------------------------------------------------------------------------------#

str(alldata)
head(alldata)

#Add calculated family size (siblings + spouse + parents + children + 1 for self)
alldata$Family_Size <- alldata$SibSp + alldata$Parch + 1

#Add simple binary variable to indicate whether a passenger had a family
alldata$Has_Family <- as.factor(ifelse(alldata$Family_Size > 1, 1, 0))

#Isolate titles from names
alldata$Name <- as.character(alldata$Name)
alldata$Title <- sapply(alldata$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
#Remove the first space in the title
alldata$Title <- sub(' ', '', alldata$Title)

table(alldata$Title)

#Combine similar titles that mean the same thing
alldata$Title[alldata$Title %in% c('Dona', 'Jonkheer', 'Lady', 'Mlle', 'Mme', 'the Countess')] <- 'Mme'
alldata$Title[alldata$Title %in% c('Capt', 'Col', 'Don', 'Sir', 'Major')] <- 'Sir'
alldata$Title[alldata$Title %in% c('Miss', 'Ms')] <- 'Miss'
alldata$Title <- factor(alldata$Title)

alldata$Missing_Age <- ifelse(is.na(alldata$Age), 1, 0)

#Create distributions to sample from, and then perform piecewise single random imputation
age_dist_child <- alldata[!is.na(alldata$Age) & alldata$Age < 16, 'Age']
age_dist_adult <- alldata[!is.na(alldata$Age) & alldata$Age >= 16, 'Age']

hist(alldata$Age, main="Histogram of Age Before Piecewise Single Random Imputation")
alldata[is.na(alldata$Age) & alldata$Title %in% c('Master', 'Miss'), 'Age'] <- sapply(alldata[is.na(alldata$Age) & alldata$Title %in% c('Master', 'Miss'), 'Age'], FUN=function(x) {sample(age_dist_child, 1)})
alldata[is.na(alldata$Age), 'Age'] <- sapply(alldata[is.na(alldata$Age), 'Age'], FUN=function(x) {sample(age_dist_adult, 1)})
hist(alldata$Age, main="Histogram of Age After Piecewise Single Random Imputation")

#Child indicator
alldata$Is_Child <- ifelse(alldata$Age < 16, 1, 0)

#Dummy encode title - be sure to set fullRank=T to dummy encode instead of one-hot
library(caret)
dummies <- dummyVars( " ~ Title", data=alldata, fullRank=T)
dummyenc <- data.frame(predict(dummies, newdata=alldata))
alldata <- cbind(alldata, dummyenc)
colnames(alldata) <- gsub("[.]", "_", colnames(alldata))
rm('dummies', 'dummyenc')

#Remove the title column and one column from each group of the one-hot encoded column groups
alldata$Title <- NULL
alldata$Sex_female <- NULL
alldata$Embarked_C <- NULL
alldata$DeckA <- NULL

#Split back into training and test sets
train_clean_feats <- alldata[alldata$Set=='train',]
train_clean_feats$Set <- NULL
train_clean_feats$Survived <- train$Survived
test_clean_feats <- alldata[alldata$Set=='test',]
test_clean_feats$Set <- NULL
write.csv(train_clean_feats, file="train_clean_feats.csv", row.names=F)
write.csv(test_clean_feats, file="test_clean_feats.csv", row.names=F)

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Logistic Regression----------------------------------------#
#-------------------------------------------------------------------------------------------------#

train <- read.csv("train_clean_feats.csv", header=T)
test <- read.csv("test_clean_feats.csv", header=T)
rm('train_clean_feats', 'test_clean_feats', 'age_dist_child', 'age_dist_adult')

######
#Testing assumptions

#Are all predictors quantitative or categorical with no more than 2 categories?
str(train)  #Oops!  Forgot to get rid of name
train$Name <- NULL
test$Name <- NULL
str(train)  #Yes - assumption met
str(test)   #Yes - assumption met

#Is there a linear relationship between the predictors and the logit?
#Cannot tell yet - need to build the model to get the observed vs predicted probabilities

#Is there multicollinearity among the predictors?
corm <- cor(train, use="pairwise.complete.obs")  
library(corrplot)
corrplot(corm, method="circle")    #Possible problems here

#Are the residuals homoskedastic?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals autocorrelated?
#Cannot tell yet - need to build the model to get the residuals

######
#Modeling

#Build logit model
logistic_reg <- glm(Survived ~ ., family=binomial(link='logit'), data=train)
summary(logistic_reg)

#Get rid of the highly correlated variables and try again
train <- train[,-which(colnames(train) %in% c('Fare', 'Family_Size', 'DeckZ', 
                                              'Title_Master', 'Title_Miss', 
                                              'Title_Mme', 'Title_Mr', 
                                              'Title_Mrs', 'Title_Rev', 
                                              'Title_Sir'))]
test <- test[,-which(colnames(test) %in% c('Fare', 'Family_Size', 'DeckZ', 
                                           'Title_Master', 'Title_Miss', 
                                           'Title_Mme', 'Title_Mr', 
                                           'Title_Mrs', 'Title_Rev', 
                                           'Title_Sir'))]
corm <- cor(train, use="pairwise.complete.obs")  
corrplot(corm, method="circle")
logistic_reg <- glm(Survived ~ ., family=binomial(link='logit'), data=train)
summary(logistic_reg)
#plot(logistic_reg)

anova(logistic_reg, test="Chisq")

#New model without insignificant predictors
new_logistic_reg <- glm(Survived ~ Pclass + Age + SibSp + Parch + Group_Size + 
                          Fare_Per_Person + Sex_male + DeckE + Ticket_Enc + 
                          Has_Family + Missing_Age + Is_Child, 
                        family=binomial(link='logit'), data=train)
summary(new_logistic_reg)
anova(new_logistic_reg, logistic_reg, test="Chisq")

#VIF test for multicollinearity
library(car)
vif(logistic_reg)

#DW test for autocorrelation - note this uses the car package
durbinWatsonTest(logistic_reg)

#Correlogram to look for significant lags in the residuals
acf(logistic_reg$residuals)

######
#Evaluating the model on new data

#Split training data into training and validation sets
set.seed(14)
train_indices <- createDataPartition(y=train$Survived, p=0.7, list=F)
train_new <- train[train_indices,]
validation <- train[-train_indices,]
logistic_reg <- glm(Survived ~ ., family=binomial(link='logit'), data=train_new)

#Make survival predictions for validation set
validation$Pred <- predict(logistic_reg, validation[,-21], type="response")
validation$Survived_Pred <- ifelse (validation$Pred >= 0.5, 1, 0)

#Evaluate the model using the log loss, AUC, and accuracy
library(MLmetrics)
LogLoss(validation$Survived_Pred, validation$Survived)
AUC(validation$Survived_Pred, validation$Survived)
Accuracy(validation$Survived_Pred, validation$Survived)
#View the confusion matrix with predicted values on the left
prop.table(table(validation$Survived_Pred, validation$Survived))

