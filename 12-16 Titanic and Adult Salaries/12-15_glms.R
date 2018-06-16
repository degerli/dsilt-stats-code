#R code for chapter 12-15 of DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#------------------------------------Chapter 12: GLMs---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries")

train <- read.csv("train_clean_feats.csv", header=T)
test <- read.csv("test_clean_feats.csv", header=T)
train$Set <- 'train'
test$Set <- 'test'
alldata <- rbind(train[,-which(colnames(train) %in% 'Survived')], test)

str(alldata)
head(alldata)

######
#t-test

#Compare the mean fare per person between males and females
t.test(alldata$Fare_Per_Person~alldata$Sex_male)

#Compare the mean fare per person with the mean fare
t.test(alldata$Fare_Per_Person, alldata$Fare)

######
#One-Way ANOVA

alldata$Group_Size <- as.factor(alldata$Group_Size)

#Test for heteroskedasticity across groups
library(car)
leveneTest(alldata$Fare_Per_Person, alldata$Group_Size, center=median)
bartlett.test(alldata$Fare_Per_Person, alldata$Group_Size)

#Heteroskedasticity is present, so run the code below for Welch's F
oneway.test(Fare_Per_Person ~ Group_Size, data=alldata)

#Check to make sure none of the groups have only one distinct value
aggregate(data=alldata, Fare_Per_Person ~ Group_Size, function(x) length(unique(x)))

#Re-run oneway.test omitting group of size 11 to compare mans by group size
oneway.test(Fare_Per_Person ~ Group_Size, data=alldata[alldata$Group_Size != 11,])

#Add embarkation port c back into the data and recreate the original embarkation port variable
alldata$Embarked_C <- ifelse(alldata$Embarked_Q + alldata$Embarked_S == 0, 1, 0)
alldata$Embarked <- 0
alldata[alldata$Embarked_Q==1, 'Embarked'] <- 'Q'
alldata[alldata$Embarked_S==1, 'Embarked'] <- 'S'
alldata[alldata$Embarked_C==1, 'Embarked'] <- 'C'
alldata$Embarked <- as.factor(alldata$Embarked)

#Perform ANOVA to compare means by embarkation port
anova_reg <- aov(Fare_Per_Person ~ Embarked, data=alldata)
summary(anova_reg)

#Check for heteroskedasticity
qqPlot(anova_reg, main="QQ Plot")  #From car package

#Redo ANOVA with Welch's F to account for heteroskedasticity
oneway.test(Fare_Per_Person ~ Embarked, data=alldata)

######
#Post Hoc Tests for One-way ANOVA

#Tukey test - good when groups are the same size and have and homogeneous variance
library(multcomp)
postHocs <- glht(anova_reg, linfct=mcp(Embarked="Tukey"))
summary(postHocs)
confint(postHocs)

#Pairwise comparison using Bonferroni correction of p-values
pairwise.t.test(alldata$Fare_Per_Person, alldata$Embarked, p.adjust.method="bonferroni")

#Pairwise comparison using Benjamini-Hochberg correction of p-values
pairwise.t.test(alldata$Fare_Per_Person, alldata$Embarked, p.adjust.method="BH")

######
#ANCOVA

#Look for heteroskedasticity
plot(alldata[alldata$Pclass==2 & alldata$Sex_male==1, 'Fare_Per_Person'], alldata[alldata$Pclass==2 & alldata$Sex_male==1, 'Group_Size'])
#Second class male passengers with a fare price > 0 seem OK
#There are a couple group sizes with only 1 observation with these criteria though, so make sure to filter them out too

#Test for heteroskedasticity
leveneTest(alldata[alldata$Pclass==2 & alldata$Sex_male==1 & alldata$Fare>0 & !(alldata$Group_Size %in% c(5, 6, 7)), 'Fare_Per_Person'], alldata[alldata$Pclass==2 & alldata$Sex_male==1 & alldata$Fare>0 & !(alldata$Group_Size %in% c(5, 6, 7)), 'Group_Size'], center=median)
bartlett.test(alldata[alldata$Pclass==2 & alldata$Sex_male==1 & alldata$Fare>0 & !(alldata$Group_Size %in% c(5, 6, 7)), 'Fare_Per_Person'], alldata[alldata$Pclass==2 & alldata$Sex_male==1 & alldata$Fare>0 & !(alldata$Group_Size %in% c(5, 6, 7)), 'Group_Size'])

sub <- alldata[alldata$Pclass==2 & alldata$Sex_male==1 & alldata$Fare>0 & !(alldata$Group_Size %in% c(5, 6, 7)), ]
head(sub)

#Show ANOVA to see how ANCOVA is different
anova_reg <- aov(Fare_Per_Person ~ Group_Size, data=sub)
summary(anova_reg)
postHocs <- glht(anova_reg, linfct=mcp(Group_Size="Tukey"))
summary(postHocs)
confint(postHocs)
#PostHocs show that fare per person for groups sizes of 1 and 2 are different from the rest

library(ggplot2)
boxplot <- ggplot(sub, aes(Group_Size, Fare_Per_Person)) + geom_boxplot() + xlab('Group Size') + ylab('Fare Per Person')
boxplot
#Box plot confirms what the post hoc tests reported

#Create the ANCOVA regression
ancova_reg <- aov(Fare_Per_Person ~ Group_Size + Age, data=sub)
#Print the model summary with type III sums of squares
Anova(ancova_reg, type="III")

#Cannot yet interpret group means because the effect of the covariate hasn't been adjusted for, so look at adjusted means using the effect function
library(effects)
adjustedMeans <- effect("Group_Size", ancova_reg, se=TRUE)
summary(adjustedMeans)
adjustedMeans$se

######
#Post Hoc Tests for ANCOVA

#Testing differences between adjusted means requires using glht() instead of pairwise.t.test()
#Using glht() limits post hoc tests to Tukey or Dunnett's tests
library(multcomp)
postHocs <- glht(ancova_reg, linfct=mcp(Group_Size="Tukey"))
summary(postHocs)
confint(postHocs)

plot(ancova_reg)

#Add interaction term to ANCOVA model to look for homogeneity in the regression slopes
ancova_reg_homo_rs <- aov(Fare_Per_Person ~ Group_Size + Age + Group_Size:Age, data=sub)

#Get the type III sums of squares
Anova(ancova_reg_homo_rs, type="III")

######
#Chi-Square Test

#Perform Chi-Square test to compare passenger class by embarkation port
library(gmodels)
CrossTable(alldata$Pclass, alldata$Embarked, expected=TRUE, prop.c=FALSE, 
           prop.t=FALSE, prop.chisq=FALSE, chisq=TRUE, sresid=TRUE, 
           format="SPSS")

######
#Loglinear Analysis

#Create contingency table of the categorical variables to be analyzed
contTable <- xtabs(~ Pclass + Sex_male + Embarked, data=alldata)

mosaicplot(contTable, shade=TRUE, main="Pclass, Sex, and Embarked")

#Start off with saturated loglinear model with all interactions
#Note that A*B is a shorcut for specifying all possible interactions between A and B
loglinear_reg <- loglm(~ Pclass*Sex_male*Embarked, data=contTable, fit=TRUE)
summary(loglinear_reg)

#Remove the highest order interaction effect and compare models
loglinear_reg_red <- update(loglinear_reg, .~. -Pclass:Sex_male:Embarked)
summary(loglinear_reg_red)
#Compare models by subtracting the likelihood ratios and DOF, or just use the anova function to do the LR test automatically
anova(loglinear_reg, loglinear_reg_red)

#Continue removing other interaction terms until the p-value of the reduced model is significant (< 0.05)
loglinear_reg_red_further <- update(loglinear_reg_red, .~. -Sex_male:Embarked)
anova(loglinear_reg, loglinear_reg_red_further)

#Final model analysis
summary(loglinear_reg_red)
mosaicplot(loglinear_reg_red$fit, shad=TRUE, 
           main="Loglinear Model of Pclass, Sex, and Embarked, without 3-Way Interaction")

#-------------------------------------------------------------------------------------------------#
#-------------------------------Chapter 13: Factorial ANOVA---------------------------------------#
#-------------------------------------------------------------------------------------------------#

d <- read.table("adult.data", header=F, sep=",")
colnames(d) <- c('age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 
                 'marital_status', 'occupation', 'relationship', 'race', 
                 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
                 'native_country', 'salary_bin')
str(d)
head(d)

d$hours_per_week <- as.numeric(d$hours_per_week)

#Test for heteroskedasticity across groups
library(car)
leveneTest(d$hours_per_week, d$education, center=median)
leveneTest(d$hours_per_week, d$relationship, center=median)
bartlett.test(d$hours_per_week, d$education)
bartlett.test(d$hours_per_week, d$relationship)

library(ggplot2)
boxplot <- ggplot(d, aes(education, hours_per_week)) + geom_boxplot() + xlab('Education') + ylab('Hours Per Week')
boxplot

#Ignore the heteroskedasticity for now, and proceed
tw_ind_anova_reg <- aov(hours_per_week ~ education + relationship + education:relationship, data=d)
Anova(tw_ind_anova_reg, type='III')

#Post Hoc tests for education only
library(multcomp)
pairwise.t.test(d$hours_per_week, d$education, p.adjust.method="bonferroni")
pairwise.t.test(d$hours_per_week, d$education, p.adjust.method="BH")
postHocs <- glht(tw_ind_anova_reg, linfct=mcp(education="Tukey"))
summary(postHocs)
confint(postHocs)

#-------------------------------------------------------------------------------------------------#
#-----------------------------Chapter 14: Nonparametric Tests-------------------------------------#
#-------------------------------------------------------------------------------------------------#

d <- read.table("adult.data", header=F, sep=",")
colnames(d) <- c('age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 
                 'marital_status', 'occupation', 'relationship', 'race', 
                 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
                 'native_country', 'salary_bin')
str(d)
head(d)

d$hours_per_week <- as.numeric(d$hours_per_week)

######
#Kruskal-Wallis test

kw_test <- kruskal.test(hours_per_week ~ education, data=d)
#If p value for test stat (H or chi-squared) is < 0.05, then the independent var does significantly affect the outcome
#Post hoc tests are needed to see which groups were responsible for the diff

#Obtain the mean rank per group
d$ranks <- rank(d$hours_per_week)
by(d$ranks, d$education, mean)

#The function below shows differences between the mean ranks for the dataset
library(pgirmess)
kwmc <- kruskalmc(hours_per_week ~ education, data=d, cont='two-tailed')

#Post Hoc tests
library(multcomp)
pairwise.t.test(d$hours_per_week, d$education, p.adjust.method="bonferroni")
pairwise.t.test(d$hours_per_week, d$education, p.adjust.method="BH")
postHocs <- glht(kw_test, linfct=mcp(education="Tukey"))
summary(postHocs)
confint(postHocs)

######
#Jonckheere-Terpstra test

#Jonckheere test to look for trends across groups (must order the independent variable in terms of an expected increasing or decreasing trend)
library(clinfun)
d <- d[with(d, order(education)),]
jonckheere.test(d$hours_per_week, as.numeric(d$education))

######
#Wilcoxon rank-sum test

wilcox_ranksum <- wilcox.test(hours_per_week ~ sex, data=d)
wilcox_ranksum

#-------------------------------------------------------------------------------------------------#
#-------------------------Chapter 15: LDA and QDA for Classification------------------------------#
#-------------------------------------------------------------------------------------------------#

d <- read.table("adult.data", header=F, sep=",")
colnames(d) <- c('age', 'workclass', 'fnlwgt', 'education', 'education_nbr', 
                 'marital_status', 'occupation', 'relationship', 'race', 
                 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
                 'native_country', 'salary_bin')
str(d)
head(d)

######
#Data Cleaning

#Convert variables to desired data types
d$age <- as.numeric(d$age)
d$fnlwgt <- as.numeric(d$fnlwgt)
d$education_nbr <- as.factor(d$education_nbr)
d$capital_gain <- as.numeric(d$capital_gain)
d$capital_loss <- as.numeric(d$capital_loss)
d$hours_per_week <- as.numeric(d$hours_per_week)

#Define a function to count the nulls in every field
naCol <- function(x){
  y <- sapply(x, function(y) {
    if (any(class(y) %in% c('Date', 'POSIXct', 'POSIXt'))) {
      sum(length(which(is.na(y) | is.nan(y) | is.infinite(y))))
    } else {
      sum(length(which(is.na(y) | is.nan(y) | is.infinite(y) | y=='NA' | y=='NaN' | y=='Inf' | y=='' | y==' ?')))
    }
  })
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

naCol(d)

#Since there are 32k rows and <2k rows with nulls, it is safe to discard them
d <- d[(d$workclass!=' ?' & d$occupation!=' ?' & d$native_country!=' ?'),]
d$workclass <- factor(d$workclass)
d$occupation <- factor(d$occupation)
d$native_country <- factor(d$native_country)

#Boxplots for numeric variables to check for outliers
for (col in 1:ncol(d[,which(sapply(d, class) == 'numeric')])) {
  boxplot(d[,which(sapply(d, class) == 'numeric')][col], 
          main=paste0("Box Plot for ", colnames(d[,which(sapply(d, class) == 'numeric')])[col]))
}

#Look at correlation matrix
allCor <- cor(d[,which(sapply(d, class) == 'numeric')], use="pairwise.complete.obs")
library('corrplot')
corrplot(allCor, method="circle")

#Dummy encode categorical variables
d[,which(sapply(d, class)=='factor')] <- sapply(d[,which(sapply(d, class)=='factor')], substring, 2, 100)  #Gets rid of leading whitespaces that show up in factor levels
d[,which(sapply(d, class)=='character')] <- lapply(d[,which(sapply(d, class)=='character')], as.factor)
library(caret)
cols_to_encode <- c('workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country')
for (c in cols_to_encode) {
  dummies <- dummyVars(paste0(" ~ ", c), data=d, fullRank=T)
  dummyenc <- data.frame(predict(dummies, newdata=d))
  d <- cbind(d, dummyenc)
}
colnames(d) <- gsub("[.]", "_", colnames(d))
str(d)
#Get rid of redundant columns now that they have been dummy encoded
d$workclass <- NULL
d$education <- NULL
d$education_nbr <- as.numeric(d$education_nbr)
d$marital_status <- NULL
d$occupation <- NULL
d$relationship <- NULL
d$race <- NULL
d$sex <- NULL
d$native_country <- NULL

#Split data into training and test sets
set.seed(14)
train_indices <- createDataPartition(y=d$salary_bin, p=0.7, list=F)
train <- d[train_indices,]
test <- d[-train_indices,]
write.csv(train, 'adult_train.csv', row.names=F)
write.csv(test, 'adult_test.csv', row.names=F)

######
#LDA

library(MASS)
lda_model <- lda(formula=salary_bin ~ ., data=train)
lda_model

#Explore the percentage of between class variance explained by each linear discriminant
lda_model$svd^2/sum(lda_model$svd^2)

######
#Evaluating the model on new data

#Make income predictions for validation set
post_lda <- predict(lda_model, test[,-7])
summary(post_lda)
test$pred <- post_lda$class
test$baseline <- '<=50K'  #Baseline assigns everything to majority class

#Evaluate the model using the log loss, AUC, and accuracy
library(MLmetrics)
LogLoss(as.integer(test$pred), as.integer(test$salary_bin))
AUC(as.integer(test$pred), as.integer(test$salary_bin))
Accuracy(test$pred, test$salary_bin)
Accuracy(test$baseline, test$salary_bin)
#View the confusion matrix with predicted values on the left
prop.table(table(test$pred, test$salary_bin))

######
#QDA

qda_model <- qda(formula=salary_bin ~ ., data=train)

#Look at complete correlation matrix
allCor <- cor(d[,which(sapply(d, class) == 'numeric')], use="pairwise.complete.obs")
allCor > 0.6
allCor < (-0.6)
