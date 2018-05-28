#R code for chapters 1-3 of DSILT: Statistics
#The code for each chapter is set up so that it can run independently of the other chapters

#-------------------------------------------------------------------------------------------------#
#---------------------------Chapter 1 - Experimental Design---------------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/01-05 and 09 Titanic Survival Prediction")

train <- read.csv("train.csv", header=T)
test <- read.csv("test.csv", header=T)

head(train)

#Examine the details for one variable
str(train$PassengerId)
range(train$PassengerId)

#Examine the details for all variables
str(train)

#-------------------------------------------------------------------------------------------------#
#---------------------------Chapter 2 - Descriptive Statistics------------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/01-07 and 09 Titanic Survival Prediction")

train <- read.csv("train.csv", header=T)
test <- read.csv("test.csv", header=T)

#Function for calculating the mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

length(train$Fare)  #Number of observations
mean(train$Fare)    #Mean
median(train$Fare)  #Median
Mode(train$Fare)    #Mode
var(train$Fare)     #Sample Variance
sd(train$Fare)      #Sample Standard Deviation

library(pastecs)
stat.desc(train$Fare, basic=F)

#Function for standard error of the mean
SEM <- function(x) {
  sd(x)/sqrt(length(x))
}

#Function for population variance
popVar <- function(x) {
  N <- length(x)
  var(x) * ((N-1)/N)
}

#Function for population standard deviation
popSD <- function(x) {
  N <- length(x)
  sd(x) * sqrt((N-1)/N)
}

popVar(train$Fare)  #Population Variance
popSD(train$Fare)   #Population Standard Deviation

?quantile
quantile(train$Fare, probs=seq(0, 1, 0.25))
summary(train$Fare)

#Draw a frequency distribution of passenger class
barplot(table(train$Pclass), main="Passengers by Class", xlab="Class", ylab="Number of Passengers", space=0)

#Draw the normal distribution for comparison
x <- seq(-4, 4, 0.01)
dens <- dnorm(x, 0, 1)
plot(x, dens, col="blue", xlab="Standard Deviation", ylab="Probability", type="l", lwd=2, cex=2, main="Standard Normal Distribution", cex.axis=.8)

#Other distributions

#Draw the t-distribution with 2 degrees of freedom on top of the normal distribution
denst <- dt(x, 2)
lines(x, denst, col="green", type="l", lwd=2)

#Draw the chi-square distribution with 4 degrees of freedom
x <- seq(0, 20, 0.1)
densq <- dchisq(x, df=4)
plot(x, densq, col="red", xlab="x", ylab="Probability", type="l", lwd=2, cex=2, main="Chi-Square Distribution with DOF=4", cex.axis=.8)

#Look at the probability distribution of fare
densf <- density(table(train$Fare))
plot(densf, xlab="Fare", ylab="Probability", type="l", lwd=2, cex=2, main="Probability Density of Fare", cex.axis=.8)

#Cross tabulate number of passengers by class
table(train$Pclass)
#Cross tabulate the number of passengers by class for each embarkation port
table(train$Embarked, train$Pclass)

library(gmodels)
CrossTable(train$Pclass, train$Embarked, prop.chisq=F)

#-------------------------------------------------------------------------------------------------#
#-----------------------------Chapter 3 - Statistical Modeling------------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/01-07 and 09 Titanic Survival Prediction")

train <- read.csv("train.csv", header=T)
test <- read.csv("test.csv", header=T)

#Two-tailed t-test to compare mean passenger age to mean UK population age
#First combine the training and test sets, removing the survived column and NA values for age from the training set
t_all <- rbind(train[,-2], test)
t_all <- t_all[!is.na(t_all$Age),]
xbar <- mean(t_all$Age)           #Sample mean of passenger age
mu <- 34                          #The hypothesized mean age
s <- sd(t_all$Age)                #Sample standard deviation of passenger age
n <- nrow(t_all)                  #The sample size
t <- (xbar-mu)/(s/sqrt(n))        #The t-test statistic

#Compare the t-test statistic to a confidence interval (CI between 2 critical values)
alpha <- 0.05                                #Level of significance
tdist.half.alpha <- qt(1-alpha/2, df=n-1)
c(-tdist.half.alpha, tdist.half.alpha)       #The confidence interval for alpha
t

#Alternatively, calculate the t-test statistic's p-value and compare it to a level of significance
pval <- 2*pt(t, df=n-1)
pval

#Function to perform z-tests
ztest <- function(data, mu, sig, alpha=0.05, tails='two') {
  #data is a numeric vector
  #mu is the estimated population value to compare against
  #sig is the population standard deviation
  #alpha is the significance level, defaults to 0.05
  #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
  xbar <- mean(data)
  n <- length(data)
  z <- (xbar-mu)/(sig/sqrt(n))
  if (tails=='two') {
    zdist.half.alpha <- qnorm(1-alpha/2)
    pval <- 2*pnorm(z)
    print(paste('Confidence Interval:', -zdist.half.alpha, ',', zdist.half.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else if (tails=='lower') {
    zdist.alpha <- qnorm(1-alpha)
    pval <- pnorm(z)
    print(paste('Critical Value:', -zdist.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else if (tails=='upper') {
    zdist.alpha <- qnorm(1-alpha)
    pval <- pnorm(z, lower.tail=F)
    print(paste('Critical Value:', zdist.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else {
    return (message('Error: invalid tails argument'))
  }
}
ztest.prop <- function(data, criterion, p0, alpha=0.05, tails='two') {
  #data is a numeric vector
  #criterion is a numeric vector of the number of samples that meet some criterion (a subset of data)
  #p0 is the estimated proportion to compare against
  #alpha is the significance level, defaults to 0.05
  #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
  pbar <- length(criterion)/length(data)
  n <- length(data)
  z <- (pbar-p0)/sqrt(p0*(1-p0)/n)
  if (tails=='two') {
    zdist.half.alpha <- qnorm(1-alpha/2)
    pval <- 2*pnorm(z, lower.tail=F)
    print(paste('Confidence Interval:', -zdist.half.alpha, ',', zdist.half.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else if (tails=='lower') {
    zdist.alpha <- qnorm(1-alpha)
    pval <- pnorm(z)
    print(paste('Critical Value:', -zdist.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else if (tails=='upper') {
    zdist.alpha <- qnorm(1-alpha)
    pval <- 2*pnorm(z, lower.tail=F)
    print(paste('Critical Value:', zdist.alpha))
    print(paste('z-statistic:', z))
    print(paste('z-statistic p-value:', pval))
  } else {
    return (message('Error: invalid tails argument'))
  }
}
#Function to perform t-tests
ttest <- function(data, mu, alpha=0.05, tails='two') {
  #data is a numeric vector
  #mu is the estimated population value to compare against
  #alpha is the significance level, defaults to 0.05
  #tails is the number of tails for the test, defaults to 'two', other options are 'upper' and 'lower'
  data <- data[!is.null(data)]
  xbar <- mean(data)
  s <- sd(data)
  n <- length(data)
  t <- (xbar-mu)/(s/sqrt(n))
  if (tails=='two') {
    tdist.half.alpha <- qt(1-alpha/2, df=n-1)
    pval <- 2*pt(t, df=n-1)
    print(paste('Confidence Interval:', -tdist.half.alpha, ',', tdist.half.alpha))
    print(paste('t-statistic:', t))
    print(paste('t-statistic p-value:', pval))
  } else if (tails=='lower') {
    tdist.alpha <- qt(1-alpha, df=n-1)
    pval <- pt(t, df=n-1)
    print(paste('Critical Value:', -tdist.alpha))
    print(paste('t-statistic:', t))
    print(paste('t-statistic p-value:', pval))
  } else if (tails=='upper') {
    tdist.alpha <- qt(1-alpha, df=n-1)
    pval <- pt(t, df=n-1, lower.tail=F)
    print(paste('Critical Value:', tdist.alpha))
    print(paste('t-statistic:', t))
    print(paste('t-statistic p-value:', pval))
  } else {
    return (message('Error: invalid tails argument'))
  }
}

#Run t-tests for example
ttest(t_all$Age, 34)
ttest(t_all$Age, 34, tails='lower')
ttest(t_all$Age, 34, tails='upper')

#Clean data for sex
t_all <- rbind(train[,-2], test)
t_all <- t_all[!is.na(t_all$Sex),]

#Run proportional z-test for example
ztest.prop(t_all$Sex, t_all[t_all$Sex=='male',which(colnames(t_all) %in% 'Sex')], p0=0.5)
#Inspect the sex ratio of Titanic passengers
prop.table(table(t_all$Sex))

#Validate normality of age visually...
t_all <- rbind(train[,-2], test)
t_all <- t_all[!is.na(t_all$Age),]
plot(table(t_all$Age))          #Histogram
plot(density(table(t_all$Age))) #Smoothed density plot
#...through a q-q plot...
qqnorm(t_all$Age)
qqline(t_all$Age, col=2)  #Optional trend line in red
#...and through Shapiro-Wilk and Jarque-Bera
shapiro.test(t_all$Age)
library(tseries)  #Required to run Jarque-Bera
jarque.bera.test(t_all$Age)

#Validate homogeneity of variance with Levene's test
library(car)
leveneTest(t_all$Age, t_all$Sex)
leveneTest(t_all$Age, as.factor(t_all$Pclass))
leveneTest(t_all$Age ~ t_all$Sex*as.factor(t_all$Pclass))

#Validate homogenetiy of variance with variance ratio (Hartley's F Max)
library(SuppDists)
hartleys_f_max <- function(num_variable, group_variable) {
  #num_variable is a numeric variable to compare variances
  #group_variable is the variable with the groups to compare the variance
  group_variances <- tapply(num_variable, as.factor(group_variable), var)
  f <- max(group_variances)/min(group_variances)
  pval <- pmaxFratio(f, df=length(num_variable)-1, 
                     k=length(levels(as.factor(group_variable))), 
                     lower.tail=F)
  print(paste('F-statistic:', f))
  print(paste('p-value:', pval))
}
hartleys_f_max(t_all$Age, t_all$Sex)
hartleys_f_max(t_all$Age, t_all$Pclass)

#Look at data to verify last 2 standard assumptions
head(t_all)
