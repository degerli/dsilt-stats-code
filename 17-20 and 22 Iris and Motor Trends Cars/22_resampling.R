#R code for chapter 22 of DSILT: Statistics

d <- iris
str(d)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Bootstrapping---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Take a sample of the dataset for comparison of bootstrap estimate
ds <- d[sample(nrow(d), size=50),]

library(boot)
#Bootstrap estimate of mean (k=1 for 1 statistic)
average <- function(data, indices) {
  return(mean(data[indices]))
}
bootresults <- boot(data=ds$Sepal.Length, statistic=average, R=1000)
plot(bootresults)
#Compare bootstrapped estimate to the sample mean and the population mean
mean(ds$Sepal.Length)
mean(d$Sepal.Length)
#Bootstrapped CI
boot.ci(bootresults, type="bca")

#Bootstrap cofficient estimates of regression (k>1 for many statistics)
linear_reg_coef <- function(formula, data, indices) {
  d <- data[indices,]
  linear_reg <- lm(formula, data=d)
  return(coef(linear_reg))
}
bootresults <- boot(data=mtcars, statistic=linear_reg_coef, R=1000, formula=mpg~wt+disp)
plot(bootresults, index=1)  #Intercept
plot(bootresults, index=2)  #wt coef
plot(bootresults, index=3)  #disp coef
#Bootstrapped CI
boot.ci(bootresults, type="bca", index=1)  #Intercept
boot.ci(bootresults, type="bca", index=2)  #wt coef
boot.ci(bootresults, type="bca", index=3)  #disp coef

#Bootstrapping a model with caret
library(caret)
train_control <- trainControl(method="boot", number=100)
nb_class <- train(Species~., data=d, trControl=train_control, method="nb")
nb_class

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Cross Validation------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#k-fold cross validation
library(caret)
library(rpart)
train_control <- trainControl(method="cv", number=10, savePredictions=TRUE)
class_tree <- train(Species~., data=d, trControl=train_control, method="rpart")

#k-fold cross validation with hyperparameter tuning
hyperparam_grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE), .adjust=0)
#Train & tune
nb_class <- train(Species~., data=d, trControl=train_control, method="nb", tuneGrid=hyperparam_grid)
nb_class

#Repeated k-fold cross validation with hyperparameter tuning for regularized regression
library(glmnet)
glmnet_grid <- expand.grid(alpha=c(0, 0.1, 0.2, 0.4, 0.6, 0.8, 1), lambda=seq(0.01, 0.2, length=20))
#glmnet_ctrl <- trainControl(method="cv", number=10)             #k-fold without repeating
glmnet_ctrl <- trainControl(method="repeatedcv", number=10, repeats=3)  #k-fold with 3 repeats
glmnet_fit <- train(Species~., data=d, method="glmnet", preProcess=c("center", "scale"), tuneGrid=glmnet_grid, trControl=glmnet_ctrl)
glmnet_fit
