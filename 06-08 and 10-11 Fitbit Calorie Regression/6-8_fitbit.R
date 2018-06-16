#R code for chapters 6 and 8 of DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#---------------------------------Chapter 6: Linear Regression------------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression")

alldata <- read.csv("fitbit.csv", header=T)
colnames(alldata) <- gsub("[.]", "_", colnames(alldata))  #Replace "." with "_" in column name - this is just personal preference

str(alldata)
head(alldata)

#Convert Date to a date
alldata$Date <- as.Date(alldata$Date, format="%m/%d/%Y")

#Convert integers to numeric
alldata$Steps <- as.numeric(alldata$Steps)
alldata$Calories <- as.numeric(alldata$Calories)
alldata$Active_Minutes <- as.numeric(alldata$Active_Minutes)
alldata$Floors_Climbed <- as.numeric(alldata$Floors_Climbed)
alldata$Times_Awake <- as.numeric(alldata$Times_Awake)

#Define a function to count the nulls in every field
naCol <- function(x){
  y <- sapply(x, function(y) {
    if (any(class(y) %in% c('Date', 'POSIXct', 'POSIXt'))) {
      sum(length(which(is.na(y) | is.nan(y) | is.infinite(y))))
    } else {
      sum(length(which(is.na(y) | is.nan(y) | is.infinite(y) | y=='NA' | y=='NaN' | y=='Inf' | y=='')))
    }
    })
  y <- data.frame('feature'=names(y), 'count.nas'=y)
  row.names(y) <- c()
  y
}

naCol(alldata)

#Multiple imputation of missing data
library(mice)
md.pattern(alldata)

mimp <- mice(alldata[,-which(colnames(alldata) %in% c('Date'))], seed=14)
imputed_vars <- complete(mimp, 1)

#Plot distributions of original data and imputed to see if imputation is on track
hist(alldata$Hours_Slept, main="Histogram of Hours Slept before Imputation")
hist(imputed_vars$Hours_Slept, main="Histogram of Hours Slept after Imputation")
hist(alldata$Times_Awake, main="Histogram of Times Awake before Imputation")
hist(imputed_vars$Times_Awake, main="Histogram of Times Awake after Imputation")

#Join imputed data back into the dataset
alldata <- cbind(Date=alldata$Date, imputed_vars)
head(alldata)
naCol(alldata)

#Boxplots for numeric variables to check for outliers
for (col in 1:ncol(alldata[,which(sapply(alldata, class) == 'numeric')])) {
  boxplot(alldata[,which(sapply(alldata, class) == 'numeric')][col], 
          main=paste0("Box Plot for ", colnames(alldata[,which(sapply(alldata, class) == 'numeric')])[col]))
}

#Inspect rows with potential outliers
head(alldata[alldata$Times_Awake>30,])
head(alldata[alldata$Hours_Slept<6,])
head(alldata[alldata$Floors_Climbed>100,])
head(alldata[alldata$Active_Minutes>100,])

#Drop date and look at correlation matrix
nodate <- alldata
nodate$Date <- NULL
allCor <- cor(nodate, use="pairwise.complete.obs")
library('corrplot')
corrplot(allCor, method="circle")

#Drop distance
nodate$Distance <- NULL

######
#Testing assumptions

#Are the predictors independent?
colnames(nodate[,-2])  #Yes - there's no reason to think any of these depend on the others

#Are all predictors quantitative or categorical with no more than 2 categories?
str(nodate[,-2])  #Yes - assumption met

#Do the predictors have non-zero variance?
summary(nodate[,-2])  #Yes - assumption met

#Is there multicollinearity among the predictors?
cor(nodate[,-2], use="pairwise.complete.obs")  #No - assumption met

#Might the predictors correlate with variables that are not in dataset?
#Possibly - date was dropped, but steps and activity might be correlated with season, since summer months are more conducive to activity

#Are the residuals homoskedastic?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals normally distributed?
#Cannot tell yet - need to build the model to get the residuals

#Are the residuals autocorrelated?
#Cannot tell yet - need to build the model to get the residuals

######
#Modeling

#Add features for the days of the week, using Friday as the baseline
library(caret)
alldata$Day <- as.factor(weekdays(alldata$Date))
dummies <- dummyVars( " ~ Day", data=alldata, fullRank=T)
dummyenc <- data.frame(predict(dummies, newdata=alldata))
colnames(dummyenc) <- gsub("[.]", "_", colnames(dummyenc))
alldata <- cbind(alldata, dummyenc)
nodate <- cbind(nodate, dummyenc)
head(nodate)

#Split data into training and test sets
set.seed(14)
train_indices <- createDataPartition(y=nodate$Calories, p=0.7, list=F)
train <- nodate[train_indices,]
test <- nodate[-train_indices,]
write.csv(train, 'train.csv', row.names=F)
write.csv(test, 'test.csv', row.names=F)

#Build linear model
linear_reg <- lm(Calories ~ ., data=train)  #Note that "." is short in R for "everything"
summary(linear_reg)
#plot(linear_reg)

#Testing normally distributed residuals assumption
mean(linear_reg$residuals)  #This should be near zero if the assumption of residual normality holds
library(car)
qqPlot(linear_reg, main="QQ Plot")
library(MASS)
sresid <- studres(linear_reg)
hist(sresid, freq=F, main="Distribution of Studentized Residuals Compared to Normal Dist")
normfitx <- seq(min(sresid), max(sresid), length=40)
normfity <- dnorm(normfitx)
lines(normfitx, normfity) 

library(gvlma)
gvmodel <- gvlma(linear_reg)
summary(gvmodel) 

#VIF test for multicollinearity
vif(linear_reg)

#DW test for autocorrelation - note this uses the car package
durbinWatsonTest(linear_reg)

#Correlogram to look for significant lags in the residuals
acf(linear_reg$residuals)

#Test idea that weekends were more active and therefore could cause serial correlation
#View average steps by day in bar chart
library(ggplot2)
ggplot(alldata, aes(Day, Steps)) + stat_summary(fun.y=mean, geom="bar", position="dodge", fill="#56B4E9") + labs(x="Day of Week", y="Average Steps") + ggtitle("Average Steps by Day of Week")
ggplot(alldata, aes(Day, Active_Minutes)) + stat_summary(fun.y=mean, geom="bar", position="dodge", fill="#56B4E9") + labs(x="Day of Week", y="Average Active Minutes") + ggtitle("Average Active Minutes by Day of Week")

#Get the Newey-West HAC estimates of the model
library(sandwich)
library(lmtest)
coeftest(linear_reg, vcov.=NeweyWest)

#Remove unnecessary variables from the model to see the result
reduced_linear_reg <- lm(Calories ~ Steps + Floors_Climbed + Times_Awake + Day_Saturday + Day_Sunday, data=train)
summary(reduced_linear_reg)
AIC(linear_reg)
AIC(reduced_linear_reg)
BIC(linear_reg)
BIC(reduced_linear_reg)

#Predict the calories burnt in the test set, using the fitted model
test$Pred <- round(predict(reduced_linear_reg, test[,-2]), 0)  #Rounding since target var is rounded to whole number
mae <- mean(abs(test$Calories-test$Pred))
rmse <- sqrt(mean((test$Calories-test$Pred)^2))
baseline_model <- mean(train$Calories)
mae_baseline <- mean(abs(test$Calories-baseline_model))
rmse_baseline <- sqrt(mean((test$Calories-baseline_model)^2))

#-------------------------------------------------------------------------------------------------#
#-----------------------------Chapter 8: Hyperparameter Optimization------------------------------#
#-------------------------------------------------------------------------------------------------#

setwd("/home/dsilt/Desktop/dsilt-stats-code/06-08 and 10-11 Fitbit Calorie Regression")

train <- read.csv('train.csv', header=T)
test <- read.csv('test.csv', header=T)

gd <- function(df, target_var, cost, learning_rate, num_iters, test_set=NULL) {
  #Split data and target variable
  x <- df[,which(colnames(df) != target_var)]
  y <- df[,target_var]
  #These will keep a running history of cost and the coefficient estimates
  cost_history <- double(num_iters)
  coef_matrix_history <- list(num_iters)
  #Initialize coefficient estimates (a 1 column matrix of k predictors + 1 rows)
  coef_matrix <- matrix(c(0,0), nrow=ncol(x)+1)
  #Convert the input data to matrix form and bind it to a column of 1's for the intercept coefficient
  x_matrix <- cbind(1, as.matrix(x))
  #Implement gradient descent
  for (i in 1:num_iters) {
    error <- (x_matrix %*% coef_matrix) - y
    gradient <- (t(x_matrix) %*% error) %*% (1/length(y))
    coef_matrix <- coef_matrix - (learning_rate * gradient)
    cost_history[i] <- cost(x_matrix, y, coef_matrix)
    coef_matrix_history[[i]] <- coef_matrix
  }
  #Make predictions if test set was provided
  if (!is.null(test_set)) {
    preds <- coef_matrix[1]
    for (i in seq_along(1:ncol(x))) {
      preds <- preds + coef_matrix[i+1]*x[,i]
    }
  } else {
    preds <- "Cannot make predictions because no test set was provided to the function."
  }
  return (list(cost_hist=cost_history, final_cost=tail(cost_history, n=1), coef_hist=coef_matrix_history, coef=coef_matrix, preds=preds))
}

sgd <- function(df, target_var, cost, learning_rate, num_iters, batch_size, seed=14, test_set=NULL) {
  #Split data and target variable
  x <- df[,which(colnames(df) != target_var)]
  y <- df[,target_var]
  #These will keep a running history of cost and the coefficient estimates
  cost_history <- double(num_iters)
  coef_matrix_history <- list(num_iters)
  #Initialize coefficient estimates (a 1 column matrix of k predictors + 1 rows)
  coef_matrix <- matrix(c(0,0), nrow=ncol(x)+1)
  #Convert the input data to matrix form and bind it to a column of 1's for the intercept coefficient
  x_matrix <- cbind(1, as.matrix(x))
  #Implement stochastic gradient descent
  for (i in 1:num_iters) {
    sample_indices <- sample(nrow(x), batch_size)
    x_matrix_sample <- x_matrix[sample_indices, , drop=F]
    y_sample <- y[sample_indices, drop=F]
    error <- (x_matrix_sample %*% coef_matrix) - y_sample
    gradient <- (t(x_matrix_sample) %*% error) / length(y_sample)
    coef_matrix <- coef_matrix - (learning_rate * gradient)
    cost_history[i] <- cost(x_matrix_sample, y_sample, coef_matrix)
    coef_matrix_history[[i]] <- coef_matrix
  }
  #Make predictions if test set was provided
  if (!is.null(test_set)) {
    preds <- coef_matrix[1]
    for (i in seq_along(1:ncol(x))) {
      preds <- preds + coef_matrix[i+1]*x[,i]
    }
  } else {
    preds <- "Cannot make predictions because no test set was provided to the function."
  }
  return (list(cost_hist=cost_history, final_cost=tail(cost_history, n=1), coef_hist=coef_matrix_history, coef=coef_matrix, preds=preds))
}

#Define the cost function to be MSE
mse_cost <- function(x_matrix, y, coef_matrix) {
  sum((y - (x_matrix %*% coef_matrix))^2)/length(y)
}

#standardize the numeric columns of the dataset to mean 0
train_std <- cbind(train[,-which(colnames(train) %in% c('Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept', 'Times_Awake'))], scale(train[,which(colnames(train) %in% c('Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept', 'Times_Awake'))]))
test_std <- cbind(test[,-which(colnames(test) %in% c('Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept', 'Times_Awake'))], test[,which(colnames(test) %in% c('Steps', 'Calories', 'Active_Minutes', 'Floors_Climbed', 'Hours_Slept', 'Times_Awake'))])

#Perform gradient descent and stochastic gradient descent with hardcoded hyperparameters
#Be sure to remove the target from the test set when passing it as the test_set argument
set.seed(14)
linear_reg_gd <- gd(train_std, 'Calories',
                    cost=mse_cost, learning_rate=0.01, num_iters=500,
                    test_set=test_std[,which(colnames(test) != 'Calories')])
linear_reg_sgd <- sgd(train_std, 'Calories', 
                      cost=mse_cost, learning_rate=0.01, num_iters=500, batch_size=50, 
                      test_set=test_std[,which(colnames(test) != 'Calories')])

linear_reg_gd$final_cost
linear_reg_sgd$final_cost
linear_reg_gd$coef
linear_reg_sgd$coef

#Plot the cost function over time
plot(linear_reg_gd$cost_hist, type='line', col='blue', lwd=2, 
     main='Cost function for gradient descent', ylab='cost', xlab='Iterations')
plot(linear_reg_sgd$cost_hist, type='line', col='blue', lwd=2, 
     main='Cost function for stochastic gradient descent', ylab='cost', xlab='Iterations')

library(caret)
cv_folds <- createFolds(train_std$Calories, k=5)
linear_reg_gd_cv <- function(lrate, niters=50) {
  #Manually perform cross validation
  cv_scores <- list()
  cv_preds <- list()
  for (fold in cv_folds) {
    trainingdata <- unlist(cv_folds[-fold])  #Use the other folds for training
    #print(nrow(trainingdata))
    fold_res <- gd(train_std, 'Calories', 
                   cost=mse_cost, learning_rate=lrate, num_iters=niters, 
                   test_set=train_std)
    cv_scores <- c(cv_scores, list(fold_res$final_cost))
    cv_preds <- c(cv_preds, list(fold_res$preds))
  }
  #Return a list of the average score for the k folds and the predictions of the fold with the best score
  return (list(Score=mean(unlist(cv_scores)), Pred=unlist(cv_preds[which(cv_scores==max(unlist(cv_scores)))])))
}

#Grid search best hyperparameters
bestScore <- 100
bestParams <- list()
for (i in c(0.001, 0.01, 0.1)) {
  for (j in c(10, 100, 500)) {
    res <- linear_reg_gd_cv(lrate=i, niter=j)
    if (res$Score < bestScore) {
      bestScore <- res$Score
      bestParams <- list(lrate=i, niter=j)
    }
  }
}

bestScore
bestParams

#Random search best hyperparameters
bestScore <- 100
bestParams <- list()
hyperparams <- list(lrate=seq(1:100)/1000, niters=seq(from=10, to=500, by=10))
for (i in seq(1:30)) {
  set.seed(14)
  lr <- sample(hyperparams$lrate, 1)
  n <- sample(hyperparams$niters, 1)
  res <- linear_reg_gd_cv(lrate=lr, niters=n)
  if (res$Score < bestScore) {
    bestScore <- res$Score
    bestParams <- list(lrate=lr, niter=n)
  }
}

bestScore
bestParams


