#R code for chapter 19 of DSILT: Statistics

d <- mtcars
str(d)

#Standard linear regression for comparison
linear_reg <- lm(mpg ~ ., data=d)
summary(linear_reg)

library(glmnet)
?glmnet
#glmnet function is the same for ridge, lasso, and elastic net
#alpha=0 for ridge, alpha=1 for lasso, alpha=0.5 for elastic net

#-------------------------------------------------------------------------------------------------#
#--------------------------------------Ridge Regression-------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Try a bunch of lambda (shrinkage) values between 0.01 and 1,000
lambdas <- 10^seq(3, -2, by=-0.1)
lambdas
#Run the ridge regression using the lambdas
ridge_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=0, lambda=lambdas)
summary(ridge_reg)
#Tune lambda using cross validation
ridge_reg <- cv.glmnet(as.matrix(d[,-1]), d$mpg, alpha=0, lambda=lambdas)
plot(ridge_reg)
#Minimum of the MSE is where the best lambda choice is
best_lambda <- ridge_reg$lambda.min
#Best ridge regression
ridge_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=0, lambda=best_lambda)
coef(ridge_reg)

#-------------------------------------------------------------------------------------------------#
#-------------------------------------LASSO Regression--------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Try a bunch of lambda (shrinkage) values between 0.01 and 1,000
lambdas <- 10^seq(3, -2, by=-0.1)
lambdas
#Run the ridge regression using the lambdas
lasso_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=1, lambda=lambdas)
summary(lasso_reg)
#Tune lambda using cross validation
lasso_reg <- cv.glmnet(as.matrix(d[,-1]), d$mpg, alpha=1, lambda=lambdas)
plot(lasso_reg)
#Minimum of the MSE is where the best lambda choice is
best_lambda <- lasso_reg$lambda.min
#Best ridge regression
lasso_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=1, lambda=best_lambda)
coef(lasso_reg)

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Elastic Net-----------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Try a bunch of lambda (shrinkage) values between 0.01 and 1,000
lambdas <- 10^seq(3, -2, by=-0.1)
lambdas
#Run the ridge regression using the lambdas
en_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=0.5, lambda=lambdas)
summary(en_reg)
#Tune lambda using cross validation
en_reg <- cv.glmnet(as.matrix(d[,-1]), d$mpg, alpha=0.5, lambda=lambdas)
plot(en_reg)
#Minimum of the MSE is where the best lambda choice is
best_lambda <- en_reg$lambda.min
#Best ridge regression
en_reg <- glmnet(as.matrix(d[,-1]), d$mpg, alpha=0.5, lambda=best_lambda)
coef(en_reg)

#Get predictions for each model
d$ridge_preds <- predict(ridge_reg, s=best_lambda, newx=as.matrix(d[,-1]))
plot(d$ridge_preds, d$mpg, main='Ridge Regression Predicted vs Actual Values')
d$lasso_preds <- predict(lasso_reg, s=best_lambda, newx=as.matrix(d[,-c(1,12)]))
plot(d$lasso_preds, d$mpg, main='Lasso Predicted vs Actual Values')
d$en_preds <- predict(en_reg, s=best_lambda, newx=as.matrix(d[,-c(1,12,13)]))
plot(d$en_preds, d$mpg, main='Elastic Net Predicted vs Actual Values')

#Compare models
print(c('Ridge MSE:', mean((d$mpg-d$ridge_preds)^2)))
print(c('Lasso MSE:', mean((d$mpg-d$lasso_preds)^2)))
print(c('Elastic Net MSE:', mean((d$mpg-d$en_preds)^2)))

#Note that glmnet penalizes MSE instead of least squares
#The lines below transform the glmnet coefficients in terms of penalized least squares
#Thanks to stack overflow for this:  https://stackoverflow.com/questions/39863367/ridge-regression-with-glmnet-gives-different-coefficients-than-what-i-compute
ridge_reg_coef <- solve(t(as.matrix(d[,-c(1,12,13,14)])) %*% as.matrix(d[,-c(1,12,13,14)]) + ridge_reg$lambda * diag(ncol(d)-4)) %*% t(as.matrix(d[,-c(1,12,13,14)])) %*% d$mpg
lasso_reg_coef <- solve(t(as.matrix(d[,-c(1,12,13,14)])) %*% as.matrix(d[,-c(1,12,13,14)]) + lasso_reg$lambda * diag(ncol(d)-4)) %*% t(as.matrix(d[,-c(1,12,13,14)])) %*% d$mpg
en_reg_coef <- solve(t(as.matrix(d[,-c(1,12,13,14)])) %*% as.matrix(d[,-c(1,12,13,14)]) + en_reg$lambda * diag(ncol(d)-4)) %*% t(as.matrix(d[,-c(1,12,13,14)])) %*% d$mpg
#Zero out the least squares penalized coefficients for the features that were removed by the models due to penalization
ridge_reg_coef * ifelse(coef(ridge_reg)[-1]==0, 0, 1)
lasso_reg_coef * ifelse(coef(lasso_reg)[-1]==0, 0, 1)
en_reg_coef * ifelse(coef(en_reg)[-1]==0, 0, 1)
