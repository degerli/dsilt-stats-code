#R code for chapter 20 of DSILT: Statistics

d <- mtcars
str(d)

#-------------------------------------------------------------------------------------------------#
#----------------------------------Polynomial Regression------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Polynomial regression of orders 3, 5, 10 with 1 predictor
poly_reg3 <- lm(mpg ~ disp + poly(disp, 3, raw=T), data=d)
poly_reg5 <- lm(mpg ~ disp + poly(disp, 5, raw=T), data=d)
poly_reg10 <- lm(mpg ~ disp + poly(disp, 10, raw=T), data=d)

summary(poly_reg3)
summary(poly_reg5)
summary(poly_reg10)

plot(poly_reg3)
plot(poly_reg5)
plot(poly_reg10)

#Plot the fitted lines
x_order <- order(d$disp)
poly_reg3_preds <- predict(poly_reg3)
poly_reg5_preds <- predict(poly_reg5)
poly_reg10_preds <- predict(poly_reg10)
plot(d$disp, d$mpg, main='Polynomial Regression of Orders 3, 5, and 10', xlab='disp', ylab='mpg')
lines(d$disp[x_order], poly_reg3_preds[x_order], col='orange', lwd=3)
lines(d$disp[x_order], poly_reg5_preds[x_order], col='green', lwd=3)
lines(d$disp[x_order], poly_reg10_preds[x_order], col='purple', lwd=3)

#Polynomial regression of order 3 with many predictors
poly_reg3 <- lm(mpg ~ disp + poly(disp, 3, raw=T) + hp + poly(hp, 3, raw=T) + drat + wt, data=d)
summary(poly_reg3)
plot(poly_reg3)

#-------------------------------------------------------------------------------------------------#
#------------------------------------Isotonic Regression------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Simulate a function of mpg that is increasing
dsim <- data.frame(mpg=d$mpg+rnorm(1), disp=d$disp*(-1))

#View scatter plot of x and mpg
plot(dsim$disp, dsim$mpg)

#Fit isotonic regression
iso_reg <- isoreg(dsim$disp, dsim$mpg)
iso_reg
iso_reg$yf  #Fitted values
plot(iso_reg, xlab='-1*disp', ylab='mpg')

#-------------------------------------------------------------------------------------------------#
#------------------------------------Smoothing Functions------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Smoothing with moving average of a 6 value window (the observation and 6 values around it)
library(zoo)
smooth_ma <- rollmean(dsim$mpg, k=6, align='center', na.pad=T)
smooth_ma
plot(row.names(dsim), dsim$mpg, main='Moving Average Smoothing', xlab='Row Index', ylab='mpg')
lines(row.names(dsim), smooth_ma, col='darkgreen', lwd=3)

#Smoothing with cubic spline
library(splines)
cubic_spline <- lm(mpg ~ bs(disp, knots=c(min(dsim$disp), mean(dsim$disp), max(dsim$disp))), data=dsim)
cubic_spline
newd <- data.frame(disp=dsim$disp)
cubic_spline_vals <- predict(cubic_spline, newdata=newd)
x_order <- order(dsim$disp)  #Store the order for the x-axis
plot(dsim$disp, dsim$mpg, main='Cubic Spline Smoothing', xlab='-1*disp', ylab='mpg')
lines(dsim$disp[x_order], cubic_spline_vals[x_order], col='red', lwd=3)

#Smoothing with LOESS
smooth_loess <- loess(mpg ~ disp, data=dsim, span=0.5)  #smoothing span of 50% of the data
smooth_loess
smoothed_points <- predict(smooth_loess)
x_order <- order(dsim$disp)  #Store the order for the x-axis
#plot(dsim$disp, dsim$mpg, main='LOESS Smoothing', xlab='-1*disp', ylab='mpg')
lines(dsim$disp[x_order], smoothed_points[x_order], col='blue', lwd=3)

#-------------------------------------------------------------------------------------------------#
#-------------------------------Generalized Additive Models---------------------------------------#
#-------------------------------------------------------------------------------------------------#

#GAMs
library(gam)
gam_model <- gam(mpg ~ s(disp, df=6) + s(wt, df=6), data=d)  #Note s = smoothing spline
summary(gam_model)
plot(gam_model, se=T, main="GAM Smoothing for Variable")

#Adding a binary variable to the GAM
gam_model_b <- gam(mpg ~ s(disp, df=6) + s(wt, df=6) + vs, data=d)
summary(gam_model_b)
plot(gam_model_b, se=T, main="GAM Smoothing for Variable")

#Adding a categorical variable to the GAM
d$hp_bin <- cut(d$hp, 5)
table(d$hp_bin)
gam_model_b_c <- gam(mpg ~ s(disp, df=6) + s(wt, df=6) + vs + hp_bin, data=d)
summary(gam_model_b_c)
plot(gam_model_b_c, se=T, main="GAM Smoothing for Variable")

#Compare models
anova(gam_model, gam_model_b, gam_model_b_c)

#Performing classification (logistic regression) with the GAM
gam_model_class <- gam(I(mpg < 20) ~ s(disp, df=6) + s(wt, df=6) + vs + hp_bin, data=d, family=binomial)
summary(gam_model_class)
plot(gam_model_class, se=T, main="GAM Smoothing for Variable")
#This models the conditional probabilities for mpg being < 20 and >=20
#Note the y-axis of these plots is the logit

#-------------------------------------------------------------------------------------------------#
#----------------------------Regression and Classification Trees----------------------------------#
#-------------------------------------------------------------------------------------------------#

#Regression tree - use rpart method='anova'
library(rpart)
reg_tree <- rpart(mpg ~ ., data=mtcars, method='anova', 
                  control=rpart.control(minsplit=20, cp=0.001))
summary(reg_tree)
rsq.rpart(reg_tree)
plot(reg_tree)
text(reg_tree)

#Classification tree - use rpart method='class'
library(rpart)
class_tree <- rpart(Species ~ ., data=iris, method='class', 
                    control=rpart.control(minsplit=20, cp=0.001))
summary(class_tree)
plot(class_tree)
text(class_tree)

#-------------------------------------------------------------------------------------------------#
#-------------------------------------K-Nearest Neighbors-----------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Split data into training and test sets
library(caret)
set.seed(14)
train_indices <- createDataPartition(y=mtcars$mpg, p=0.7, list=F)
train <- mtcars[train_indices,]
test <- mtcars[-train_indices,]

#Normalize numeric variables
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
train_norm <- as.data.frame(sapply(train, normalize), row.names=row.names(train))
test_norm <- as.data.frame(sapply(test, normalize), row.names=row.names(test))

#KNN Regression, k=3
library(FNN)
knn_reg <- knn.reg(train=train_norm, test=test_norm, y=train_norm$mpg, k=3, algorithm='kd_tree')
knn_reg
plot(mtcars$mpg, knn_reg$pred, xlab="actual mpg", ylab="predicted mpg")
print(c("RMSE:", sqrt(mean(knn_reg$residuals^2))))

#Split data into training and test sets
library(caret)
set.seed(14)
train_indices <- createDataPartition(y=iris$Species, p=0.7, list=F)
train <- iris[train_indices,]
test <- iris[-train_indices,]

#Normalize numeric variables
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
train_norm <- as.data.frame(sapply(train[,-which(colnames(train) %in% c('Species'))], normalize), row.names=row.names(train))
test_norm <- as.data.frame(sapply(test[,-which(colnames(train) %in% c('Species'))], normalize), row.names=row.names(test))
#Label encode target
train_norm$Species <- as.integer(train$Species)
test_norm$Species <- as.integer(test$Species)

#KNN Classification, k=3
library(class)
knn_classifier <- class::knn(train=train_norm, test=test_norm, cl=train$Species, k=3, prob=F)
knn_preds <- as.integer(knn_classifier)
library(MLmetrics)
LogLoss(knn_preds, test_norm$Species)
Accuracy(knn_preds, test_norm$Species)
#View the confusion matrix with predicted values on the left
prop.table(table(knn_preds, test_norm$Species))
