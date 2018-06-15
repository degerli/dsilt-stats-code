#R code for chapter 25 of DSILT: Statistics

setwd("C:/Users/Nick/Documents/Word Documents/Data Science Books/DSILT Stats Code/25 Airline Passengers and Stock Price Time Series")

library(tseries)
library(forecast)

d <- read.csv('AirPassengers.csv', header=T)
d$X <- NULL
plot(d$time, d$value, type='l')

#-------------------------------------------------------------------------------------------------#
#---------------------------------Time Series Decomposition---------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Look at ACF and PACF, and perform Augmented Dickey-Fuller test for stationarity
acf(d$value)
pacf(d$value)
adf.test(d$value)  #Stationary

#Note that as the time series increases in magnitude, seasonal variation increases too.  This is a multiplicative relationship (trend*season*random)
#If the above statement were not true, it would be an additive relationship (trend+season+random)

#Smooth by using a centered moving average (data is monthly, so 12 month trend seems appropriate)
trend <- ma(d$value, order=12, centre=T)
plot(d$time, d$value, type='l')
lines(d$time, trend, col='red')

#Remove the trend: remove multiplicative trends by dividing and additive trends by subtracting
d_detrend <- d$value/trend
plot(d$time, d_detrend, type='l')

#Average the seasonality over the period of the MA trend (it was 12 for this dataset)
mseas <- t(matrix(d_detrend, nrow=12))  #Create a matrix of the time periods, where columns are the periods
season <- colMeans(mseas, na.rm=T)      #Column means are the averaged seasonality for each period
plot(as.ts(rep(season, 12)))

#Remove the seasonality: remove multiplicative seasonality by dividing and additive seasonality by subtracting
d_random <- d$value / (trend*season)
plot(as.ts(d_random))

#Do the same thing using the decompose function (requires a time series object) - this is classical decompostion with MAs
?decompose
dts <- ts(d$value, frequency=12)  #Use 12 as frequency because data is measured monthly so a 12 month period seems reasonable
d_decomp <- decompose(dts, "multiplicative")
d_decomp
plot(d_decomp)

#Do the same thing using LOESS decomposition
?stlm
dts <- ts(d$value, frequency=12)
d_stl_decomp <- stlm(dts, "periodic", allow.multiplicative.trend=T)
d_stl_decomp
d_stl_decomp_ts <- as.data.frame(d_stl_decomp$stl)
par(mfrow=c(4,1))
plot(d_stl_decomp_ts$Data, type='l', ylab='observed')
plot(d_stl_decomp_ts$Trend, type='l', ylab='trend')
plot(d_stl_decomp_ts$Seasonal12, type='l', ylab='seasonal')
plot(d_stl_decomp_ts$Remainder, type='l', ylab='random')
par(mfrow=c(1,1))

#Test for stationarity after seasonality and trend have been removed
adf.test(d_stl_decomp_ts$Remainder)  #Stationary
#Perform first differencing to see if that makes series stationary (don't need to here, but it's good for example)
adf.test(diff(d_stl_decomp_ts$Remainder, differences=1))

#-------------------------------------------------------------------------------------------------#
#------------------------------------------ARIMA--------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Find best ARIMA model, using BIC, up to maximum orders (max.p, max.d, max.q for ARIMA(p,d,q) models)
arima_model <- auto.arima(d_stl_decomp_ts$Remainder, max.p=5, max.d=5, max.q=5, ic="bic")
plot(arima_model)
tsdisplay(residuals(arima_model), main="ARIMA(2,0,1) Residuals")
Box.test(residuals(arima_model), type='Ljung-Box')  #No evidence of autocorrelation in the residuals
#Check residuals for normality
qqnorm(residuals(arima_model))
qqline(residuals(arima_model))

#Make future predictions for next 24 months
preds <- forecast(arima_model, h=24)
plot(preds)

#Do the same thing using a holdout test set to determine model accuracy
train <- ts(d[1:(144-24), 'value'], frequency=12)
test <- ts(d[(144-24):144, 'value'], frequency=12)
train_stl_decomp <- stlm(train, 'periodic', allow.multiplicative.trend=T)
test_stl_decomp <- stlm(test, 'periodic', allow.multiplicative.trend=T)
train_stl_decomp_ts <- as.data.frame(train_stl_decomp$stl)
test_stl_decomp_ts <- as.data.frame(test_stl_decomp$stl)
arima_model_train <- auto.arima(train_stl_decomp_ts$Remainder, ic='bic')
#arima_model_train <- arima(train_stl_decomp_ts$Remainder, order=c(2,0,1))
preds_test <- forecast(arima_model_train, h=24)
plot(preds_test, main="Forecasted (blue) vs Actual (red) for Last 24 Months")
lines(seq(120,144), test_stl_decomp_ts$Remainder, col='red')

#Holt-Winters exponential smoothing, similar to MA or LOESS smoothing, assumes trend & seasonality are additive
hw_model <- HoltWinters(dts)
hw_model
plot(hw_model)
tsdisplay(residuals(hw_model), main="Holt-Winters Residuals")
Box.test(residuals(hw_model), type='Ljung-Box')  #There IS evidence of autocorrelation in the residuals
#Check residuals for normality
qqnorm(residuals(hw_model))
qqline(residuals(hw_model))
#Plot forecasts
preds_hw <- forecast(hw_model, h=24)
plot(preds_hw)

#ARIMA model the original series to avoid having to add trend and seasonality back to make forecasts
arima_model_two <- auto.arima(dts, ic='bic')
preds_orig_scale <- forecast(arima_model_two, h=24)
plot(preds_orig_scale)

# #Explore using Fourier transform to determine seasonality
# library(TSA)  #Note that this overrides acf and arima from stats package
# 
# #View the periodogram of the Fourier transform of a series
# p = periodogram(as.ts(d$value))
# 
# #Get a dataframe of the frequencies and their power
# freqdf <- data.frame(freq=p$freq, spec=p$spec)
# freqdf$freq <- 1/freqdf$freq  #Convert frequency to time periods (since data is measured in months, this will be months)
# freqdf <- freqdf[order(-freqdf$spec),]  #Order strongest to weakest
# head(freqdf)
# #Biggest seasonality is 144 months (12 years), which is the entire length of the data -> there is a long term trend
# #Seasonality every 6 years and every 1 year

#-------------------------------------------------------------------------------------------------#
#----------------------------------------ARIMA + GARCH--------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#ARIMA + GARCH
library(quantmod)
library(timeSeries)
library(tseries)
library(forecast)
library(rugarch)

#Created differenced log returns of daily closing prices (Cl) for the S&P500 back to 2000
getSymbols("^GSPC", from="2000-01-01")
spret <- diff(log(Cl(GSPC)))
spret[as.character(head(index(Cl(GSPC)),1))] <- 0

#Fit ARIMA
arima_model <- auto.arima(spret, ic='bic')
arima_model
preds <- forecast(arima_model, h=60)  #Forecast next 60 days
plot(preds)

#Fit ARIMA+GARCH
garch_model <- ugarchspec(mean.model=list(armaOrder=c(2, 0), include.mean=T), 
                          variance.model=list(garchOrder=c(1,1)), 
                          distribution.model='sged')
#Fit the model to all but the last 100 observations
garchfit <- ugarchfit(spec=garch_model, data=spret, solver='hybrid', out.sample=100)
garchfit@fit$coef  #Model coefficients
print(garchfit)

#Plot observed log returns vs predicted log returns - notice that the predicted returns show what happends when volatility clustering is removed
plot(spret, main='Actual vs Fitted (red) Values')
lines(fitted(garchfit), col='red')

#Plot the volatility by itself
plot(garchfit@fit$sigma, type='l', ylab='Volatility')

#Plot the volatility with the observed log returns
par(mfrow=c(2,1))
plot(spret, type='l', main='S&P500 Daily Log Returns')
plot(garchfit, which=3)
par(mfrow=c(1,1))

#Plot forecasts
garch_preds <- ugarchforecast(garchfit, n.ahead=100, n.roll=100, out.sample=100)
plot(garch_preds)

#-------------------------------------------------------------------------------------------------#
#---------------------------Cointegration and Pairs Trading---------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Cointegration != correlation
#Start with 2 that you're fairly certain will be cointegrated
getSymbols("GOOGL", from="2012-05-01", to="2018-05-01")
getSymbols("GOOG", from="2012-05-01", to="2018-05-01")
googlp <- unclass(GOOGL$GOOGL.Adjusted)
googp <- unclass(GOOG$GOOG.Adjusted)
plot(googlp, type='l', col='purple')
lines(googp, type='l', col='orange')
plot(googlp, googp)  #This should be generally linear

#Test for cointegration by regressin each on the other
googlm <- lm(googlp~googp)
googm <- lm(googp~googlp)
adf.test(googlm$residuals, k=1)  #Cannot reject null for stationary series -> nonstationary so cannot use cointegration
adf.test(googm$residuals, k=1)   #Cannot reject null for stationary series -> nonstationary so cannot use cointegration

#To try again for 2 other stocks, let's make a function
coint_test <- function(stock1, stock2) {
  stock1m <- lm(stock1~stock2)
  stock2m <- lm(stock2~stock1)
  adf1 <- adf.test(stock1m$residuals, k=1)
  adf2 <- adf.test(stock2m$residuals, k=1)
  if (adf1$p.value>=0.05 & adf2$p.value>=0.05) {
    return(print('Non-stationary, cannot do cointegration'))
  } else if (adf1$statistic < adf2$statistic) {
    return(print(paste0('Use ', colnames(stock1), ' as the dependent variable in the linear combination.')))
  } else {
    return(print(paste0('Use ', colnames(stock2), ' as the dependent variable in the linear combination.')))
  }
}

#Let's cheat and pick 2 that are known to be cointegrated to see what happens
#We'll also cherry pick a time frame during which these stocks were stationary
getSymbols("EWA", from="2008-02-01", to="2012-02-01")
getSymbols("EWC", from="2008-02-01", to="2012-02-01")
ewap <- unclass(EWA$EWA.Adjusted)
ewcp <- unclass(EWC$EWC.Adjusted)
par(mfrow=c(2,1))
plot(ewap, type='l', col='purple')
plot(ewcp, type='l', col='orange')
par(mfrow=c(1,1))
plot(ewap, ewcp)  #This should be generally linear
coint_test(ewap, ewcp)
ewa_hedge_model <- lm(ewap~ewcp)
summary(ewa_hedge_model)
#Hedge ratio is 0.71

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Benford's Law---------------------------------------------#
#-------------------------------------------------------------------------------------------------#

library(benfordsLaw)

census <-  read.csv("Census_Town&CityPopulations.csv", header=TRUE)
#Isolate town population and remove commas and NAs
x <- as.numeric(sub( ",", "", census$X7_2009))
x <- x[!is.na(x)]

first_digit <- benfordFirstDigit(x)
first_two_digits <- benfordFirstTwoDigit(x)
last_two_digits <- benfordLastTwoDigit(x)
plotFD(first_digit)
plotF2D(first_two_digits)
plotL2D(last_two_digits)

#-------------------------------------------------------------------------------------------------#
#---------------------------------Change Point Detection------------------------------------------#
#-------------------------------------------------------------------------------------------------#

library(ecp)

getSymbols("AAPL", from="2008-02-01", to="2012-02-01")
AAPL <- Cl(AAPL)

#AAPL test
aapl_ecp <- e.divisive(X=AAPL, min.size=120)
plot(AAPL)
abline(v=.index(AAPL)[aapl_ecp$estimates[-c(1, length(aapl_ecp$estimates))]], col='blue', lwd=2)

#Get median and mean closing prices for each segment (area between change points)
aapl_segments <- AAPL
aapl_segments$AAPL.Segment <- aapl_ecp$cluster
aapl_seg_meds <- aggregate(aapl_segments, by=list(aapl_segments$AAPL.Segment), FUN=median)
aapl_seg_means <- aggregate(aapl_segments, by=list(aapl_segments$AAPL.Segment), FUN=mean)
#Find differences between segments as percent change from prior segment
diff(aapl_seg_meds)/aapl_seg_meds*100
diff(aapl_seg_means)/aapl_seg_means*100
