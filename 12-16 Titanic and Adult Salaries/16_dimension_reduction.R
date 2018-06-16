#R code for chapter 16 of DSILT: Statistics

setwd("/home/dsilt/Desktop/dsilt-stats-code/12-16 Titanic and Adult Salaries")

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

#Since there are 32k rows and <2k rows with nulls, it is safe to discard them
d <- d[(d$workclass!=' ?' & d$occupation!=' ?' & d$native_country!=' ?'),]
d$workclass <- factor(d$workclass)
d$occupation <- factor(d$occupation)
d$native_country <- factor(d$native_country)

d[,which(sapply(d, class)=='factor')] <- sapply(d[,which(sapply(d, class)=='factor')], substring, 2, 100)  #Gets rid of leading whitespaces that show up in factor levels
d[,which(sapply(d, class)=='character')] <- lapply(d[,which(sapply(d, class)=='character')], as.factor)

#Save the version of the dataset without dummies
d_no_dummies <- d

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
train_no_dummies <- d_no_dummies[train_indices,]
test_no_dummies <- d_no_dummies[-train_indices,]
train_numeric <- train_no_dummies[,which(sapply(train_no_dummies, class)=='numeric')]
test_numeric <- test_no_dummies[,which(sapply(test_no_dummies, class)=='numeric')]

#-------------------------------------------------------------------------------------------------#
#------------------------------------PCA on Numeric Data------------------------------------------#
#-------------------------------------------------------------------------------------------------#

######
#Run a couple tests to make sure PCA can actually be carried out

#Barlett's test to see if a correlation matrix is an identity matrix
corm <- cor(train_numeric)
library(psych)
cortest.bartlett(corm, n=nrow(train_numeric))
#Low p-value indicates that the correlation matrix is factorable so dimension reduction can be done

#Create function to find KMO statistic
kmo = function( data ){
  library(MASS) 
  X <- cor(as.matrix(data)) 
  iX <- ginv(X) 
  S2 <- diag(diag((iX^-1)))
  AIS <- S2%*%iX%*%S2                      # anti-image covariance matrix
  IS <- X+AIS-2*S2                         # image covariance matrix
  Dai <- sqrt(diag(diag(AIS)))
  IR <- ginv(Dai)%*%IS%*%ginv(Dai)         # image correlation matrix
  AIR <- ginv(Dai)%*%AIS%*%ginv(Dai)       # anti-image correlation matrix
  a <- apply((AIR - diag(diag(AIR)))^2, 2, sum)
  AA <- sum(a) 
  b <- apply((X - diag(nrow(X)))^2, 2, sum)
  BB <- sum(b)
  MSA <- b/(b+a)                        # indiv. measures of sampling adequacy
  AIR <- AIR-diag(nrow(AIR))+diag(MSA)  # Examine the anti-image of the correlation matrix. That is the  negative of the partial correlations, partialling out all other variables.
  kmo <- BB/(AA+BB)                     # overall KMO statistic
  # Reporting the conclusion 
  if (kmo >= 0.00 && kmo < 0.50){test <- 'The KMO test yields a degree of common variance unacceptable for FA.'} 
  else if (kmo >= 0.50 && kmo < 0.60){test <- 'The KMO test yields a poor degree of common variance of >= 0.5 to < 0.6.'} 
  else if (kmo >= 0.60 && kmo < 0.70){test <- 'The KMO test yields a cautionary degree of common variance of >= 0.6 to < 0.7'} 
  else if (kmo >= 0.70 && kmo < 0.80){test <- 'The KMO test yields a satisfactory degree of common variance of >= 0.7 to < 0.8' } 
  else if (kmo >= 0.80 && kmo < 0.90){test <- 'The KMO test yields a high degree of common variance of >= 0.8 to < 0.9' }
  else { test <- 'The KMO test yields a superbly high degree of common variance.' }
  
  ans <- list( overall = kmo,
               report = test,
               individual = MSA,
               AIS = AIS,
               AIR = AIR )
  return(ans)
}

#Perform the KMO test to get the KMO statistic
kmo(corm)$overall
kmo(corm)$report

#Find the determinant of the correlation matrix to see if variable elimination would be a good idea
det(corm) > 0.00001
#Determinant is sufficiently high so there is no need to eliminate variables

######
#Perform PCA

#Starting with all variables included
pc1 <- principal(corm, nfactors=length(corm[,1]), rotate="none")
pc1

#All 5 eigenvalues are > 0.7
#Draw a scree plot to confirm and decide how many variables to extract
plot(pc1$values, type="b")
#Slope kind of levels off at the 4th variable, so extracting 3

#PCA with 3 variables
pc2 <- principal(corm, nfactors=3, rotate="none")
pc2

#Validate PCA with reproduced correlation matrix
factor.model(pc2$loadings)
factor.residuals(corm, pc2$loadings)
#Check how many large residuals there are
residuals <- factor.residuals(corm, pc2$loadings)
residuals <- as.matrix(residuals[upper.tri(residuals)])
sum(abs(residuals) > 0.05) / nrow(residuals)
sqrt(mean(residuals^2))
#Percent of large residuals and the MSE are very high, so adding another factor or two might help reduce the error

#PCA with 4 variables
pc3 <- principal(corm, nfactors=4, rotate="none")
pc3

#Check how many large residuals there are
residuals <- factor.residuals(corm, pc3$loadings)
residuals <- as.matrix(residuals[upper.tri(residuals)])
sum(abs(residuals) > 0.05) / nrow(residuals)
sqrt(mean(residuals^2))
#Better, but still high - this really supports original analysis that PCA is not needed

#View the sorted factor loadings for each variable
print.psych(pc3, sort=TRUE)

#Using oblique rotation to examine the factor loadings for each variable
pc4 <- principal(corm, nfactors=4, rotate="oblimin")
print.psych(pc4, sort=TRUE)
#Remove loadings below 0.5 to more clearly see the variables in their factor groups
print.psych(pc4, cut=0.5, sort=TRUE)

#Compute PC scores for new data - use predict.psych from the psych package
test_numeric_pcs <- predict.psych(pc4, test_numeric, train_numeric)

#-------------------------------------------------------------------------------------------------#
#-------------------------------------------FAMD--------------------------------------------------#
#-------------------------------------------------------------------------------------------------#

#FAMD, starting with all variables extracted
library(FactoMineR)
famd1 <- FAMD(train_no_dummies[,-15], ncp=ncol(train_no_dummies[,-15]), graph=FALSE)
famd1

#View the eignvalues of the latent features (the loadings)
library(factoextra)
famd1_eigenvalues <- get_eigenvalue(famd1)
famd1_eigenvalues

#Scree plot
fviz_screeplot(famd1)

#FAMD with 4 extracted features
famd2 <- FAMD(train_no_dummies[,-15], ncp=4, graph=FALSE)

#View the combinations of the original features that produced the latent features
var <- get_famd_var(famd2)
var
var$contrib

#Plot the features in the coordinate plane of the first two latent features
fviz_famd_var(famd2, repel=TRUE)

#Compute FAMD scores for new data - use predict.FAMD from the FactoMineR package
test_no_dummies_lvs <- predict.FAMD(famd2, test_no_dummies[,-15])

#Plot the FAMD on the first 2 components and save results to png
png(filename="FAMD_Plot_First_2_Components.png")
plot.FAMD(famd2, axes=c(1,2), choix="ind", habillage=2)
dev.off()

######
#Use LDA and QDA for classification on the reduced dimensions from FAMD

#coord contains the scores for each dimension
train_famd2_res <- famd2$ind
train_reduced <- cbind(salary_bin=train_no_dummies[,15], as.data.frame(train_famd2_res$coord))
test_reduced <- cbind(salary_bin=test_no_dummies[,15], as.data.frame(test_no_dummies_lvs$coord))
colnames(test_reduced) <- gsub(" ", ".", colnames(test_reduced))

library(MASS)
lda_model <- lda(formula=salary_bin ~ ., data=train_reduced)
lda_model

#Make predictions for test set
post_lda <- predict(lda_model, test_reduced[,-1])
summary(post_lda)
test_reduced$pred <- post_lda$class
test_reduced$baseline <- '<=50K'  #Baseline assigns everything to majority class
test_reduced$baseline <- factor(test_reduced$baseline, levels=c('<=50K', '>50K'))

#Evaluate the model using the log loss, AUC, and accuracy
library(MLmetrics)
LogLoss(as.integer(test_reduced$pred), as.integer(test_reduced$salary_bin))
AUC(as.integer(test_reduced$pred), as.integer(test_reduced$salary_bin))
Accuracy(as.factor(test_reduced$pred), test_reduced$salary_bin)
Accuracy(as.factor(test_reduced$baseline), test_reduced$salary_bin)
#View the confusion matrix with predicted values on the left
prop.table(table(test_reduced$pred, test_reduced$salary_bin))

qda_model <- qda(formula=salary_bin ~ ., data=train_reduced)
qda_model

#Make predictions for test set
post_qda <- predict(qda_model, test_reduced[,-1])
summary(post_qda)
test_reduced$qda_pred <- post_qda$class

#Evaluate the model using the log loss, AUC, and accuracy
LogLoss(as.integer(test_reduced$qda_pred), as.integer(test_reduced$salary_bin))
AUC(as.integer(test_reduced$qda_pred), as.integer(test_reduced$salary_bin))
Accuracy(as.factor(test_reduced$qda_pred), test_reduced$salary_bin)
Accuracy(as.factor(test_reduced$baseline), test_reduced$salary_bin)
#View the confusion matrix with predicted values on the left
prop.table(table(test_reduced$qda_pred, test_reduced$salary_bin))

#-------------------------------------------------------------------------------------------------#
#---------------------------------LDA for Dimension Reduction-------------------------------------#
#-------------------------------------------------------------------------------------------------#

#Run LDA for dimension reduction, ignoring the multicollinearity
lda_dim_reduced <- lda(formula=salary_bin ~ ., data=train_no_dummies)

#Explore LDA results from the model that was just built
lda_dim_reduced$prior    #Class prior probabilities
lda_dim_reduced$means    #Class specific means for each predictor
lda_dim_reduced$scaling  #The factor loadings of the predictors on the latent variables (linear discriminants)
lda_dim_reduced$svd      #Singular values: the ratio of the between group to the within group standard deviations of the linear discriminants
perc_var_explained <- lda_dim_reduced$svd^2/sum(lda_dim_reduced$svd^2)
perc_var_explained
#Plot the distribution of each class over the linear discriminant (only 1 LD was computed)
plot(lda_dim_reduced)
#Plot the features by their loadings on the linear discriminant
plot(lda_dim_reduced$scaling, xlab='Feature')

#Calculate the values of each observation using the linear discriminant
train_lda_reduced <- as.data.frame(predict(lda_dim_reduced, train_no_dummies))
head(train_lda_reduced$LD1)  #The observation values of the linear discriminant

