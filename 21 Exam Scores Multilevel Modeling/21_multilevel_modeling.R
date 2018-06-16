#R code for chapter 21 of DSILT: Statistics

setwd("/home/dsilt/Desktop/dsilt-stats-code/21 Exam Scores Multilevel Modeling")

library(lme4)

#Data has exam scores (both the total and the score for an essay on the exam), for 1905 students from 73 schools in England
#Data from http://www.bristol.ac.uk/cmm/learning/support/datasets/
d <- read.table('exam_scores.DAT', header=F, sep='')
colnames(d) <- c('school', 'student', 'female_flag', 'essay_score', 'exam_score')
d <- d[d$school!='\032',]  #Gets rid of junk row
d$school <- as.integer(d$school)
head(d)

#Linear regression of exam_score on school, female_flag, and essay_score
linear_reg <- lm(exam_score ~ school + female_flag + essay_score, data=d)
summary(linear_reg)
preds <- predict(linear_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #15.1
#Reg is signif but poor fit (R^2=0.27)

#Linear regression of exam_score on school dummies, female_flag, and essay_score
d$school <- as.factor(d$school)
dummies <- model.matrix(~d$school)[,-1]
colnames(dummies) <- substr(colnames(dummies), 3, 100)
d <- cbind(d[,-1], dummies)
d <- d[,-1]  #Drop student
linear_reg <- lm(exam_score ~ ., data=d)
summary(linear_reg)
preds <- predict(linear_reg, d[,-3])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #12.5
#Reg is signif but still poor fit (R^2=0.5) and now there are so many dummies it is hard to look through them all, and cannot easily test for interaction between school and female_flag

#Many linear regressions (1 for each school) of exam_score on female_flag, and essay_score
d <- read.table('exam_scores.DAT', header=F, sep='')  #Read data back in to reset everything
colnames(d) <- c('school', 'student', 'female_flag', 'essay_score', 'exam_score')
d <- d[d$school!='\032',]  #Gets rid of junk row
d$school <- as.integer(d$school)
schools <- unique(d$school)
linear_regs <- by(data=d, INDICES=d$school, FUN=function(x) {lm(exam_score ~ school + female_flag + essay_score, data=d)})

#Use multilevel model to account for exam_score differences between schools and within schools
#start with varying intercept model - allows intercept to vary by the group variable school
ml_reg <- lmer(exam_score ~ (1 | school) + female_flag + essay_score, data=d)
summary(ml_reg)
preds <- predict(ml_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #12.5

#Create artificial grouping variable using k-means
kclusters <- kmeans(as.matrix(d[,c('female_flag', 'essay_score')]), centers=5)
library(fpc)
plotcluster(d[,c('school', 'essay_score')], kclusters$cluster)
d$cluster <- kclusters$cluster  #Assumes the row ordering is the same

#Varying intercept model with mixed effect term for varying intercepts by cluster and schools within clusters
ml_clust_reg <- lmer(exam_score ~ (1 | cluster/school) + female_flag + essay_score, data=d)
summary(ml_clust_reg)
preds <- predict(ml_clust_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #11.6
#Alternatively, do a varying intercept model without nesting
ml_clust_reg <- lmer(exam_score ~ (1 | cluster) + (1 | school) + female_flag + essay_score, data=d)
summary(ml_clust_reg)
preds <- predict(ml_clust_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #12.4

#Varying slope model to explore the effect of essay_score (a student level variable) as it varies across clusters and schools
ml_clust_reg <- lmer(exam_score ~ (1 + essay_score | cluster) (1 + essay_score | school) + female_flag, data=d)
summary(ml_clust_reg)
preds <- predict(ml_clust_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #12.4
#Same thing but with mixed effect term for varying intercepts by cluster and schools within clusters, and varying slopes by essay_score by cluster and schools within clusters
ml_clust_reg <- lmer(exam_score ~ (1 + essay_score | cluster/school) + female_flag, data=d)  #Fails to converge
summary(ml_clust_reg)
preds <- predict(ml_clust_reg, d[,-c(2,5)])
rmse <- sqrt(mean((preds-d$exam_score)^2))
rmse  #11.6

#Plot error by observation to see where the model messed up
library(ggplot2)
abs_dev <- abs(preds-d$exam_score)
d$abs_dev <- abs_dev
ggplot(d, aes(x=row.names(d), y=exam_score)) + geom_point(aes(colour=abs_dev)) + scale_colour_gradient(low='white', high='black') + ggtitle('Absolute Error by Observation')
