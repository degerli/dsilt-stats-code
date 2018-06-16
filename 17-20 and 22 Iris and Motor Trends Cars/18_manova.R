#R code for chapter 18 of DSILT: Statistics

d <- iris
str(d)

library(mvnormtest)
#Note that the matrix must be transposed so observations become columns
mshapiro.test(t(as.matrix(d[,c('Sepal.Length', 'Sepal.Width')])))

library(car)
leveneTest(d$Sepal.Length ~ d$Species)
leveneTest(d$Sepal.Width ~ d$Species)

#Ratio of largest to smallest variance for sepal length
library(dplyr)
group_vars <- group_by(d, Species) %>% 
  summarise(Group_Variance=var(Sepal.Length))
max(group_vars$Group_Variance)/min(group_vars$Group_Variance)

#Perform MANOVA
manova_reg <- manova(cbind(Sepal.Length, Sepal.Width) ~ Species, data=d)
#Default test statistic is Pillai's trace
summary(manova_reg, intercept=TRUE)
summary(manova_reg, intercept=TRUE, test="Wilks")
summary(manova_reg, intercept=TRUE, test="Hotelling")
summary(manova_reg, intercept=TRUE, test="Roy")

#Use ANOVA to see which species differ in sepal length and width
summary.aov(manova_reg)
