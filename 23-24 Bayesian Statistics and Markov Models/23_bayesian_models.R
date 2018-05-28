#R code for chapter 23 of DSILT: Statistics

#-------------------------------------------------------------------------------------------------#
#---------------------------------------Bayesian Networks-----------------------------------------#
#-------------------------------------------------------------------------------------------------#

library(bnlearn)
set.seed(14)
d <- coronary
colnames(d) <- c('smoker', 'mental_work',
                 'physical_work', 'blood_pressure',
                 'protein_ratio', 
                 'heart_disease')
head(d)
#Find a good network structure using hill climbing algorithm
net <- hc(d)
plot(net)
#Some causal relationships look ok, but others do not
#Remove link from mental_work to heart_disease, bc heart disease does not depend on mental work
head(net$arcs)  #Shows how links are structured
net$arcs <- net$arcs[-which((net$arcs[,'from'] == 'mental_work' & net$arcs[,'to'] == 'heart_disease')),]
#Add link from smoker to heart_disease
set.arc(net, from='smoker', to='heart_disease')
amat(net)['smoker', 'heart_disease'] <- 1  #Update adjacency matrix
plot(net)

#Create conditional probability tables for each node
cpts <- bn.fit(net, data=d)
print(cpts$blood_pressure)  #BP only conditional on smoking
print(cpts$protein_ratio)  #Protein conditional on smoking and mental work

#Perform inference
cpquery(cpts, even=(protein_ratio=="<3"), evidence=(smoker=="no"))
#So there is a 63% probability that protein ratio <3 if smoker is no

#What is the probability that a non-smoker with blood pressure > 140 has protein ratio < 3?
cpquery(cpts, event=(protein_ratio=='<3'), evidence=(smoker=='no' & blood_pressure=='>140'))

#What is the probability that a smoker has heart disease?
cpquery(cpts, event=(heart_disease=='pos'), evidence=(smoker=='yes'))

##############

library(bnlearn)
set.seed(14)

#Look at skewed data
hist(faithful$eruptions)

#Discreteize dataframe of entirely continuous variables
dfaithful <- discretize(faithful, metho='hartemink', breaks=3, ibreaks=25, idisc='quantile')

#Bootstrap resampling to average multiple DAGs, where R = number of network strcutures
boot <- boot.strength(dfaithful, R=500, algorithm='hc', algorithm.args=list(score='bde', iss=10))

#Look at strength (edge frequency), direction
boot[boot$strength > 0.85 & boot$direction >= 0.5,]
#Since edge strengths are the same, the directions are not well established
avg.boot <- averaged.network(boot, threshold=1)
plot(avg.boot)

#-------------------------------------------------------------------------------------------------#
#----------------------------------Naive Bayes Classification-------------------------------------#
#-------------------------------------------------------------------------------------------------#

library(e1071)

nb_model <- naiveBayes(Species ~ ., data=iris)
nb_model
preds <- predict(nb_model, iris[,-which(colnames(iris) %in% c('Species'))])
table(preds, iris$Species)
