'''
Model Picking Assistant
'''

print("What is the goal of the analysis? \n"
      + "1 - Regression (predicting a number) \n"
      + "2 - Classification (predicting a category) \n"
      + "3 - Forecasting a time series \n"
      + "4 - Finding the most probable sequence of events \n"
      + "5 - Interpolating or smoothing a continuous function \n"
      + "6 - Finding outliers \n")
while True:
    try:
        task = int(input("Enter option: "))
        break
    except ValueError:
        print("Invalid option entered.")

#Regression
if task == 1:
    print("How many target variables are there? \n"
          + "1 - One \n"
          + "2 - Two or more \n")
    while True:
        try:
            targets = int(input("Enter option: "))
            break
        except ValueError:
            print("Invalid option entered.")
    if targets == 1:
        print("How many predictor variables are there? \n"
              + "1 - One \n"
              + "2 - Two or more \n")
        while True:
            try:
                predictors = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if predictors == 1:
            print("Which types of predictors are in the dataset? \n"
                  + "1 - Only numeric (continuous or integer) \n"
                  + "2 - Only categorical \n"
                  + "3 - Mixed \n")
            while True:
                try:
                    predictor_types = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if predictor_types == 1:
                print("You can use Pearson correlation or linear regression. \n"
                      + "If the assumptions for these models are not met, \n"
                      + "you can use bootstrapped correlation or regression, \n"
                      + "Spearman's rank correlation, or Kendall's tau correlation.")
            elif predictor_types == 2:
                print("How many categories does the predictor have? \n"
                      + "1 - Two \n"
                      + "2 - More than two \n")
                while True:
                    try:
                        predictor_categories = int(input("Enter option: "))
                        break
                    except ValueError:
                        print("Invalid option entered.")
                if predictor_categories == 1:
                    print("Do the categories measure the same entity or different entities? \n"
                          + "1 - The same entity \n"
                          + "2 - Different entities \n")
                    while True:
                        try:
                            predictor_category_entity = int(input("Enter option: "))
                            break
                        except ValueError:
                            print("Invalid option entered.")
                    if predictor_category_entity == 1:
                        print("You can use a Dependent t-test. \n"
                              + "If the assumptions for the t-test are not met, \n"
                              + "you can bootstrap it or use the Wilcoxon test.")
                    elif predictor_category_entity == 2:
                        print("You can use an Independent t-test or Point-biserial correlation. \n"
                              + "If the assumptions for these models are not met, \n"
                              + "you can bootstrap the t-test or use the Mann-Whitney test.")
                elif predictor_categories == 2:
                    print("Do the categories measure the same entity or different entities? \n"
                          + "1 - The same entity \n"
                          + "2 - Different entities \n")
                    while True:
                        try:
                            predictor_category_entity = int(input("Enter option: "))
                            break
                        except ValueError:
                            print("Invalid option entered.")
                    if predictor_category_entity == 1:
                        print("You can use a One-way repeated measures ANOVA. \n"
                              + "If the assumptions for this model are not met, \n"
                              + "you can bootstrap it or use the Friedman's ANOVA.")
                    elif predictor_category_entity == 2:
                        print("You can use a One-way independent ANOVA. \n"
                              + "If the assumptions for this model are not met, \n"
                              + "you can bootstrap it or use the Kruskal-Wallis test.")
            elif predictor_types == 3:
                print("Sorry.  There is no way to model this data.")
        elif predictors == 2:
            print("Which types of predictors are in the dataset? \n"
                  + "1 - Only numeric (continuous or integer) \n"
                  + "2 - Only categorical \n"
                  + "3 - Mixed \n")
            while True:
                try:
                    predictor_types = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if predictor_types == 1:
                print("You can use multiple regression, regularized regression, or Bayesian regression. \n"
                      + "If the assumptions for regression are not met (excluding Bayesian regression), \n"
                      + "you can bootstrap it, use cross validation, polynomial regression, \n"
                      + "a generalized additive model, KNN, or a regression tree.  \n"
                      + "If the target variable is a count or integer variable, then \n"
                      + "Poisson and negative binomial regression are also options.")
            elif predictor_types == 2:
                print("Do the categories measure the same entity or different entities? \n"
                      + "1 - The same entity \n"
                      + "2 - Different entities \n"
                      + "3 - Both")
                while True:
                    try:
                        predictor_category_entity = int(input("Enter option: "))
                        break
                    except ValueError:
                        print("Invalid option entered.")
                if predictor_category_entity == 1:
                    print("You can use factorial repeated measures ANOVA. \n"
                          + "If the assumptions for this model are not met, \n"
                          + "you can bootstrap it.")
                elif predictor_category_entity == 2:
                    print("You can use multiple regression or independent factorial ANOVA. \n"
                          + "If the assumptions for these models are not met, \n"
                          + "you can bootstrap them, use cross validation, polynomial regression, \n"
                          + "a generalized additive model, KNN with Gower distance, or a regression tree.")
                elif predictor_category_entity == 3:
                    print("You can use factorial mixed ANOVA. \n"
                          + "If the assumptions for this model are not met, \n"
                          + "you can bootstrap it.")
            elif predictor_types == 3:
                print("You can use multiple regression, ANCOVA, or Bayesian regression. \n"
                      + "If the assumptions for these models are not met (excluding Bayesian regression), \n"
                      + "you can bootstrap them, use cross validation, polynomial regression, \n"
                      + "a generalized additive model, or a regression tree.")
    elif targets == 2:
        print("How many predictor variables are there? \n"
              + "1 - One \n"
              + "2 - Two or more \n")
        while True:
            try:
                predictors = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if predictors == 1:
            print("Which types of predictors are in the dataset? \n"
                  + "1 - Only numeric (continuous or integer) \n"
                  + "2 - Only categorical \n"
                  + "3 - Mixed \n")
            while True:
                try:
                    predictor_types = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if predictor_types == 2:
                print("You can use MANOVA.")
            else:
                print("Sorry.  There is no way to model this data.")
        elif predictors == 2:
            print("Which types of predictors are in the dataset? \n"
                  + "1 - Only numeric (continuous or integer) \n"
                  + "2 - Only categorical \n"
                  + "3 - Mixed \n")
            while True:
                try:
                    predictor_types = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if predictor_types == 1:
                print("Sorry.  There is no way to model this data.")
            elif predictor_types == 2:
                print("You can use factorial MANOVA.")
            elif predictor_types == 3:
                print("You can use MANCOVA.")
#Classification
elif task == 2:
    print("How many predictor variables are there? \n"
          + "1 - One \n"
          + "2 - Two or more \n")
    while True:
        try:
            predictors = int(input("Enter option: "))
            break
        except ValueError:
            print("Invalid option entered.")
    if predictors == 1:
        print("Which types of predictors are in the dataset? \n"
              + "1 - Only numeric (continuous or integer) \n"
              + "2 - Only categorical \n"
              + "3 - Mixed \n")
        while True:
            try:
                predictor_types = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if predictor_types == 1:
            print("You can use logistic regression, biserial or point-biserial correlation, \n"
                  + "or linear discriminant analysis. \n"
                  + "If the assumptions for these models are not met, \n"
                  + "you can bootstrap them or use cross validation, \n"
                  + "quadratic discriminant analysis, KNN, a classification tree, \n"
                  + "or Naive Bayes.")
        elif predictor_types == 2:
            print("Assuming the categories measure different entities, \n"
                  + "you can use the chi-square test or likelihood ratio.")
        elif predictor_types == 3:
            print("Sorry.  There is no way to model this data.")
    elif predictors == 2:
        print("Which types of predictors are in the dataset? \n"
              + "1 - Only numeric (continuous or integer) \n"
              + "2 - Only categorical \n"
              + "3 - Mixed \n")
        while True:
            try:
                predictor_types = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if predictor_types == 1:
            print("You can use logistic regression or linear discriminant analysis. \n"
                  + "If the assumptions for these models are not met, \n"
                  + "you can bootstrap them or use cross validation, \n"
                  + "quadratic discriminant analysis, KNN, a classification tree, \n"
                  + "or Naive Bayes.")
        elif predictor_types == 2:
            print("Assuming the categories measure different entities, \n"
                  + "you can use loglinear analysis.")
        elif predictor_types == 3:
            print("Assuming the categories measure different entities, \n"
                  + "you can use logistic regression, linear discriminant analysis, \n"
                  + "KNN, a classification tree, or Naive Bayes.")
#Forecasting
elif task == 3:
    print("Is the variance of the time series constant over time? \n"
          + "1 - Yes \n"
          + "2 - No \n")
    while True:
        try:
            non_constant_variance = int(input("Enter option: "))
            break
        except ValueError:
            print("Invalid option entered.")
    if non_constant_variance == 1:
        print("Is the time series by influenced by its past behavior or random shocks? \n"
              + "1 - The time series depends only on random variation in its past behavior. \n"
              + "2 - The time series depends only on random shocks and not on its past behavior. \n"
              + "3 - The time series depends on both random shocks and its past behavior. \n")
        while True:
            try:
                arima_type = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if arima_type == 1:
            print("You can use any ARIMA model, but a series that depends only on its past \n"
                  + "behavior can be most easily modeled as an AR process.")
        elif arima_type == 2:
            print("You can use any ARIMA model, but a series that depends only on random \n"
                  + "shocks can be most easily modeled as an MA process.")
        elif arima_type == 3:
            print("You can use any ARIMA model.  A series that depends both on its past \n"
                  + "behavior and random shocks is best modeled as an ARIMA process, \n"
                  + "rather than as an AR or MA process.")
    elif non_constant_variance == 2:
        print("You can use GARCH.")
#Sequence of events
elif task == 4:
    print("Does the sequence contain a hidden state? \n"
          + "1 - Yes \n"
          + "2 - No \n")
    while True:
        try:
            hidden_state = int(input("Enter option: "))
            break
        except ValueError:
            print("Invalid option entered.")
    if hidden_state == 1:
        print("Are the events part of a discrete or continuous system? \n"
              + "1 - Discrete \n"
              + "2 - Continuous \n")
        while True:
            try:
                hidden_state_type = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if hidden_state_type == 1:
            print("Assuming states exhibit the Markov property, \n"
                  + "you can use a hidden Markov model.")
        elif hidden_state_type == 2:
            print("You can use a Kalman filter.")
    elif hidden_state == 2:
        print("Assuming states exhibit the Markov property, \n"
              + "you can use a Markov chain.  Otherwise, \n"
              + "you can use a Bayesian network.")
#Interpolation/smoothing
elif task == 5:
    print("You can use spline interpolation, a moving average, \n"
          +"a regression spline, LOESS, or a Kalman filter.")
#Finding outliers
elif task == 6:
    print("Are you looking for outliers in 1 variable or over many? \n"
          + "1 - Univariate \n"
          + "2 - Multivariate \n")
    while True:
        try:
            nbr_variables = int(input("Enter option: "))
            break
        except ValueError:
            print("Invalid option entered.")
    if nbr_variables == 1:
        print("Is the variable normally distributed? \n"
              + "1 - Yes \n"
              + "2 - No \n")
        while True:
            try:
                normal = int(input("Enter option: "))
                break
            except ValueError:
                print("Invalid option entered.")
        if normal == 1:
            print("How many outliers are expected? \n"
                  + "1 - One \n"
                  + "2 - Two or more, or unknown \n")
            while True:
                try:
                    nbr_outliers = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if nbr_outliers == 1:
                print("You can use the IQR test, Grubb's test, the z-score test, \n"
                      + "or MAD.")
            elif nbr_outliers == 2:
                print("You can use the IQR test or the extreme studentized deviates \n"
                      + "test (a.k.a. iterative Grubb's), the z-score test, or MAD.")
        elif normal == 2:
            print("Does that data contain naturally occurring numbers that follow an expected \n"
                  + "non-normal distribution? \n"
                  + "1 - Yes \n"
                  + "2 - No \n")
            while True:
                try:
                    benfords = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if benfords == 1:
                print("You can use Benford's Law or a custom 'My Law' and test for conformity \n"
                      + "using MAD, the z-test, or the chi-square test.")
            elif benfords == 2:
                print("Is the variable categorical or numeric? \n"
                      + "1 - Numeric \n"
                      + "2 - Categorical \n")
                while True:
                    try:
                        var_type = int(input("Enter option: "))
                        break
                    except ValueError:
                        print("Invalid option entered.")
                if var_type == 1:
                    print("You can use the modified z-score test, MAD, or visual inspection.")
                elif var_type ==2:
                    print("You can use the modified z-score test, MAD, the chi-square test, \n"
                          + "or visual inspection.")
        elif nbr_variables == 2:
            print("Are you looking for outliers across 2 variables or more? \n"
                  + "1 - Two (bivariate) \n"
                  + "2 - More than two (multivariate) \n")
            while True:
                try:
                    bivariate = int(input("Enter option: "))
                    break
                except ValueError:
                    print("Invalid option entered.")
            if bivariate == 1:
                print("What are the variables types? \n"
                      + "1 - Both are continuous, or one or both are integers to be treated as numeric \n"
                      + "2 - Both are categorical, or one is categorical and the other is an integer to be treated as categorical \n"
                      + "3 - One is continuous or an integer to be treated as numeric, and the other is categorical or an integer to be treated as categorical")
                while True:
                    try:
                        var_types = int(input("Enter option: "))
                        break
                    except ValueError:
                        print("Invalid option entered.")
                if var_types == 1:
                    print("You can use KNN or visual inspection.")
                elif var_types == 2:
                    print("You can use a mosaic plot, heat map, or other visual inspection.")
                elif var_types == 3:
                    print("You can use box plots by each category of the categorical variable.")
            elif bivariate == 2:
                print("What are the variables types? \n"
                      + "1 - All are continuous \n"
                      + "2 - All are categorical \n"
                      + "3 - The variables are mixed")
                while True:
                    try:
                        var_types = int(input("Enter option: "))
                        break
                    except ValueError:
                        print("Invalid option entered.")
                if var_types == 1:
                    print("You can use KNN or a parallel coordinates plot.")
                else:
                    print("You can use a parallel coordinates plot.")


