#STAT454: Machine Learning - Project 

# install packages if needed
install.packages("caret")
install.packages("caTools")
install.packages("MASS")
install.packages("e1071")
install.packages("naivebayes")
install.packages("RSNNS")
install.packages("nnet")
install.packages("randomForest")

accuracy = modelDriver(simulations=50) # run to execute program

# function loads in required libraries
loadLibraries = function() {
  
  library(e1071)
  library(caret)
  library(caTools)
  library(MASS)
  library(naivebayes)
  library(RSNNS)
  library(nnet)
  library(randomForest)
  
}

# function trains various models and returns their final form
modelData = function(msft.train) {
  
  train.X = msft.train[, -6]
  train.Y = as.factor(make.names(msft.train$Increase))
  
  #10-Fold-CV repeated 3 times
  fitControl = trainControl(method="repeatedcv", number=10, repeats=3, classProbs=TRUE)  

  #Random Forest (mtry)
  message("   Training with Random Forest")
  random.forest = caret::train(train.X, train.Y, 
                               method="rf", 
                               trControl=fitControl)
  
  #Boosted Logistic Regression (nIter)
  message("   Training with Boosted Logisitic Regression")
  boosted.logistic = caret::train(train.X, train.Y, 
                                  method="LogitBoost", 
                                  trControl=fitControl)
  
  #Linear Support Vector Machines with Class Weigths (cost, weight)
  message("   Training with Linear Support Vector Machines with Class Weights")
  support.vm = caret::train(train.X, train.Y, 
                            method="svmLinearWeights", 
                            trControl=fitControl,
                            preProcess=c("center", "scale"))

  #Neural Network (size, decay)
  message("   Training with Neural Network")
  grid.nnet = expand.grid(size=10, decay=0)
  neural.net = caret::train(train.X, train.Y,
                            method="nnet",
                            trControl=fitControl,
                            preProcess=c("center", "scale"),
                            tuneGrid=grid.nnet,
                            trace=FALSE,
                            reltol=1e-3)
  
  #Linear Discriminant Analysis (dimen)
  message("   Training with Linear Discriminant Analysis")
  linear.da = caret::train(train.X, train.Y,
                           method="lda",
                           trControl=fitControl,
                           preProcess=c("center", "scale"))
  
  #Naive Bayes (laplace, usekernel, adjust)
  message("   Training with Naive Bayes")
  naive.bayes = caret::train(train.X, train.Y,
                             method="naive_bayes",
                             trControl=fitControl,
                             preProcess=c("center", "scale"))
  
  #k-Nearest Neighbors (k)
  message("   Training with k-Nearest Neighbors")
  kn.neighbors = caret::train(train.X, train.Y,
                              method="knn",
                              trControl=fitControl,
                              preProcess=c("center", "scale"))
  
  #Multi-Layer Perceptron (size)
  message("   Training with Multi-Layer Perceptron")
  ml.perceptron = caret::train(train.X, train.Y,
                               method="mlp",
                               trControl=fitControl,
                               preProcess=c("center", "scale"))
  
  models = list(random.forest,
                boosted.logistic,
                support.vm,
                neural.net,
                linear.da,
                naive.bayes,
                kn.neighbors,
                ml.perceptron)
  
  return(models)
  
}

# function evaluates the performance of trained models on test set
evaluateModels = function(models, msft.test) {
  
  test.Y = msft.test[, 6]
  test.X = msft.test[, -6]
  
  #Random Forest
  rf.probs = predict(models[1], test.X, type="prob")[[1]]
  #Boosted Logistic Regression
  blr.probs = predict(models[2], test.X, type="prob")[[1]]
  #Support Vector Machine
  svm.probs = predict(models[3], test.X, type="prob")[[1]]
  #Neural Network
  nn.probs = predict(models[4], test.X, type="prob")[[1]]
  #Linear Disciminant Analysis
  lda.probs = predict(models[5], test.X, type="prob")[[1]]
  #Naive Bayes
  nb.probs = predict(models[6], test.X, type="prob")[[1]]
  #k-Nearest Neighbors
  knn.probs = predict(models[7], test.X, type="prob")[[1]]
  #Multi-Layer Perceptron
  mlp.probs = predict(models[8], test.X, type="prob")[[1]]
  
  #use predicted probabilities to predict class 
  rf.preds = ifelse(rf.probs$X0 > rf.probs$X1, 0, 1)
  blr.preds = ifelse(blr.probs$X0 > blr.probs$X1, 0 ,1)
  svm.preds = ifelse(svm.probs$X0 > svm.probs$X1, 0 ,1)
  nn.preds = ifelse(nn.probs$X0 > nn.probs$X1, 0 ,1)
  lda.preds = ifelse(lda.probs$X0 > lda.probs$X1, 0 ,1)
  nb.preds = ifelse(nb.probs$X0 > nb.probs$X1, 0 ,1)
  knn.preds = ifelse(knn.probs$X0 > knn.probs$X1, 0 ,1)
  mlp.preds = ifelse(mlp.probs$X0 > mlp.probs$X1, 0 ,1)
  
  #compare predicted classes with actual classes
  rf.accuracy = 1-mean(rf.preds != test.Y)
  blr.accuracy = 1-mean(blr.preds != test.Y)
  svm.accuracy = 1-mean(svm.preds != test.Y)
  nn.accuracy = 1-mean(nn.preds != test.Y)
  lda.accuracy = 1-mean(lda.preds != test.Y)
  nb.accuracy = 1-mean(nb.preds != test.Y)
  knn.accuracy = 1-mean(knn.preds != test.Y)
  mlp.accuracy = 1-mean(mlp.preds != test.Y)
  
  model.accuracies = list(rf = rf.accuracy,
                          blr = blr.accuracy,
                          svm = svm.accuracy,
                          nn = nn.accuracy,
                          lda = lda.accuracy,
                          nb = nb.accuracy,
                          knn = knn.accuracy,
                          mlp = mlp.accuracy)
  
  return(model.accuracies)
  
}

# function 'drives' the program executing the workflow
modelDriver = function(simulations) {
  
  setwd("~/Desktop/STAT454/Project")
  load("data/msft.rda")
  load("data/msft_new.rda")
  loadLibraries()
  
  #intitialize results matrices
  accuracy = matrix(nrow=8, ncol=simulations)
  new_accuracy = matrix(nrow=8, ncol=simulations)
  
  for (i in 1:simulations) {
    
    message(paste("Simulation", i, sep=" "))
    
    #split data into train/test
    set.seed(i)
    indexes = createDataPartition(msft$Increase, times=1, p=0.7, list=FALSE)
    msft.train = msft[indexes, ]
    msft.test = msft[-indexes, ]
    
    #train models
    models = modelData(msft.train)
    
    message("   Evaluating models performance")
    
    #make model predictions on holdout set and calculate accuracy 
    accuracy[ ,i] = as.numeric(unlist(evaluateModels(models, msft.test)))
    
    #make predictions on 30 most recent trading days and calculate accuracy
    new_accuracy[ ,i] = as.numeric(unlist(evaluateModels(models, msft_new)))
    
  }
  
  save(accuracy, file="modelPerformance.rda")
  save(new_accuracy, file="practicalPerformance.rda")
  
  return(accuracy)
  
}

