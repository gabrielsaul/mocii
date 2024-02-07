library("iml")
library("mlr")
library("mlrCPO")

data = read.csv("datasets/processed/german_credit_data.csv", stringsAsFactors = TRUE)
data = na.omit(data)
names(data) = tolower(names(data))

test_sample_i = sample(nrow(data), nrow(data) %/% 4)
test_data = data[test_sample_i,]
data = data[-test_sample_i,]


data.task = makeClassifTask(id = "german",
                            data = data,
                            target = "risk",
                            positive = "good")

data.learner = makeLearner("classif.randomForest", 
                           predict.type = "prob", 
                           fix.factors.prediction = TRUE)

# Define parameter space for tuning
param_grid = makeParamSet(
  makeIntegerParam("ntree", lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 1, upper = ncol(data) - 1)
  # Add more parameters as needed
)

# Choose tuning method (e.g., random search)
tune_control = makeTuneControlRandom(maxit = 25L)

parallelMap::parallelStartSocket(parallel::detectCores(), level = "mlr.tuneParams")

# Perform hyperparameter tuning
data.learner.tuned = tuneParams(data.learner, 
                                task = data.task, 
                                resampling = cv10, 
                                par.set = param_grid, 
                                control = tune_control)

data.model.tuned = mlr::train(data.learner.tuned$learner, data.task)

parallelMap::parallelStop()

# Create Predictor object with the tuned model
pred = Predictor$new(model = data.model.tuned,
                     data = data,
                     class = "good")

sink("rf_acc_test.txt")
corr = 0
ns = 0
for (i in 1:nrow(test_data)) {
  x.interest = test_data[i,]
  result = pred$predict(x.interest)
  print(x.interest)
  print(result)
  
  if ((result >= 0.5 & x.interest[["risk"]] == "good") | (result < 0.5 & x.interest[["risk"]] == "bad")) {
    corr = corr + 1
  }
  ns = ns + 1
}
corr / ns
sink()