# Load necessary library packages.
library("mlr")
library("mlrCPO")
library("janitor")

# Load IML.
devtools::load_all("../iml", export_all = FALSE)

# Indexing constants.
POS_CLV_INDEX = 1
NEG_CLV_INDEX = 2
LOW_TARG_RANGE = 1
UPP_TARG_RANGE = 2

# ML algorithm types.
NN_ALG_ID = "nn"
RF_ALG_ID = "rf"
SVM_ALG_ID = "svm"

# Max tuning iterations & resampling.
TUNEITERS = 100L
RESAMPLING = cv5


# Struct class for datasets.
dataStruct <- setClass(
  "DataStruct",
  slots = c(df = "data.frame",
            id = "character",
            target = "character",
            target_values = "character",
            nonactionable = "character")
)

# Struct class for ML algorithm types.
mlAlgStruct <- setClass(
  "MLAlg",
  slots = c(id = "character",
            target_range = "numeric")
)

# Struct class for resilience results.
resilienceTestResults <- setClass(
  "resilienceTestResults",
  slots = c(resilience_df = "data.frame",
            resilience_feat = "numeric",
            resilience_feat_avg = "numeric",
            resilience_feat_max = "numeric",
            resilience_feat_min = "numeric",
            resilience_cf = "numeric",
            resilience_cf_avg = "numeric",
            resilience_cf_max = "numeric",
            resilience_cf_min = "numeric",
            feat_mut_frequency = "numeric",
            feat_mut_total = "numeric",
            cf_total_count = "numeric",
            cf_valid_count = "numeric",
            cf_tested_count = "numeric",
            cf_invalid_count = "numeric",
            points_of_interest_tested = "numeric",
            points_of_interest_rejected = "numeric")
)

# Given a point of interest & a counterfactual, return a data frame of the 
# resilience scores of each feature. 
#
getResilience <- function(x.interest,
                          cf,
                          pred,
                          ml_alg_target_range,
                          max_feat_values,
                          min_feat_values,
                          data_feat_types) {

  # Empty resilience dataframe.
  cf_nms = names(cf)
  resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), 
                           cf_nms)
  
  # Check if the counterfactual is testable: Mutated numeric features present.
  changed = getMutNumericFeatures(cf, x.interest)
  if (length(changed) > 0) {
    resilience_df[1,] = rep(NaN, ncol(cf))
  }
  else {
    return(NULL)
  }
  
  writeLines(sprintf("\nNumber of Mutated Features: %d", length(changed)))
  print("\nMutated Features:")
  print(changed)
  
  # For each mutated numeric feature...
  for (k in changed) {
    
    # If between min/max of the feature value in dataset.
    if (cf[[k]] > min_feat_values[[k]] & cf[[k]] < max_feat_values[[k]]) {
      writeLines(sprintf("\nMutated Feature: %s", k))
      
      # Set limit & direction of the MV search.
      limit = 0
      flip = 1
      if (cf[[k]] < x.interest[[k]]) {
        limit = min_feat_values[[k]]
        flip = -1
      }
      else {
        limit = max_feat_values[[k]]
        flip = 1
      }
      
      # Set increment.
      inc = (limit - cf[[k]]) / 10
      
      # Adjust increment for integer-valued features.
      if (data_feat_types[k] == "integer") {
        inc = round(inc)
        
        # If below 1, default to 1.
        if (inc == 0) {
          inc = 1 * flip
        }
      }
      
      # Set maximum number of steps for MV search.
      max_steps = abs(limit - cf[[k]]) %/% abs(inc)
      print("\nMAX_STEPS:")
      print(max_steps)
      
      writeLines(sprintf("\nIncrement: %f", inc))
      
      # Mutate feature to search for closest MV.
      successful_steps = 0
      while (successful_steps < max_steps) {
        cf[k] = cf[k] + inc
        print(cf)
        
        result = pred$predict(cf)
        print(result)
        if (isNegativeClass(result, ml_alg_target_range)) {
          print("!MV! -- MONOTONICITY VIOLATION")
          break;
        }
        
        successful_steps = successful_steps + 1
      }
      
      # Log step ratio for feature.
      resilience_df[1,k] = successful_steps / max_steps
      
      writeLines(sprintf("Successful Steps: %d/%d", successful_steps, max_steps))
      print("\n###########################################################\n")
    }
    else {
      # Feature is already at min/max: Assume full resilience.
      print("Already at min/max.")
      print("Full resilience.")
      resilience_df[1,k] = 1.0
    }
  }
  
  return(resilience_df)
}

# Return a predictor based on an ML model (of a given type) that is trained on
# given data.
#
# Options:
#   nn    neural network
#   rf    random forest
#   svm   support vector machine
#
getPredictor <- function(ml_alg_id, 
                         data,
                         data_id,
                         target,
                         target_values) {
  
  # Task for classification.
  data.task = makeClassifTask(id = data_id,
                              data = data,
                              target = target,
                              positive = target_values[POS_CLV_INDEX])
  
  # Initialise parallelisation.
  parallelMap::parallelStartSocket(parallel::detectCores(), level = "mlr.tuneParams")
  
  # Set seed.
  set.seed(as.numeric(Sys.time()))
  
  # Choose & train the model and set the predictor.
  pred = NULL
  if (ml_alg_id == NN_ALG_ID) {
    
    # Learner: Neural network.
    lrn = makeLearner("classif.nnet",
                      predict.type = "prob",
                      fix.factors.prediction = TRUE)
    
    # Normalisation/dummy encode.
    data.lrn = cpoScale() %>>% cpoDummyEncode() %>>% lrn
  
    # Parameters for tuning.
    param_grid = makeParamSet(
      makeIntegerParam("size", lower = 1, upper = 10),
      makeNumericParam("decay", lower = 0.1, upper = 0.9)
    )
    
    # Random search for tuning method.
    tune_control = makeTuneControlRandom(maxit = TUNEITERS)
    
    # Tune.
    data.lrn.tuned = tuneParams(data.lrn, 
                                task = data.task, 
                                resampling = RESAMPLING, 
                                par.set = param_grid, 
                                control = tune_control)
    
    # Train the model.
    data.model = mlr::train(data.lrn.tuned$learner, data.task)
    
    # Set as predictor.
    pred = Predictor$new(model = data.model,
                         data = data,
                         class = target_values[POS_CLV_INDEX])
  }
  else if (ml_alg_id == RF_ALG_ID) {
    
    # Learner: Random Forest.
    lrn = makeLearner("classif.randomForest", 
                      predict.type = "prob", 
                      fix.factors.prediction = TRUE)

    # Parameters for tuning.
    param_grid = makeParamSet(
      makeIntegerParam("ntree", lower = 50, upper = 500),
      makeIntegerParam("mtry", lower = 1, upper = ncol(data) - 1)
    )
    
    # Random search for tuning method.
    tune_control = makeTuneControlRandom(maxit = TUNEITERS)
    
    # Tune.
    data.lrn.tuned = tuneParams(lrn, 
                                task = data.task, 
                                resampling = cv10, 
                                par.set = param_grid, 
                                control = tune_control)
    
    # Train the model.
    data.model = mlr::train(data.lrn.tuned$learner, data.task)
    
    # Set as predictor.
    pred = Predictor$new(model = data.model,
                         data = data,
                         class = target_values[POS_CLV_INDEX])
  }
  else if (ml_alg_id == SVM_ALG_ID) {
    
    # Learner: Support Vector Machine.
    lrn = makeLearner("classif.svm", predict.type = "prob")
    
    # Normalisation/dummy encoding preprocessing.
    data.lrn = cpoScale() %>>% cpoDummyEncode() %>>% lrn
    
    # Parameters for tuning.
    param.set = pSS(
      cost: numeric[0.01, 1]
    )
    
    # Tune.
    ctrl = makeTuneControlRandom(maxit = TUNEITERS * length(param.set$pars))
    lrn.tuning = makeTuneWrapper(lrn, RESAMPLING, list(mlr::acc), param.set, ctrl, show.info = FALSE)
    res = tuneParams(lrn, data.task, RESAMPLING, par.set = param.set, control = ctrl,
                     show.info = FALSE)
    performance = resample(lrn.tuning, data.task, RESAMPLING, list(mlr::acc))$aggr
    data.lrn = setHyperPars2(data.lrn, res$x) 
    
    # Train the model.
    data.model = mlr::train(data.lrn, data.task)
    
    # Set as predictor.
    pred = Predictor$new(model = data.model, 
                         data = data, 
                         class = target_values[POS_CLV_INDEX],
                         conditional = FALSE)
    
    # Fit conditional inference trees.
    ctr = partykit::ctree_control(maxdepth = 5L)
    set.seed(1234)
    pred$conditionals = fit_conditionals(pred$data$get.x(), ctrl = ctr)
  }
  else {
    stop("Error: Invalid ML algorithm ID passed to getPredictor()")
  }
  
  # Stop parallelisation.
  parallelMap::parallelStop()
  
  return(pred)
}


# Generate a results object for a resilience test.
generateResilienceTestResults <- function(resilience_df,
                                          cf_count,
                                          cf_valid_count,
                                          cf_test_count,
                                          cf_inv_count,
                                          points_of_interest_tested,
                                          points_of_interest_rejected) {
  # Feature resilience scores.
  resilience_feat = apply(resilience_df, 2, function(x) mean(na.omit(x)))
  
  # Average feature resilience.
  resilience_feat_avg = mean(na.omit(resilience_feat))
  
  # Maximum & minimum feature resilience.
  resilience_feat_max = max(na.omit(resilience_feat))
  resilience_feat_min = min(na.omit(resilience_feat))
  
  # Counterfactual reslience scores.
  resilience_cf = apply(resilience_df, 1, function(x) mean(na.omit(x)))
  
  # Average counterfactual resilience.
  resilience_cf_avg = mean(resilience_cf)
  
  # Maximum & minimum counterfactual resilience.
  resilience_cf_max = max(resilience_cf)
  resilience_cf_min = min(resilience_cf)
  
  # Count of mutation instances per feature.
  feat_mut_frequency = apply(resilience_df, 2, function(x) length(na.omit(x)))
  
  # Total count of mutation instances across all features.
  feat_mut_total = sum(na.omit(feat_mut_frequency))
  
  
  return(resilienceTestResults("resilience_df" = resilience_df,
                               "resilience_feat" = resilience_feat,
                               "resilience_feat_avg" = resilience_feat_avg,
                               "resilience_feat_max" = resilience_feat_max,
                               "resilience_feat_min" = resilience_feat_min,
                               "resilience_cf" = resilience_cf,
                               "resilience_cf_avg" = resilience_cf_avg,
                               "resilience_cf_max" = resilience_cf_max,
                               "resilience_cf_min" = resilience_cf_min,
                               "feat_mut_frequency" = feat_mut_frequency,
                               "feat_mut_total" = feat_mut_total,
                               "cf_total_count" = cf_count, 
                               "cf_valid_count" = cf_valid_count,
                               "cf_tested_count" = cf_test_count,
                               "cf_invalid_count" = cf_inv_count,
                               "points_of_interest_tested" = points_of_interest_tested,
                               "points_of_interest_rejected" = points_of_interest_rejected))
}

# Return the names of all columns with different numeric values in the given
# counterfactual when compared to the other given data point.
getMutNumericFeatures <- function(cf, orig) {
  if (ncol(cf) != ncol(orig)) {
    return(FALSE)
  }
  
  nms = c()
  for (nm in names(cf)) {
    if (is.numeric(cf[[nm]]) & cf[[nm]] != orig[[nm]]) {
      nms = append(nms, nm)
    }
  }
  return(nms)
}

# Insert the given new row into the given data frame after the given row index.
insertRow <- function(df, new_row, r) { 
  df_new = rbind(df[1:r, ], new_row, df[- (1:r), ])         
  rownames(df_new) = 1:nrow(df_new)     
  return(df_new) 
} 

# Return TRUE if the given string is numeric, else FALSE.
isNumericString <- function(s) {
  suppressWarnings(return(!is.na(as.numeric(s))))
}

# Return TRUE if the given result equates to a negative class prediction,
# else return FALSE.
isNegativeClass <- function(result, ml_alg_target_range) {
  lower = ml_alg_target_range[LOW_TARG_RANGE]
  upper = ml_alg_target_range[UPP_TARG_RANGE]
  return(!(result >= lower & result <= upper))
}