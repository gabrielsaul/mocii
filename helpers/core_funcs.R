# Load necessary library packages.
library("mlr")
library("mlrCPO")
library("janitor")

# Load IML.
devtools::load_all("../iml", export_all = FALSE)

# Sentinel values for fitness objectives (in default order).
OBJ_VALID     = 1
OBJ_SIMILAR   = 2
OBJ_SPARSE    = 3
OBJ_PLAUSIBLE = 4

# Number of fitness objectives.
N_OBJ         = 4

# Number of metadata columns in MOC results.
NCOLS_CF_METADATA = 5

# Indexing constants.
POS_CLV_INDEX = 1
NEG_CLV_INDEX = 2
LOW_TARG_RANGE = 1
UPP_TARG_RANGE = 2

# ML algorithm types.
NN_ALG_ID = "nn"
RF_ALG_ID = "rf"
SVM_ALG_ID = "svm"

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

# Max tuning iterations & resampling.
TUNEITERS = 100L
RESAMPLING = cv5

# Struct class for algorithm performance results.
algorithmTestResults <- setClass(
  "algorithmTestResults",
  slots = c(
    test_run_id = "character",
    runtime_total = "numeric",
    runtime_avg = "numeric",
    generations_total = "integer",
    generations_avg = "numeric",
    predictor = "ANY",
    test_df = "data.frame",
    poi_tested_idxs = "integer",
    poi_tested_count = "integer",
    poi_vs_idxs = "integer",
    poi_vs_count = "integer",
    poi_res_idxs = "integer",
    poi_res_count = "integer",
    poi_rejected_count = "integer",
    cf_df = "data.frame",
    cf_total = "integer",
    cf_valid = "integer",
    cf_valid_pct = "numeric",
    cf_invalid = "integer",
    cf_all_invalid = "integer",
    cf_null_returned = "integer",
    fit_avg = "numeric",
    fit_sd = "numeric",
    fit_var = "numeric",
    fit_min = "numeric",
    fit_max = "numeric",
    feat_mut_frequency = "numeric",
    feat_mut_total = "numeric",
    pd_comparisons = "integer",
    pd_wlt_record = "numeric",
    pd_win_pct = "numeric",
    pd_loss_pct = "numeric",
    pd_tie_pct = "numeric",
    lx_comparisons = "integer",
    lx_wlt_record = "numeric",
    lx_win_pct = "numeric",
    lx_loss_pct = "numeric",
    lx_tie_pct = "numeric",
    resilience_df = "data.frame",
    resilience_cf = "numeric",
    resilience_cf_perfect_pct = "numeric",
    resilience_cf_zero_pct = "numeric",
    resilience_cf_avg = "numeric",
    resilience_cf_sd = "numeric", 
    resilience_cf_var = "numeric",
    resilience_cf_min = "numeric",
    resilience_cf_max = "numeric",
    resilience_feat = "numeric",
    resilience_feat_perfect_pct = "numeric",
    resilience_feat_zero_pct = "numeric",
    resilience_feat_sd = "numeric", 
    resilience_feat_var = "numeric",
    resilience_feat_min = "numeric",
    resilience_feat_max = "numeric")
)

# Generate results for algorithm metrics (runtime, generations, etc.)
generateAlgorithmTestResults <- function(test_run_id = "test",
                                         runtime_total,
                                         generations_total,
                                         predictor,
                                         test_df,
                                         poi_tested_idxs,
                                         poi_vs_idxs,
                                         poi_res_idxs,
                                         poi_rejected_count,
                                         cf_df,
                                         cf_all_invalid,
                                         cf_null_returned,
                                         pd_wlt,
                                         lx_wlt,
                                         resilience_df)
{
  # Nunber of points of interest tested.
  poi_tested_count = length(poi_tested_idxs)
  poi_vs_count = length(poi_vs_idxs)
  poi_res_count = length(poi_res_idxs)
  
  # Average runtime of the algorithm.
  runtime_avg = runtime_total / poi_tested_count
  
  # Average generations of the algorithm.
  generations_avg = generations_total / poi_tested_count
  
  # Total number of counterfactuals produced by the algorithm.
  cf_total = nrow(cf_df)
  
  # Total number of valid counterfactuals produced by the algorithm.
  cf_valid = length(which(cf_df$dist.target == 0))
  
  # Percentage of valid counterfactuals produced by the algorithm.
  cf_valid_pct = (cf_valid / cf_total) * 100
  
  # Total number of invalid counterfactuals produced by the algorithm.
  cf_invalid = cf_total - cf_valid
  
  # Average objective fitness values (including prediction).
  fit = cf_df[,((ncol(cf_df) - NCOLS_CF_METADATA) + 1):ncol(cf_df)]
  fit_avg = apply(fit, 2, function(x) mean(na.omit(x)))
  
  # Minimum & maximum fitness values (including prediction).
  fit_min = apply(fit, 2, function(x) min(na.omit(x)))
  fit_max = apply(fit, 2, function(x) max(na.omit(x)))
  
  # Standard deviation & variance of fitness values (including prediction).
  fit_sd = apply(fit, 2, function(x) sd(na.omit(x)))
  fit_var = apply(fit, 2, function(x) var(na.omit(x)))
  
  # Count of Pareto-dominace comparisons.
  pd_comparisons = sum(pd_wlt)
  
  # Win percentage for Pareto-dominance.
  pd_win_pct = (pd_wlt[1] / pd_comparisons) * 100
  
  # Loss percentage for Pareto-dominance.
  pd_loss_pct = (pd_wlt[2] / pd_comparisons) * 100
  
  # Tie percentage for Pareto-dominance.
  pd_tie_pct = (pd_wlt[3] / pd_comparisons) * 100
  
  # Count of lexicographic comparisons.
  lx_comparisons = sum(lx_wlt)
  
  # Win percentage for lexicographic.
  lx_win_pct = (lx_wlt[1] / lx_comparisons) * 100
  
  # Loss percentage for lexicographic.
  lx_loss_pct = (lx_wlt[2] / lx_comparisons) * 100
  
  # Tie percentage for lexicographic.
  lx_tie_pct = (lx_wlt[3] / lx_comparisons) * 100
  
  # Redacted resilience dataframe for feature resilience metrics.
  cs = colSums(is.na(resilience_df))
  csb = colSums(is.na(resilience_df)) < nrow(resilience_df)
  res_nms = names(resilience_df)[colSums(is.na(resilience_df)) < nrow(resilience_df)]
  resilience_df_redacted = resilience_df[,res_nms]
  if (!is.data.frame(resilience_df_redacted)) {
    resilience_df_redacted = setNames(as.data.frame(resilience_df_redacted), res_nms)
  }
  
  # Feature resilience scores (column-wise mean).
  resilience_feat = apply(resilience_df_redacted, 2, function(x) mean(na.omit(x)))
  
  # Percentage of perfectly resilient feature scores.
  f1 = function(x) {
    x = na.omit(x)
    t = table(x)
    return((t[names(t) == 1] / length(x)) * 100)
  }
  resilience_feat_perfect_pct = setNames(rep(0, ncol(resilience_df_redacted)),
                                         names(resilience_df_redacted))
  pp_vec = as.numeric(apply(resilience_df_redacted, 2, f1))
  if (length(pp_vec) > 0) {
    resilience_feat_perfect_pct = setNames(pp_vec, names(resilience_df_redacted))
  }
  
  # Percentage of zero resilience feature scores.
  f2 = function(x) {
    x = na.omit(x)
    t = table(x)
    return((t[names(t) == 0] / length(x)) * 100)
  }
  resilience_feat_zero_pct = setNames(rep(0, ncol(resilience_df_redacted)),
                                      names(resilience_df_redacted))
  zp_vec = as.numeric(apply(resilience_df_redacted, 2, f2))
  if (length(zp_vec) > 0) {
    resilience_feat_zero_pct = setNames(zp_vec, names(resilience_df_redacted))
  }
  
  # Standard deviation & variance of feature resilience.
  resilience_feat_sd = apply(resilience_df_redacted, 2, function(x) sd(na.omit(x)))
  resilience_feat_var = apply(resilience_df_redacted, 2, function(x) var(na.omit(x)))
  
  # Minimum & maximum feature resilience.
  resilience_feat_min = apply(resilience_df_redacted, 2, function(x) min(na.omit(x)))
  resilience_feat_max = apply(resilience_df_redacted, 2, function(x) max(na.omit(x)))
  
  # Counterfactual resilience scores (row-wise mean).
  resilience_cf = na.omit(apply(resilience_df, 1, function(x) mean(na.omit(x))))
  
  # Percentage of perfectly resilient counterfactuals.
  cf_tbl = table(resilience_cf)
  resilience_cf_perfect_pct = (cf_tbl[names(cf_tbl) == 1] / length(resilience_cf)) * 100
  if (length(resilience_cf_perfect_pct) == 0) {
    resilience_cf_perfect_pct = 0
  }
  
  # Percentage of zero resilience counterfactuals.
  resilience_cf_zero_pct = (cf_tbl[names(cf_tbl) == 0] / length(resilience_cf)) * 100
  if (length(resilience_cf_zero_pct) == 0) {
    resilience_cf_zero_pct = 0
  }
  
  # Average counterfactual resilience.
  resilience_cf_avg = mean(resilience_cf)
  
  # Standard deviation & variance of counterfactual resilience.
  resilience_cf_sd = sd(resilience_cf)
  resilience_cf_var = var(resilience_cf)
  
  # Minimum & maximum counterfactual resilience.
  resilience_cf_min = min(resilience_cf)
  resilience_cf_max = max(resilience_cf)
  
  # Count of mutation instances per feature.
  feat_mut_frequency = apply(resilience_df_redacted, 2, function(x) length(na.omit(x)))
  
  # Total count of mutation instances across all features.
  feat_mut_total = sum(na.omit(feat_mut_frequency))

  
  return(algorithmTestResults("test_run_id" = test_run_id,
                              "runtime_total" = runtime_total,
                              "runtime_avg" = runtime_avg,
                              "generations_total" = as.integer(generations_total),
                              "generations_avg" = generations_avg,
                              "predictor" = predictor,
                              "test_df" = test_df,
                              "poi_tested_idxs" = as.integer(poi_tested_idxs),
                              "poi_tested_count" = as.integer(poi_tested_count),
                              "poi_vs_idxs" = as.integer(poi_vs_idxs),
                              "poi_vs_count" = as.integer(poi_vs_count),
                              "poi_res_idxs" = as.integer(poi_res_idxs),
                              "poi_res_count" = as.integer(poi_res_count),
                              "poi_rejected_count" = as.integer(poi_rejected_count),
                              "cf_df" = cf_df,
                              "cf_total" = as.integer(cf_total),
                              "cf_valid" = as.integer(cf_valid),
                              "cf_valid_pct" = cf_valid_pct,
                              "cf_invalid" = as.integer(cf_invalid),
                              "cf_all_invalid" = as.integer(cf_all_invalid),
                              "cf_null_returned" = as.integer(cf_null_returned),
                              "fit_avg" = fit_avg,
                              "fit_sd" = fit_sd,
                              "fit_var" = fit_var,
                              "fit_min" = fit_min,
                              "fit_max" = fit_max,
                              "feat_mut_frequency" = feat_mut_frequency,
                              "feat_mut_total" = feat_mut_total,
                              "pd_comparisons" = as.integer(pd_comparisons),
                              "pd_wlt_record" = pd_wlt,
                              "pd_win_pct" = pd_win_pct,
                              "pd_loss_pct" = pd_loss_pct,
                              "pd_tie_pct" = pd_tie_pct,
                              "lx_comparisons" = as.integer(lx_comparisons),
                              "lx_wlt_record" = lx_wlt,
                              "lx_win_pct" = lx_win_pct,
                              "lx_loss_pct" = lx_loss_pct,
                              "lx_tie_pct" = lx_tie_pct,
                              "resilience_df" = resilience_df,
                              "resilience_cf" = resilience_cf,
                              "resilience_cf_perfect_pct" = resilience_cf_perfect_pct,
                              "resilience_cf_zero_pct" = resilience_cf_zero_pct,
                              "resilience_cf_avg" = resilience_cf_avg,
                              "resilience_cf_sd" = resilience_cf_sd, 
                              "resilience_cf_var" = resilience_cf_var,
                              "resilience_cf_min" = resilience_cf_min,
                              "resilience_cf_max" = resilience_cf_max,
                              "resilience_feat" = resilience_feat,
                              "resilience_feat_perfect_pct" = resilience_feat_perfect_pct,
                              "resilience_feat_zero_pct" = resilience_feat_zero_pct,
                              "resilience_feat_sd" = resilience_feat_sd, 
                              "resilience_feat_var" = resilience_feat_var,
                              "resilience_feat_min" = resilience_feat_min,
                              "resilience_feat_max" = resilience_feat_max))
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
    param.set = makeParamSet(
      makeIntegerParam("size", lower = 1, upper = 5),
      makeNumericParam("decay", lower = 0.1, upper = 0.9)
    )
    
    # Random search for tuning method.
    tune_control = makeTuneControlRandom(maxit = TUNEITERS)
    
    # Tune.
    data.lrn.tuned = tuneParams(data.lrn, 
                                task = data.task, 
                                resampling = RESAMPLING, 
                                par.set = param.set, 
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
    param.set = makeParamSet(
      makeIntegerParam("ntree", lower = 50, upper = 500),
      makeIntegerParam("mtry", lower = 1, upper = ncol(data) - 1)
    )
    
    # Random search for tuning method.
    tune_control = makeTuneControlRandom(maxit = TUNEITERS)
    
    # Tune.
    data.lrn.tuned = tuneParams(lrn, 
                                task = data.task, 
                                resampling = cv10, 
                                par.set = param.set, 
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
    lrn = makeLearner("classif.svm", 
                      predict.type = "prob")
    
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

# Return the names of all columns with different numeric values in the given
# counterfactual when compared to the other given data point.
getMutNumericFeatures <- function(cf, orig) {
  if (ncol(cf) != ncol(orig)) {
    return(c())
  }
  
  nms = c()
  for (nm in names(cf)) {
    if (is.numeric(cf[[nm]]) & cf[[nm]] != orig[[nm]]) {
      nms = append(nms, nm)
    }
  }
  return(nms)
}

# Return TRUE if the given set of counterfactuals contains a counterfactual
# with mutated numeric features, else return FALSE.
containsMutNumericFeatures <- function(cfs, orig) {
  if (nrow(cfs) <= 0) {
    return(FALSE)
  }
  
  for (i in 1:nrow(cfs)) {
    if (length(getMutNumericFeatures(cfs[i,], orig)) > 0) {
      return(TRUE)
    }
  }
  return(FALSE)
}

# Return TRUE if the given result equates to a negative class prediction,
# else return FALSE.
isNegativeClass <- function(result, target_range) {
  if (is.null(result)) {
    stop("error: Null result passed to isNegativeClass()")
  }
  lower = target_range[LOW_TARG_RANGE]
  upper = target_range[UPP_TARG_RANGE]
  return(!(result >= lower & result <= upper))
}

# Given a set of test data and a set of training data, return the test data
# such that it only contains instances that comprise a subset of the training
# data's factors.
subsetFactors <- function(test_data,
                          train_data) {
  if (!compare_df_cols_same(test_data, train_data)) {
    stop("Error: Non-identical test data & training data sets passed to subsetFactors")
  }

  train_data.uniq = setNames(lapply(train_data, unique), names(df))
  isf.idxs = which(as.logical(lapply(train_data.uniq, is.factor)))
  rem.idxs = integer()
  i = 1
  for (c in isf.idxs) {
    for (r in 1:length(test_data[,c])) {
      val = test_data[r,c]
      if (!(test_data[r,c] %in% train_data.uniq[[c]])) {
        rem.idxs[i] = r
        i = i + 1
      }
    }
  }
  rem.idxs = unique(rem.idxs)

  if (length(rem.idxs) > 0) {
    test_data = test_data[-rem.idxs,]
  }
  return(test_data)
}

# Given a data set and a predictor, return a filtered data set containing
# only usable data points for base testing. If the size of the data set is not
# sufficient, return NULL.
#
# If SUBSET_FACTORS = TRUE, ensure the test data set contains only the same
# set or a subset of the factors of the training data.
#
filterTestData <- function(test_data,
                           train_data,
                           pred, 
                           target_range,
                           size_req = nrow(test_data) %/% 2,
                           SUBSET_FACTORS = FALSE) {

  if (SUBSET_FACTORS) {
    test_data = subsetFactors(test_data, train_data)
  }

  is.neg.idxs = sapply(pred$predict(test_data), 
                       isNegativeClass, 
                       target_range = target_range)
  is.neg.idxs[is.na(is.neg.idxs)] = FALSE
  
  test_data = test_data[is.neg.idxs,]
  if (nrow(test_data) < size_req) {
    return(NULL)
  }
  return(test_data)
}