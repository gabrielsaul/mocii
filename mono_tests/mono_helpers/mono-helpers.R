# Indexing constants.
POS_CLV_INDEX = 1
NEG_CLV_INDEX = 2
LOW_TARG_RANGE = 1
UPP_TARG_RANGE = 2

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
monoTestResults <- setClass(
  "monoTestResults",
  slots = c(resilience_total = "numeric",
            resilience_ft = "numeric",
            resilience_cf = "numeric",
            n_cf = "numeric",
            n_cf_invalid = "numeric",
            n_ft_mutated_total = "numeric",
            n_ft_mutated = "numeric",
            n_samples_tested = "numeric",
            n_samples_rejected = "numeric",
            n_succ_steps_ft = "numeric",
            n_max_steps_ft = "numeric")
)


# Generate a results object for a resilience test.
generateMonoTestResults <- function(feat_resilience_scores,
                                    cf_resilience_scores,
                                    cf_count,
                                    cf_inv_count,
                                    feat_mut_count,
                                    data_samples_tested,
                                    data_samples_rejected,
                                    feat_succ_steps,
                                    feat_max_steps) {
  overall_res_score = 0
  total_mut_feat = 0
  total_mut_inst = 0
  for (key in names(feat_resilience_scores)) {
    if (feat_mut_count[[key]] > 0) {
      feat_resilience_scores[[key]] = feat_resilience_scores[[key]] / feat_mut_count[[key]]
      overall_res_score = overall_res_score + feat_resilience_scores[[key]]
      total_mut_feat = total_mut_feat + 1
      total_mut_inst = total_mut_inst + feat_mut_count[[key]]
    }
    else {
      feat_resilience_scores[[key]] = NaN
    }
  }
  
  if (total_mut_feat != 0) {
    overall_res_score = overall_res_score / total_mut_feat
  } 
  else {
    overall_res_score = 1.0
  }
  
  return(monoTestResults("resilience_total" = overall_res_score,
                         "resilience_ft" = feat_resilience_scores,
                         "resilience_cf" = cf_resilience_scores,
                         "n_cf" = cf_count, 
                         "n_cf_invalid" = cf_inv_count,
                         "n_ft_mutated_total" = total_mut_inst,
                         "n_ft_mutated" = feat_mut_count,
                         "n_samples_tested" = data_samples_tested,
                         "n_samples_rejected" = data_samples_rejected,
                         "n_succ_steps_ft" = feat_succ_steps,
                         "n_max_steps_ft" = feat_max_steps))
}

# Return a hashmap initialised to the given value for the given set of column 
# names.
getHashMap <- function(nms, val = 0, type = "c") {
  
  if (type == "c") {
    hm = c()
  }
  else if (type == "l") {
    hm = list()
  }
  else {
    stop("Error: Invalid type passed to getHashMap")
  }
  
  for (nm in nms) {
    hm[nm] = val
  }
  
  return(hm)
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