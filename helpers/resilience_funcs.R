source("../helpers/core_funcs.R")

# Given a point of interest & a set of counterfactual(s), return a data frame of 
# the resilience scores of each feature for each counterfactual.
#
getResilience <- function(x.interest,
                          cfs,
                          pred,
                          ml_alg_target_range,
                          min_feat_values,
                          max_feat_values,
                          data_feat_types) {
  
  # No counterfactuals.
  if (nrow(cfs) <= 0) {
    return(NULL)
  }
  
  # Empty resilience dataframe for main resilience dataframe.
  cf_nms = names(cfs)
  resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), 
                           cf_nms)
  
  # For each counterfactual...
  for (i in 1:nrow(cfs)) {
    cf = cfs[i,]
  
    # Empty resilience dataframe for counterfactual resilience dataframe.
    cf_resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), 
                                cf_nms)
    cf_resilience_df[1,] = rep(NaN, ncol(cf))
    
    # Get mutated numeric feature names.
    changed = getMutNumericFeatures(cf, x.interest)
    
    # For each mutated numeric feature...
    for (k in changed) {
      
      # If between min/max of the feature value in observed data.
      if (cf[[k]] > min_feat_values[[k]] & cf[[k]] < max_feat_values[[k]]) {
        
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
        if (data_feat_types[[k]] == "integer") {
          inc = round(inc)
          
          # If below 1, default to 1.
          if (inc == 0) {
            inc = 1 * flip
          }
        }
        
        # Set maximum number of steps for MV search.
        max_steps = abs(limit - cf[[k]]) %/% abs(inc)
        
        # Mutate feature to search for closest MV.
        successful_steps = 0
        while (successful_steps < max_steps) {
          cf[[k]] = cf[[k]] + inc
          
          class.predicted = isNegativeClass(pred$predict(cf), ml_alg_target_range)
          if (is.na(class.predicted) | class.predicted) {
            break;
          }
          
          successful_steps = successful_steps + 1
        }
        
        # Log step ratio for feature.
        cf_resilience_df[[k]] = successful_steps / max_steps
      }
      else {
        # Feature is already at min/max: Assume full resilience.
        cf_resilience_df[[k]] = 1.0
      }
    }
    
    # Append to main dataframe.
    resilience_df[i,] = cf_resilience_df
  }
  
  return(resilience_df)
}