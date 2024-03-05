source("../helpers/core_funcs.R")

# Given a point of interest & a set of counterfactual(s), return a data frame of 
# the resilience scores of each feature for each counterfactual.
#
getResilience <- function(x.interest,
                          cfs,
                          pred,
                          ml_alg_target_range,
                          max_feat_values,
                          min_feat_values,
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
    
    # Check if the counterfactual is testable: Mutated numeric features present.
    changed = getMutNumericFeatures(cf, x.interest)
    if (length(changed) > 0) {
      cf_resilience_df[1,] = rep(NaN, ncol(cf))
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
        if (data_feat_types[[k]] == "integer") {
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
          cf[[k]] = cf[[k]] + inc
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
        cf_resilience_df[[k]] = successful_steps / max_steps
        
        writeLines(sprintf("Successful Steps: %d/%d", successful_steps, max_steps))
        print("\n###########################################################\n")
      }
      else {
        # Feature is already at min/max: Assume full resilience.
        print("Already at min/max.")
        print("Full resilience.")
        cf_resilience_df[[k]] = 1.0
      }
    }
    
    # Append to main dataframe.
    resilience_df[i,] = cf_resilience_df
  }
  
  return(resilience_df)
}