############################################################
### Counterfactual Resilience & Lexicographic Selection: ###
###                                                      ###
###  + Testing counterfactuals' proximity to violations  ###
###    of monotonicity constraints                       ###
###  + Adjusting the fitness function to consider        ###
###    counterfactual resilience as an extension to      ###
###    validity                                          ###
###  + Comparing the effectiveness of lexicographic      ###
###    selection to the Pareto-based approach            ###
###                                                      ###
############################################################

# Sentinel values for fitness objectives (in default order).
OBJ_VALID     = 1
OBJ_SIMILAR   = 2
OBJ_SPARSE    = 3
OBJ_PLAUSIBLE = 4

# Number of fitness objectives.
N_OBJ         = 4

# Number of metadata columns in MOC results.
NCOLS_CF_METADATA = 5

#--- Setup ----
# Set the working directory.
wdir = "C:\\Users\\Owner\\OneDrive\\Documents\\Academic\\UKC\\CS\\3\\All\\COMP6200 Research Project\\MONOMOC\\monomoc\\testing"
setwd(wdir)

# Load necessary library packages.
library("purrr")

# Load helper functions for testing.
source("../helpers/core_funcs.R")
source("../helpers/resilience_funcs.R")
source("../helpers/compete_funcs.R")

# Load counterfactuals package.
devtools::load_all("../counterfactuals", export_all = FALSE)


# Data structs.
# Adult Income Census.
adult_data = dataStruct(df = read.csv("datasets/processed/adult_data_less.csv", 
                                      stringsAsFactors = TRUE),
                        id = "adult",
                        target = "class",
                        target_values = c("GT50K", "LTE50K"),
                        nonactionable = c("age", 
                                          "education", 
                                          "marital_status", 
                                          "relationship", 
                                          "race", 
                                          "sex", 
                                          "native_country"))

# COMPAS Recidivism Racial Bias.
compas_data = dataStruct(df = read.csv("datasets/processed/compas_data_less.csv", 
                                       stringsAsFactors = TRUE),
                         id = "compas",
                         target = "two_year_recid",
                         target_values = c("no", "yes"),
                         nonactionable = c("age", "age_cat", "sex", "race"))

# Diabetes.
diabetes_data = dataStruct(df = read.csv("datasets/processed/diabetes_data.csv", 
                                         stringsAsFactors = TRUE),
                           id = "diabetes",
                           target = "outcome",
                           target_values = c("neg", "pos"),
                           nonactionable = c("age", "pregnancies"))

# HELOC (FICO 2018).
fico_data = dataStruct(df = read.csv("datasets/processed/fico_data_less.csv", 
                                     stringsAsFactors = TRUE),
                       id = "fico",
                       target = "riskperformance",
                       target_values = c("Good", "Bad"),
                       nonactionable = c("externalriskestimate"))

# German credit risk.
german_data = dataStruct(df = read.csv("datasets/processed/german_credit_data.csv", 
                                       stringsAsFactors = TRUE),
                         id = "german",
                         target = "risk",
                         target_values = c("good", "bad"),
                         nonactionable = c("age", "sex"))

# ML Algorithm Types.
# Neural network (NN).
nn_alg = mlAlgStruct(id = NN_ALG_ID,
                     target_range = c(0.5, 1.0))

# Random Forest (RF)
rf_alg = mlAlgStruct(id = RF_ALG_ID,
                     target_range = c(0.5, 1.0))

# Support Vector Machine (SVM).
svm_alg = mlAlgStruct(id = SVM_ALG_ID,
                      target_range = c(0.5, 1.0))

# Full list of datasets.
DATASETS = list(adult_data, compas_data, diabetes_data, fico_data, german_data)

# Full list of ML algorithm types.
ML_ALGS = list(nn_alg, rf_alg, svm_alg)

# Main test function for monotonicity constraint violation proximity (resilience)
# and comparisons to the lexicographic selection function. 
run <- function(data,
                data_id,
                target,
                target_values,
                naf,
                ml_alg_id,
                ml_alg_target_range,
                obj.ordering,
                n_points_of_interest = 1,
                data_feat_types = sapply(type.convert(data, as.is = TRUE), class),
                TD_PADDING_MULTIPLIER = 5,
                ext.resilience = FALSE,
                best.params = readRDS("../saved_objects/best_configs.rds")) 
{
  
  # Check parameters.
  if ((n_points_of_interest != round(n_points_of_interest))|
      (TD_PADDING_MULTIPLIER != round(TD_PADDING_MULTIPLIER))) {
    stop("Error: n_points_of_interest & TD_PADDING_MULTIPLIER must be integers")
  }
  if (n_points_of_interest < 1) {
    warning("Warning: Setting n_points_of_interest to at least 1")
    n_points_of_interest = 1
  }
  if (TD_PADDING_MULTIPLIER < 1) {
    warning("Warning: Setting TD_PADDING_MULTIPLIER to at least 1")
    TD_PADDING_MULTIPLIER = 1
  }
  if ((n_points_of_interest * TD_PADDING_MULTIPLIER) > (nrow(data) %/% 2)) {
    stop(sprintf("Error: n_points_of_interest can be no greater than %d%% of nrow(data)", 
                  50 / TD_PADDING_MULTIPLIER))
  }
  
  # Count of usable data samples tested on any MOC runs.
  points_of_interest_tested = 0
  
  # Algorithm metrics for Pareto-based GA (number of generations, runtime).
  par_gen_total = 0
  par_runtime_total = 0
  
  # Count of rejected points of interest for Pareto CF tests.
  par_poi_rejected = 0
  
  # Aggregated data frame of Pareto counterfactuals.
  par_cf_df = NULL
  
  # Count of a full set of invalid Pareto counterfactuals being returned.
  par_cf_all_invalid = 0
  
  # Count of a null set of Pareto counterfactuals being returned.
  par_cf_null_returned = 0
  
  # Pareto dominance WLT record for Pareto counterfactuals.
  par_pd_wlt = c(0, 0, 0)
  
  # Lexicographic WLT record for Pareto counterfactuals.
  par_lx_wlt = c(0, 0, 0)
  
  # Resilience data frame for Pareto counterfactuals.
  cf_nms = colnames(data[,-which(names(data) == target)])
  par_resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), cf_nms)
  
  # Algorithm metrics for lexicographic GA (number of generations, runtime).
  lex_gen_total = 0
  lex_runtime_total = 0
  
  # Count of rejected points of interest for lexicographic CF tests.
  lex_poi_rejected = 0
  
  # Aggregated data frame of lexicographic counterfactuals.
  lex_cf_df = NULL
  
  # Count of a full set of invalid lexicographic counterfactuals being returned.
  lex_cf_all_invalid = 0
  
  # Count of a null set of lexicographic counterfactuals being returned.
  lex_cf_null_returned = 0
  
  # Pareto dominance WLT record for lexicographic counterfactuals.
  lex_pd_wlt = c(0, 0, 0)
  
  # Lexicographic WLT record for lexicographic counterfactuals.
  lex_lx_wlt = c(0, 0, 0)
  
  # Resilience data frame for lexicographic counterfactuals.
  cf_nms = colnames(data[,-which(names(data) == target)])
  lex_resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), cf_nms)

  
  # Minimum & maximum values for each numeric feature in the test data.
  min_feat_values = apply(data, 2, min)
  min_feat_values = map_if(min_feat_values, isNumericString, as.numeric)
  max_feat_values = apply(data, 2, max)
  max_feat_values = map_if(max_feat_values, isNumericString, as.numeric)
  
  print("\nMIN FEAT VALUES:")
  print(min_feat_values)
  print("\nMAX FEAT VALUES:")
  print(max_feat_values)
  
  # Select training & test data.
  set.seed(as.numeric(Sys.time()))
  test_data_idx = sample(1:nrow(data), n_points_of_interest * TD_PADDING_MULTIPLIER)
  train_data = data[-test_data_idx,]
  test_data = data[test_data_idx,]
  
  # Remaining test data idx.
  test_data_idx = 1:nrow(data)[-test_data_idx]
  
  # Create predictor.
  pred = getPredictor(ml_alg_id = ml_alg_id, 
                      data = train_data,
                      data_id = data_id,
                      target = target,
                      target_values = target_values)
  
  # Operate on points of interest.
  poi_tested_res = 0
  poi_tested_vs = 0
  while (poi_tested_res < n_points_of_interest | 
         poi_tested_vs < n_points_of_interest) {

    # Randomly sample a usable data point.
    is_usable = FALSE
    while (!is_usable) {

      # No more usable points of interest in current test data segment.
      if (nrow(test_data) == 0) {
        
        # Terminate early, or...
        if (length(test_data_idx) == 0) {
          warning("Warning: Terminated before full points of interest set was tested")
          
          par_pd_wlt[c(1, 2)] = lex_pd_wlt[c(2, 1)]
          par_lx_wlt[c(1, 2)] = lex_lx_wlt[c(2, 1)]
          return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s", data_id, ml_alg_id),
                                                   par_runtime_total,
                                                   par_gen_total,
                                                   par_runtime_total,
                                                   points_of_interest_tested,
                                                   poi_tested_res,
                                                   poi_tested_vs,
                                                   par_poi_rejected,
                                                   par_cf_df,
                                                   par_cf_all_invalid,
                                                   par_cf_null_returned,
                                                   par_pd_wlt,
                                                   par_lx_wlt,
                                                   par_resilience_df),
                      generateAlgorithmTestResults(test_run_id = sprintf("lex_%s_%s", data_id, ml_alg_id),
                                                   lex_runtime_total,
                                                   lex_gen_total,
                                                   lex_runtime_total,
                                                   points_of_interest_tested,
                                                   poi_tested_res,
                                                   poi_tested_vs,
                                                   lex_poi_rejected,
                                                   lex_cf_df,
                                                   lex_cf_all_invalid,
                                                   lex_cf_null_returned,
                                                   lex_pd_wlt,
                                                   lex_lx_wlt,
                                                   lex_resilience_df)))
        }
        
        print("Retraining...")
        
        # Set new test data segment size.
        td_segment_size = n_points_of_interest * TD_PADDING_MULTIPLIER
        if (length(test_data_idx) < td_segment_size) {
          td_segment_size = length(test_data_idx)
        }
        
        # Re-segment data into training & test.
        set.seed(as.numeric(Sys.time()))
        new_test_data_idx = sample(length(test_data_idx), td_segment_size)
        test_data = data[test_data_idx[new_test_data_idx],]
        train_data = data[-test_data_idx[new_test_data_idx],]
        
        # Remove new test data idx from remaining test data idx.
        test_data_idx = test_data_idx[-new_test_data_idx]
        
        # Train new predictor.
        pred = getPredictor(ml_alg_id,
                            train_data,
                            data_id,
                            target,
                            target_values)
      }
      
      # Select point of interest.
      set.seed(as.numeric(Sys.time()))
      sample_i = sample(1:nrow(test_data), 1)
      x.interest = test_data[sample_i,]
      
      # Remove point of interest from test data.
      test_data = test_data[-sample_i,]
      
      # Verify that the point of interest is predicted as negative class.
      print(x.interest)
      result = pred$predict(x.interest)
      print(result)
      
      if (isNegativeClass(result, ml_alg_target_range)) {
        is_usable = TRUE
      }
      else {
        # Predicted positive class: Data point unusable.
        par_poi_rejected = par_poi_rejected + 1
        lex_poi_rejected = lex_poi_rejected + 1
      }
    }
    
    print("\nPoint of Interest: ")
    print(x.interest)
    x.interest = x.interest[,-which(names(x.interest) == target)]
    
    # Compute counterfactuals for data point of interest (Pareto-based).
    set.seed(1000)
    system.time({pareto.cf = Counterfactuals$new(predictor = pred, 
                                               x.interest = x.interest,
                                               target = ml_alg_target_range, 
                                               epsilon = 0, 
                                               generations = list(mosmafs::mosmafsTermStagnationHV(10),
                                                                  mosmafs::mosmafsTermGenerations(200)), 
                                               mu = best.params$mu, 
                                               p.mut = best.params$p.mut, 
                                               p.rec = best.params$p.rec, 
                                               p.mut.gen = best.params$p.mut.gen, 
                                               p.mut.use.orig = best.params$p.mut.use.orig, 
                                               p.rec.gen = best.params$p.rec.gen, 
                                               initialization = "icecurve",
                                               p.rec.use.orig = best.params$p.rec.use.orig,
                                               fixed.features = naf,
                                               lexicographic.selection = FALSE,
                                               ext.resilience = ext.resilience)})
    
    # Log Pareto-based GA metrics.
    par_gen_total = par_gen_total + pareto.cf$n.generations
    par_runtime_total = par_runtime_total + pareto.cf$runtime
    
    system.time({lexico.cf = Counterfactuals$new(predictor = pred, 
                                                 x.interest = x.interest,
                                                 target = ml_alg_target_range, 
                                                 epsilon = 0, 
                                                 generations = pareto.cf$n.generations, 
                                                 mu = best.params$mu, 
                                                 p.mut = best.params$p.mut, 
                                                 p.rec = best.params$p.rec, 
                                                 p.mut.gen = best.params$p.mut.gen, 
                                                 p.mut.use.orig = best.params$p.mut.use.orig, 
                                                 p.rec.gen = best.params$p.rec.gen, 
                                                 initialization = "icecurve",
                                                 p.rec.use.orig = best.params$p.rec.use.orig,
                                                 fixed.features = naf,
                                                 lexicographic.selection = TRUE,
                                                 obj.ordering = obj.ordering,
                                                 ext.resilience = ext.resilience)})
    
    # Log lexicographic algorithm metrics.
    lex_gen_total = lex_gen_total + lexico.cf$n.generations
    lex_runtime_total = lex_runtime_total + lexico.cf$runtime
    
    # Increment points of interest tested on any MOC runs.
    points_of_interest_tested = points_of_interest_tested + 1
    
    # No counterfactuals returned.
    if (is.null(pareto.cf) | is.null(lexico.cf)) {
      if (is.null(pareto.cf)) {
        par_poi_rejected = par_poi_rejected + 1
        par_cf_null_returned = par_cf_null_returned + 1
      }
      
      if (is.null(lexico.cf)) {
        lex_poi_rejected = lex_poi_rejected + 1
        lex_cf_null_returned = lex_cf_null_returned + 1
      }
      print("###############################")
      print("# No counterfactuals returned #")
      print("###############################")
    }
    else {
      
      # Store Pareto counterfactuals.
      if (!is.null(pareto.cf) & is.null(par_cf_df)) {
        par_cf_nms = colnames(pareto.cf$results$counterfactuals)
        par_cf_df = setNames(data.frame(matrix(ncol = length(par_cf_nms), nrow = 0)), par_cf_nms)
      }
      par_cf_df = rbind(par_cf_df, pareto.cf$results$counterfactuals)
      
      # Store lexicographic counterfactuals.
      if (!is.null(lexico.cf) & is.null(lex_cf_df)) {
        lex_cf_nms = colnames(lexico.cf$results$counterfactuals)
        lex_cf_df = setNames(data.frame(matrix(ncol = length(lex_cf_nms), nrow = 0)), lex_cf_nms)
      }
      lex_cf_df = rbind(lex_cf_df, lexico.cf$results$counterfactuals)
      
      print("Counterfactuals (Pareto): ")
      print(pareto.cf$results$counterfactuals)
      
      print("\n\nCounterfactuals (Lexicographic): ")
      print(lexico.cf$results$counterfactuals)
      
      # Comparison testing: Pareto vs. Lexicographic.
      if (poi_tested_vs < n_points_of_interest) {
        # Extract fitness data frames.
        ncols.cf = ncol(pareto.cf$results$counterfactuals)
        fit.col.idxs = (ncols.cf - NCOLS_CF_METADATA + 1):(ncols.cf - 1)
        pareto.fit = pareto.cf$results$counterfactuals[,fit.col.idxs]
        lexico.fit = lexico.cf$results$counterfactuals[,fit.col.idxs]
        
        # Compete in two arenas: Lexicographic selection and Pareto dominance.
        lex_lx_wlt = lex_lx_wlt + competeLX(lexico.set = lexico.fit, 
                                            pareto.set = pareto.fit,
                                            obj.ordering = obj.ordering)
        lex_pd_wlt = lex_pd_wlt + competePD(lexico.set = lexico.fit, 
                                            pareto.set = pareto.fit)
        
        poi_tested_vs = poi_tested_vs + 1
      }
      
      # Remove invalid counterfactuals & metadata.
      valid_cf_idx = which(pareto.cf$results$counterfactuals$dist.target == 0)
      par_valid_cfs = pareto.cf$results$counterfactuals[valid_cf_idx,
                                                        1:(ncols.cf - NCOLS_CF_METADATA)]
      pareto.cf$results$counterfactuals.diff = pareto.cf$results$counterfactuals.diff[valid_cf_idx, ]
      
      # Get relative frequency of feature changes.
      par_rel_freq = pareto.cf$get_frequency()
      lex_rel_freq = lexico.cf$get_frequency()
      
      # Check for at least one valid counterfactual in both sets.
      is_testable = TRUE
      if (nrow(par_valid_cfs) <= 0) {
        is_testable = FALSE
        par_poi_rejected = par_poi_rejected + 1
        par_cf_all_invalid = par_cf_all_invalid + 1
      } 
      if (lexico.cf$results$counterfactuals[1,]$dist.target != 0) {
        is_testable = FALSE
        lex_poi_rejected = lex_poi_rejected + 1
        lex_cf_all_invalid = lex_cf_all_invalid + 1
      }
      lex_valid_cfs = lexico.cf$results$counterfactuals[1,1:(ncols.cf - NCOLS_CF_METADATA)]
      
      # Check for at least one counterfactual with mutated numerical values in
      # both sets.
      if (is_testable & (!containsMutNumericFeatures(par_valid_cfs, x.interest) | 
                         !containsMutNumericFeatures(lex_valid_cfs, x.interest))) {
        is_testable = FALSE
      }
      
      # Both sets of counterfactuals are testable. 
      if (is_testable & (poi_tested_res < n_points_of_interest)) {
        
        # Resilience tests: Expand resilience dataframes.
        par_resilience_df = rbind(par_resilience_df, getResilience(x.interest,
                                                                   par_valid_cfs,
                                                                   pred,
                                                                   ml_alg_target_range,
                                                                   max_feat_values,
                                                                   min_feat_values,
                                                                   data_feat_types))
        lex_resilience_df = rbind(lex_resilience_df, getResilience(x.interest,
                                                                   lex_valid_cfs,
                                                                   pred,
                                                                   ml_alg_target_range,
                                                                   max_feat_values,
                                                                   min_feat_values,
                                                                   data_feat_types))
        poi_tested_res = poi_tested_res + 1
      }
    }
  }

  par_pd_wlt[c(1, 2, 3)] = lex_pd_wlt[c(2, 1, 3)]
  par_lx_wlt[c(1, 2, 3)] = lex_lx_wlt[c(2, 1, 3)]
  return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s", data_id, ml_alg_id),
                                           par_runtime_total,
                                           par_gen_total,
                                           points_of_interest_tested,
                                           poi_tested_res,
                                           poi_tested_vs,
                                           par_poi_rejected,
                                           par_cf_df,
                                           par_cf_all_invalid,
                                           par_cf_null_returned,
                                           par_pd_wlt,
                                           par_lx_wlt,
                                           par_resilience_df),
              generateAlgorithmTestResults(test_run_id = sprintf("lex_%s_%s", data_id, ml_alg_id),
                                           lex_runtime_total,
                                           lex_gen_total,
                                           points_of_interest_tested,
                                           poi_tested_res,
                                           poi_tested_vs,
                                           lex_poi_rejected,
                                           lex_cf_df,
                                           lex_cf_all_invalid,
                                           lex_cf_null_returned,
                                           lex_pd_wlt,
                                           lex_lx_wlt,
                                           lex_resilience_df)))
}

# Objective ordering for tests.
obj.ordering1 = list(OBJ_VALID, OBJ_SIMILAR, OBJ_SPARSE, OBJ_PLAUSIBLE)
obj.ordering2 = list(OBJ_VALID, OBJ_SPARSE, OBJ_SIMILAR, OBJ_PLAUSIBLE)

# Test datasets x ML alg types.
for (ds in DATASETS) {
  # Prepare data.
  names(ds@df) = tolower(names(ds@df))
  ds@df = na.omit(ds@df)
  ds@df[,ds@target] = as.factor(ds@df[,ds@target])
  
  for (ml_alg in ML_ALGS) {
    sink(sprintf("logs/resilience_tests_%s_%s.txt", ds@id, ml_alg@id))
    print(nrow(ds@df))
    start_time = Sys.time()
    results = run(ds@df,
                  ds@id,
                  ds@target,
                  ds@target_values,
                  ds@nonactionable,
                  ml_alg@id,
                  ml_alg@target_range,
                  obj.ordering = obj.ordering1,
                  n_points_of_interest = 1,
                  TD_PADDING_MULTIPLIER = 5)
    end_time = Sys.time()
    print("\nRESULTS: ")
    print(results)
    writeLines(sprintf("Execution Time: %f", (end_time - start_time)))
    sink()
  }
}