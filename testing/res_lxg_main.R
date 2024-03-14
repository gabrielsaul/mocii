############################################################
### Counterfactual Resilience & Lexicographic Selection: ###
###                                                      ###
###   AUTHOR:      Gabriel Doyle-Finch                   ###
###   SUPERVISOR:  Alex Freitas                          ###
###   INSTITUTION: University of Kent                    ###
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
DATASETS = list(adult_data)

# Full list of ML algorithm types.
ML_ALGS = list(svm_alg)

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
                TD_PADDING_MULTIPLIER = 3,
                ext.resilience = TRUE,
                best.params = readRDS("../saved_objects/best_configs.rds")) 
{
  checkmate::assert_data_frame(data)
  checkmate::assert_character(data_id)
  checkmate::assert_character(target)
  checkmate::assert_character(target_values)
  checkmate::assert_character(naf)
  checkmate::assert_character(ml_alg_id)
  checkmate::assert_numeric(ml_alg_target_range)
  checkmate::assert_list(obj.ordering)
  checkmate::assert_numeric(n_points_of_interest)
  checkmate::assert_numeric(TD_PADDING_MULTIPLIER)
  checkmate::assert_logical(ext.resilience)
  checkmate::assert_data_frame(best.params)
  
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
  
  # Select training & test data.
  set.seed(as.numeric(Sys.time()))
  test_data_idx = sample(1:nrow(data), n_points_of_interest * TD_PADDING_MULTIPLIER)
  train_data = data[-test_data_idx,]
  test_data = data[test_data_idx,]
  
  # Create predictor.
  pred = getPredictor(ml_alg_id = ml_alg_id, 
                      data = train_data,
                      data_id = data_id,
                      target = target,
                      target_values = target_values)
  
  # Prune test data.
  test_data = pruneTestData(test_data,
                            train_data,
                            pred, 
                            ml_alg_target_range,
                            size_req = n_points_of_interest,
                            SUBSET_FACTORS = TRUE)
  
  # Operate on points of interest.
  poi_tested_res = 0
  poi_tested_vs = 0
  while (poi_tested_res < n_points_of_interest | 
        poi_tested_vs < n_points_of_interest) {
    
    # Terminate early if test data is exhausted.
    if (nrow(test_data) <= 0) {
      warning("Warning: Terminated before full points of interest set was tested")
      
      par_pd_wlt[c(1, 2, 3)] = lex_pd_wlt[c(2, 1, 3)]
      par_lx_wlt[c(1, 2, 3)] = lex_lx_wlt[c(2, 1, 3)]
      return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s_%s", 
                                                                     data_id, 
                                                                     ml_alg_id,
                                                                     ifelse(ext.resilience,
                                                                            "res",
                                                                            "nores")),
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
                  generateAlgorithmTestResults(test_run_id = sprintf("lex_%s_%s_%s", 
                                                                     data_id, 
                                                                     ml_alg_id,
                                                                     ifelse(ext.resilience,
                                                                            "res",
                                                                            "nores")),
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
    writeLines(sprintf("points_of_interest_tested   %d", points_of_interest_tested))
    writeLines(sprintf("poi_tested_res              %d", poi_tested_res))
    writeLines(sprintf("poi_tested_vs               %d", poi_tested_vs))
    
    # Sample a usable test data point as a point of interest.
    set.seed(as.numeric(Sys.time()))
    x.interest.id = sample(1:nrow(test_data), 1)
    x.interest = test_data[x.interest.id,]
    x.interest = x.interest[,-which(names(x.interest) == target)]
    test_data = test_data[-x.interest.id,]
    
    # Compute counterfactuals for point of interest (Pareto-based).
    writeLines("Generating Pareto counterfactuals...")
    set.seed(as.numeric(Sys.time()))
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
    
    # Compute counterfactuals for point of interest (lexicographic).
    writeLines("Generating lexicographic counterfactuals...")
    set.seed(as.numeric(Sys.time()))
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
      
      # Comparison testing: Pareto vs. Lexicographic.
      if (poi_tested_vs < n_points_of_interest) {
        # Extract fitness data frames.
        ncols.cf = ncol(pareto.cf$results$counterfactuals)
        fit.col.idxs = (ncols.cf - NCOLS_CF_METADATA + 1):(ncols.cf - 1)
        pareto.fit = pareto.cf$results$counterfactuals[,fit.col.idxs]
        lexico.fit = lexico.cf$results$counterfactuals[,fit.col.idxs]
        
        # Compete in two arenas: Lexicographic selection and Pareto dominance.
        writeLines("Competing... (lexicographic selection)")
        lex_lx_wlt = lex_lx_wlt + competeLX(lexico.set = lexico.fit, 
                                            pareto.set = pareto.fit,
                                            obj.ordering = obj.ordering)
        writeLines("Competing... (Pareto-dominance)")
        lex_pd_wlt = lex_pd_wlt + competePD(lexico.set = lexico.fit, 
                                            pareto.set = pareto.fit)
        
        poi_tested_vs = poi_tested_vs + 1
      }
      
      # Remove invalid counterfactuals & metadata.
      valid_cf_idx = which(pareto.cf$results$counterfactuals$dist.target <= 0)
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
      if (lexico.cf$results$counterfactuals[1,]$dist.target > 0) {
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
        
        # Resilience tests.
        writeLines("Calculating Pareto counterfactuals' resilience...")
        par_res_df = getResilience(x.interest,
                                   par_valid_cfs,
                                   pred,
                                   ml_alg_target_range,
                                   pareto.cf$min_feat_values,
                                   pareto.cf$max_feat_values,
                                   pareto.cf$data_feat_types)
        writeLines("Calculating lexicographic counterfactuals' resilience...")
        lex_res_df = getResilience(x.interest,
                                   lex_valid_cfs,
                                   pred,
                                   ml_alg_target_range,
                                   lexico.cf$min_feat_values,
                                   lexico.cf$max_feat_values,
                                   lexico.cf$data_feat_types)
        
        # Expand resilience dataframes if returned. 
        if (nrow(par_res_df) > 0 & nrow(lex_res_df) > 0) {
          poi_tested_res = poi_tested_res + 1
          par_resilience_df = rbind(par_resilience_df, par_res_df)
          lex_resilience_df = rbind(lex_resilience_df, lex_res_df)
        }

        if (nrow(par_res_df) <= 0) {
          par_poi_rejected = par_poi_rejected + 1
        }
        if (nrow(lex_res_df) <= 0) {
          lex_poi_rejected = lex_poi_rejected + 1
        }
      }
    }
    writeLines("Done.")
  }
  writeLines("Generating results...")

  par_pd_wlt[c(1, 2, 3)] = lex_pd_wlt[c(2, 1, 3)]
  par_lx_wlt[c(1, 2, 3)] = lex_lx_wlt[c(2, 1, 3)]
  return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s_%s", 
                                                                 data_id, 
                                                                 ml_alg_id,
                                                                 ifelse(ext.resilience,
                                                                        "res",
                                                                        "nores")),
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
              generateAlgorithmTestResults(test_run_id = sprintf("lex_%s_%s_%s", 
                                                                 data_id, 
                                                                 ml_alg_id,
                                                                 ifelse(ext.resilience,
                                                                        "res",
                                                                        "nores")),
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

# Test across all datasets.
sink("logs/progress.txt")
for (ds in DATASETS) {
  # Prepare data.
  names(ds@df) = tolower(names(ds@df))
  ds@df = na.omit(ds@df)
  ds@df[,ds@target] = as.factor(ds@df[,ds@target])
  
  # Test across all ML algorithms.
  for (ml_alg in ML_ALGS) {
    
    # Test across two objective orderings.
    oo = 1
    for (obj.ordering in list(obj.ordering1, obj.ordering2)) {
      # Test both without resilience and with resilience.
      for (ext.resilience in list(FALSE, TRUE)) {
        rds.filename = sprintf("logs/test_%s_%s_%s_oo%d.rds", 
                               ds@id, 
                               ml_alg@id,
                               ifelse(ext.resilience, "res", "nores"),
                               oo)
        print(rds.filename)
        results = run(ds@df,
                      ds@id,
                      ds@target,
                      ds@target_values,
                      ds@nonactionable,
                      ml_alg@id,
                      ml_alg@target_range,
                      ext.resilience = ext.resilience,
                      obj.ordering = obj.ordering,
                      n_points_of_interest = 20,
                      TD_PADDING_MULTIPLIER = 3)
        saveRDS(results, file = rds.filename)
      }
      oo = oo + 1
    }
  }
}
sink()