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
# setwd(wdir)

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
adult_data = dataStruct(df = read.csv("datasets/processed/adult_data.csv", 
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
compas_data = dataStruct(df = read.csv("datasets/processed/compas_data.csv", 
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
fico_data = dataStruct(df = read.csv("datasets/processed/fico_data.csv", 
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
ML_ALGS = list(nn_alg, rf_alg, svm_alg)

# Objective orderings for tests.
obj.ordering1 = list(OBJ_VALID, OBJ_SIMILAR, OBJ_SPARSE, OBJ_PLAUSIBLE)
obj.ordering2 = list(OBJ_VALID, OBJ_SPARSE, OBJ_SIMILAR, OBJ_PLAUSIBLE)

# Main test function for monotonicity constraint violation proximity (resilience)
# and comparisons to the lexicographic selection function. 
run <- function(data,
                data_id,
                target,
                target_values,
                naf,
                ml_alg_id,
                ml_alg_target_range,
                obj.ordering1,
                obj.ordering2,
                test_data = NULL,
                poi_idxs = NULL,
                predictor = NULL,
                n_points_of_interest = 1,
                TD_PADDING_MULTIPLIER = 3,
                TD_PADDING_ADDEND = 1,
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
  checkmate::assert_list(obj.ordering1)
  checkmate::assert_numeric(n_points_of_interest)
  checkmate::assert_numeric(TD_PADDING_MULTIPLIER)
  checkmate::assert_logical(ext.resilience)
  checkmate::assert_data_frame(best.params)
  
  # Check parameters.
  if ((n_points_of_interest != round(n_points_of_interest))) {
    stop("Error: n_points_of_interest must be an integer")
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
  lex1_gen_total = 0
  lex1_runtime_total = 0
  lex2_gen_total = 0
  lex2_runtime_total = 0
  
  # Count of rejected points of interest for lexicographic CF tests.
  lex1_poi_rejected = 0
  lex2_poi_rejected = 0
  
  # Aggregated data frame of lexicographic counterfactuals.
  lex1_cf_df = NULL
  lex2_cf_df = NULL
  
  # Count of a full set of invalid lexicographic counterfactuals being returned.
  lex1_cf_all_invalid = 0
  lex2_cf_all_invalid = 0
  
  # Count of a null set of lexicographic counterfactuals being returned.
  lex1_cf_null_returned = 0
  lex2_cf_null_returned = 0
  
  # Pareto dominance WLT record for lexicographic counterfactuals.
  lex1_pd_wlt = c(0, 0, 0)
  lex2_pd_wlt = c(0, 0, 0)
  
  # Lexicographic WLT record for lexicographic counterfactuals.
  lex1_lx_wlt = c(0, 0, 0)
  lex2_lx_wlt = c(0, 0, 0)
  
  # Resilience data frame for lexicographic counterfactuals.
  lex1_resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), cf_nms)
  lex2_resilience_df = setNames(data.frame(matrix(ncol = length(cf_nms), nrow = 0)), cf_nms)
  
  # Select training & test data.
  mult = TD_PADDING_MULTIPLIER
  while(is.null(predictor) | is.null(test_data)) {
    
    # Check if padding multiplier has grown too large.
    if ((n_points_of_interest * mult) > (nrow(data) %/% 2)) {
      stop("Error: Padding multiplier for test data is too large")
    }
    
    # Sample test data set.
    set.seed(as.numeric(Sys.time()))
    test_data_idx = sample(1:nrow(data), as.integer(round(n_points_of_interest * mult)))
    train_data = data[-test_data_idx,]
    test_data = data[test_data_idx,]
    
    # Increment padding multiplier.
    mult = mult + 0.025
    
    # Create predictor.
    writeLines("Training new predictor...")
    predictor = getPredictor(ml_alg_id = ml_alg_id, 
                        data = train_data,
                        data_id = data_id,
                        target = target,
                        target_values = target_values)
    
    # Filter test data.
    writeLines("Filtering test data...")
    test_data = filterTestData(test_data,
                               train_data,
                               predictor, 
                               ml_alg_target_range,
                               size_req = n_points_of_interest + TD_PADDING_ADDEND,
                               SUBSET_FACTORS = TRUE)
  }
  test_data_size = nrow(test_data)
  test_data_full = test_data
  writeLines(sprintf("Test data size: %d", test_data_size))
  
  # Vectors of test point indices.
  poi_tested_idxs = integer(0L)
  poi_vs_idxs = integer(0L)
  poi_res_idxs = integer(0L)
  
  # Operate on points of interest.
  poi_tested_res = 0
  poi_tested_vs = 0
  while (poi_tested_res < n_points_of_interest | 
         poi_tested_vs < n_points_of_interest) {
    
    # Output progress data.
    writeLines(sprintf("Time:                       %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")))
    writeLines(sprintf("Test data points remaining: %d", test_data_size - points_of_interest_tested))
    writeLines(sprintf("points_of_interest_tested   %d", points_of_interest_tested))
    writeLines(sprintf("poi_tested_vs               %d", poi_tested_vs))
    writeLines(sprintf("poi_tested_res              %d", poi_tested_res))
    
    # Terminate early if test data is exhausted.
    if (nrow(test_data) <= 0) {
      warning("Warning: Terminated before full points of interest set was tested")
      
      par_pd_wlt[c(1, 2, 3)] = lex1_pd_wlt[c(2, 1, 3)] + lex1_pd_wlt[c(2, 1, 3)]
      par_lx_wlt[c(1, 2, 3)] = lex1_lx_wlt[c(2, 1, 3)] + lex2_lx_wlt[c(2, 1, 3)]
      return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s_%s", 
                                                                     data_id, 
                                                                     ml_alg_id,
                                                                     ifelse(ext.resilience,
                                                                            "res",
                                                                            "nores")),
                                               runtime_total = par_runtime_total,
                                               generations_total = par_gen_total,
                                               predictor = predictor,
                                               test_df = test_data_full,
                                               poi_tested_idxs = poi_tested_idxs,
                                               poi_vs_idxs = poi_vs_idxs,
                                               poi_res_idxs = poi_res_idxs,
                                               poi_rejected_count = par_poi_rejected,
                                               cf_df = par_cf_df,
                                               cf_all_invalid = par_cf_all_invalid,
                                               cf_null_returned = par_cf_null_returned,
                                               pd_wlt = par_pd_wlt,
                                               lx_wlt = par_lx_wlt,
                                               resilience_df = par_resilience_df),
                  generateAlgorithmTestResults(test_run_id = sprintf("lex1_%s_%s_%s", 
                                                                     data_id, 
                                                                     ml_alg_id,
                                                                     ifelse(ext.resilience,
                                                                            "res",
                                                                            "nores")),
                                               runtime_total = lex1_runtime_total,
                                               generations_total = lex1_gen_total,
                                               predictor = predictor,
                                               test_df = test_data_full,
                                               poi_tested_idxs = poi_tested_idxs,
                                               poi_vs_idxs = poi_vs_idxs,
                                               poi_res_idxs = poi_res_idxs,
                                               poi_rejected_count = lex1_poi_rejected,
                                               cf_df = lex1_cf_df,
                                               cf_all_invalid = lex1_cf_all_invalid,
                                               cf_null_returned = lex1_cf_null_returned,
                                               pd_wlt = lex1_pd_wlt,
                                               lx_wlt = lex1_lx_wlt,
                                               resilience_df = lex1_resilience_df),
                  generateAlgorithmTestResults(test_run_id = sprintf("lex2_%s_%s_%s", 
                                                                     data_id, 
                                                                     ml_alg_id,
                                                                     ifelse(ext.resilience,
                                                                            "res",
                                                                            "nores")),
                                               runtime_total = lex2_runtime_total,
                                               generations_total = lex2_gen_total,
                                               predictor = predictor,
                                               test_df = test_data_full,
                                               poi_tested_idxs = poi_tested_idxs,
                                               poi_vs_idxs = poi_vs_idxs,
                                               poi_res_idxs = poi_res_idxs,
                                               poi_rejected_count = lex2_poi_rejected,
                                               cf_df = lex2_cf_df,
                                               cf_all_invalid = lex2_cf_all_invalid,
                                               cf_null_returned = lex2_cf_null_returned,
                                               pd_wlt = lex2_pd_wlt,
                                               lx_wlt = lex2_lx_wlt,
                                               resilience_df = lex2_resilience_df)))
    }

    # Sample a usable test data point as a point of interest.
    x.interest.idx = integer(0L)
    if (!is.null(poi_idxs) & points_of_interest_tested < length(poi_idxs)) {
      x.interest.idx = poi_idxs[points_of_interest_tested + 1]
    }
    else {
      set.seed(as.numeric(Sys.time()))
      x.interest.idx = sample(1:nrow(test_data), 1)
    }
    x.interest = test_data[x.interest.idx,]
    x.interest = x.interest[,-which(names(x.interest) == target)]
    test_data = test_data[-x.interest.idx,]
    
    # Compute counterfactuals for point of interest (Pareto-based).
    writeLines("Generating Pareto counterfactuals...")
    set.seed(as.numeric(Sys.time()))
    system.time({pareto.cf = Counterfactuals$new(predictor = predictor, 
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
    
    # Compute counterfactuals for point of interest (lexicographic oo1).
    writeLines("Generating lexicographic counterfactuals (oo1)...")
    set.seed(as.numeric(Sys.time()))
    system.time({lexico1.cf = Counterfactuals$new(predictor = predictor, 
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
                                                 obj.ordering = obj.ordering1,
                                                 ext.resilience = ext.resilience)})
    
    # Log lexicographic algorithm metrics.
    lex1_gen_total = lex1_gen_total + lexico1.cf$n.generations
    lex1_runtime_total = lex1_runtime_total + lexico1.cf$runtime
    
    # Compute counterfactuals for point of interest (lexicographic oo1).
    writeLines("Generating lexicographic counterfactuals (oo2)...")
    set.seed(as.numeric(Sys.time()))
    system.time({lexico2.cf = Counterfactuals$new(predictor = predictor, 
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
                                                  obj.ordering = obj.ordering2,
                                                  ext.resilience = ext.resilience)})
    
    # Log lexicographic algorithm metrics.
    lex2_gen_total = lex2_gen_total + lexico2.cf$n.generations
    lex2_runtime_total = lex2_runtime_total + lexico2.cf$runtime

    
    # Increment points of interest tested on any MOC runs.
    points_of_interest_tested = points_of_interest_tested + 1
    poi_tested_idxs[points_of_interest_tested] = x.interest.idx
    
    # No counterfactuals returned.
    if (is.null(pareto.cf) | is.null(lexico1.cf) | is.null(lexico2.cf)) {
      if (is.null(pareto.cf)) {
        par_poi_rejected = par_poi_rejected + 1
        par_cf_null_returned = par_cf_null_returned + 1
      }
      
      if (is.null(lexico1.cf)) {
        lex1_poi_rejected = lex1_poi_rejected + 1
        lex1_cf_null_returned = lex1_cf_null_returned + 1
      }
      
      if (is.null(lexico2.cf)) {
        lex2_poi_rejected = lex2_poi_rejected + 1
        lex2_cf_null_returned = lex2_cf_null_returned + 1
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
      if (!is.null(lexico1.cf) & is.null(lex1_cf_df)) {
        lex1_cf_nms = colnames(lexico1.cf$results$counterfactuals)
        lex1_cf_df = setNames(data.frame(matrix(ncol = length(lex1_cf_nms), nrow = 0)), lex1_cf_nms)
      }
      if (!is.null(lexico2.cf) & is.null(lex2_cf_df)) {
        lex2_cf_nms = colnames(lexico2.cf$results$counterfactuals)
        lex2_cf_df = setNames(data.frame(matrix(ncol = length(lex2_cf_nms), nrow = 0)), lex2_cf_nms)
      }
      lex1_cf_df = rbind(lex1_cf_df, lexico1.cf$results$counterfactuals)
      lex2_cf_df = rbind(lex2_cf_df, lexico2.cf$results$counterfactuals)
      
      # Comparison testing: Pareto vs. Lexicographic.
      if (poi_tested_vs < n_points_of_interest) {
        # Extract fitness data frames.
        ncols.cf = ncol(pareto.cf$results$counterfactuals)
        fit.col.idxs = (ncols.cf - NCOLS_CF_METADATA + 1):(ncols.cf - 1)
        pareto.fit = pareto.cf$results$counterfactuals[,fit.col.idxs]
        lexico1.fit = lexico1.cf$results$counterfactuals[,fit.col.idxs]
        lexico2.fit = lexico2.cf$results$counterfactuals[,fit.col.idxs]
        
        # Compete in two arenas: Lexicographic selection and Pareto dominance.
        writeLines("Competing... (lexicographic selection)")
        lex1_lx_wlt = lex1_lx_wlt + competeLX(lexico.set = lexico1.fit, 
                                              pareto.set = pareto.fit,
                                              obj.ordering = obj.ordering1)
        lex2_lx_wlt = lex2_lx_wlt + competeLX(lexico.set = lexico2.fit, 
                                              pareto.set = pareto.fit,
                                              obj.ordering = obj.ordering2)
        writeLines("Competing... (Pareto-dominance)")
        lex1_pd_wlt = lex1_pd_wlt + competePD(lexico.set = lexico1.fit, 
                                              pareto.set = pareto.fit)
        lex2_pd_wlt = lex2_pd_wlt + competePD(lexico.set = lexico2.fit, 
                                              pareto.set = pareto.fit)
        
        poi_tested_vs = poi_tested_vs + 1
        poi_vs_idxs[length(poi_vs_idxs) + 1] = x.interest.idx
      }
      
      # Remove invalid counterfactuals & metadata.
      valid_cf_idx = which(pareto.cf$results$counterfactuals$dist.target <= 0)
      par_valid_cfs = pareto.cf$results$counterfactuals[valid_cf_idx,
                                                        1:(ncols.cf - NCOLS_CF_METADATA)]
      pareto.cf$results$counterfactuals.diff = pareto.cf$results$counterfactuals.diff[valid_cf_idx, ]
      
      # Check for at least one valid counterfactual in all sets.
      is_testable = TRUE
      if (nrow(par_valid_cfs) == 0) {
        is_testable = FALSE
        par_poi_rejected = par_poi_rejected + 1
        par_cf_all_invalid = par_cf_all_invalid + 1
      } 
      if (lexico1.cf$results$counterfactuals[1,]$dist.target > 0) {
        is_testable = FALSE
        lex1_poi_rejected = lex1_poi_rejected + 1
        lex1_cf_all_invalid = lex1_cf_all_invalid + 1
      }
      if (lexico2.cf$results$counterfactuals[1,]$dist.target > 0) {
        is_testable = FALSE
        lex2_poi_rejected = lex2_poi_rejected + 1
        lex2_cf_all_invalid = lex2_cf_all_invalid + 1
      }
      lex1_valid_cfs = lexico1.cf$results$counterfactuals[1,1:(ncols.cf - NCOLS_CF_METADATA)]
      lex2_valid_cfs = lexico2.cf$results$counterfactuals[1,1:(ncols.cf - NCOLS_CF_METADATA)]
      
      # Check for at least one counterfactual with mutated numerical values in
      # all sets.
      if (is_testable & (!containsMutNumericFeatures(par_valid_cfs, x.interest) | 
                         !containsMutNumericFeatures(lex1_valid_cfs, x.interest))) {
        is_testable = FALSE
      }
      
      # All sets of counterfactuals are resilience-testable. 
      if (is_testable & (poi_tested_res < n_points_of_interest)) {
        
        # Resilience tests.
        writeLines("Calculating Pareto counterfactuals' resilience...")
        par_res_df = getResilience(x.interest,
                                   par_valid_cfs,
                                   predictor,
                                   ml_alg_target_range,
                                   pareto.cf$min_feat_values,
                                   pareto.cf$max_feat_values,
                                   pareto.cf$data_feat_types)
        writeLines("Calculating lexicographic (oo1) counterfactuals' resilience...")
        lex1_res_df = getResilience(x.interest,
                                   lex1_valid_cfs,
                                   predictor,
                                   ml_alg_target_range,
                                   lexico1.cf$min_feat_values,
                                   lexico1.cf$max_feat_values,
                                   lexico1.cf$data_feat_types)
        writeLines("Calculating lexicographic (oo2) counterfactuals' resilience...")
        lex2_res_df = getResilience(x.interest,
                                    lex2_valid_cfs,
                                    predictor,
                                    ml_alg_target_range,
                                    lexico2.cf$min_feat_values,
                                    lexico2.cf$max_feat_values,
                                    lexico2.cf$data_feat_types)
        
        # Expand resilience dataframes if returned. 
        if (nrow(par_res_df) > 0 & nrow(lex1_res_df) > 0 & nrow(lex2_res_df) > 0) {
          poi_tested_res = poi_tested_res + 1
          poi_res_idxs[length(poi_res_idxs) + 1] = x.interest.idx
          par_resilience_df = rbind(par_resilience_df, par_res_df)
          lex1_resilience_df = rbind(lex1_resilience_df, lex1_res_df)
          lex2_resilience_df = rbind(lex2_resilience_df, lex2_res_df)
        }

        if (nrow(par_res_df) == 0) {
          par_poi_rejected = par_poi_rejected + 1
        }
        if (nrow(lex1_res_df) == 0) {
          lex1_poi_rejected = lex1_poi_rejected + 1
        }
        if (nrow(lex2_res_df) == 0) {
          lex2_poi_rejected = lex2_poi_rejected + 1
        }
      }
    }
    writeLines("Done.")
  }
  writeLines(sprintf("Time:                       %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")))
  writeLines(sprintf("Test data points remaining: %d", test_data_size - points_of_interest_tested))
  writeLines(sprintf("points_of_interest_tested   %d", points_of_interest_tested))
  writeLines(sprintf("poi_tested_vs               %d", poi_tested_vs))
  writeLines(sprintf("poi_tested_res              %d", poi_tested_res))
  writeLines("Generating results...")

  par_pd_wlt[c(1, 2, 3)] = lex1_pd_wlt[c(2, 1, 3)] + lex2_pd_wlt[c(2, 1, 3)]
  par_lx_wlt[c(1, 2, 3)] = lex1_lx_wlt[c(2, 1, 3)] + lex2_lx_wlt[c(2, 1, 3)]
  return(list(generateAlgorithmTestResults(test_run_id = sprintf("par_%s_%s_%s", 
                                                                 data_id, 
                                                                 ml_alg_id,
                                                                 ifelse(ext.resilience,
                                                                        "res",
                                                                        "nores")),
                                           runtime_total = par_runtime_total,
                                           generations_total = par_gen_total,
                                           predictor = predictor,
                                           test_df = test_data_full,
                                           poi_tested_idxs = poi_tested_idxs,
                                           poi_vs_idxs = poi_vs_idxs,
                                           poi_res_idxs = poi_res_idxs,
                                           poi_rejected_count = par_poi_rejected,
                                           cf_df = par_cf_df,
                                           cf_all_invalid = par_cf_all_invalid,
                                           cf_null_returned = par_cf_null_returned,
                                           pd_wlt = par_pd_wlt,
                                           lx_wlt = par_lx_wlt,
                                           resilience_df = par_resilience_df),
              generateAlgorithmTestResults(test_run_id = sprintf("lex1_%s_%s_%s", 
                                                                 data_id, 
                                                                 ml_alg_id,
                                                                 ifelse(ext.resilience,
                                                                        "res",
                                                                        "nores")),
                                           runtime_total = lex1_runtime_total,
                                           generations_total = lex1_gen_total,
                                           predictor = predictor,
                                           test_df = test_data_full,
                                           poi_tested_idxs = poi_tested_idxs,
                                           poi_vs_idxs = poi_vs_idxs,
                                           poi_res_idxs = poi_res_idxs,
                                           poi_rejected_count = lex1_poi_rejected,
                                           cf_df = lex1_cf_df,
                                           cf_all_invalid = lex1_cf_all_invalid,
                                           cf_null_returned = lex1_cf_null_returned,
                                           pd_wlt = lex1_pd_wlt,
                                           lx_wlt = lex1_lx_wlt,
                                           resilience_df = lex1_resilience_df),
              generateAlgorithmTestResults(test_run_id = sprintf("lex2_%s_%s_%s", 
                                                                 data_id, 
                                                                 ml_alg_id,
                                                                 ifelse(ext.resilience,
                                                                        "res",
                                                                        "nores")),
                                           runtime_total = lex2_runtime_total,
                                           generations_total = lex2_gen_total,
                                           predictor = predictor,
                                           test_df = test_data_full,
                                           poi_tested_idxs = poi_tested_idxs,
                                           poi_vs_idxs = poi_vs_idxs,
                                           poi_res_idxs = poi_res_idxs,
                                           poi_rejected_count = lex2_poi_rejected,
                                           cf_df = lex2_cf_df,
                                           cf_all_invalid = lex2_cf_all_invalid,
                                           cf_null_returned = lex2_cf_null_returned,
                                           pd_wlt = lex2_pd_wlt,
                                           lx_wlt = lex2_lx_wlt,
                                           resilience_df = lex2_resilience_df)))
}


sink("logs/progress.txt", append = TRUE)
# Test across all ML algorithms.
for (ml_alg in ML_ALGS) {
  
  # Test across all datasets.
  for (ds in DATASETS) {
    # Placeholder for re-used data.
    test_data = NULL
    poi_idxs = NULL
    predictor = NULL
    
    # Prepare data.
    names(ds@df) = tolower(names(ds@df))
    ds@df = na.omit(ds@df)
    ds@df[,ds@target] = as.factor(ds@df[,ds@target])

    # Test both without resilience and with resilience.
    for (ext.resilience in list(FALSE, TRUE)) {
      rds.filename = sprintf("logs/test_%s_%s_%s.rds", 
                             ds@id, 
                             ml_alg@id,
                             ifelse(ext.resilience, "res", "nores"))
      # To avoid overwrite.
      if (!file.exists(rds.filename)) {
        writeLines(sprintf("Test ID:                    %s", rds.filename))
        writeLines(sprintf("Start time:                 %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")))
        results = run(ds@df,
                      ds@id,
                      ds@target,
                      ds@target_values,
                      ds@nonactionable,
                      ml_alg@id,
                      ml_alg@target_range,
                      ext.resilience = ext.resilience,
                      obj.ordering1 = obj.ordering1,
                      obj.ordering2 = obj.ordering2,
                      test_data = test_data,
                      poi_idxs = poi_idxs,
                      predictor = predictor,
                      n_points_of_interest = 1,
                      TD_PADDING_MULTIPLIER = 20,
                      TD_PADDING_ADDEND = 4)
        saveRDS(results, file = rds.filename)
        writeLines(sprintf("End time:                   %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")))
        writeLines("--------------------------------------")
        
        test_data = results[[1]]@test_df
        poi_idxs = results[[1]]@poi_tested_idxs
        predictor = results[[1]]@predictor
      }
    }
  }
}
sink()