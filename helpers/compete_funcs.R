source("../counterfactuals/R/generate_counterfactuals.R")

# Compare two candidate solutions (represented as fitness vectors) for Pareto
# dominance. If v1 dominates v2, return c(1, 0, 0) to symbolise a win. If v2
# dominates v1, return c(0, 1, 0) to symbolise a loss. If neither candidate
# dominates the other, return c(0, 0, 1) to symbolise a tie. 
comparePD <- function(v1, v2) {
  # V1 dominates V2: Win.
  if (all(v1 <= v2) & any(v1 < v2)) {
    return(c(1, 0, 0))
  }
  
  # V2 dominates V1: Loss.
  if (all(v2 <= v1) & any(v2 < v1)) {
    return(c(0, 1, 0))
  }
  
  # Neither candidate is dominant or dominated: Tie. 
  return(c(0, 0, 1))
}

# Compare two candidate solutions (represented as fitness vectors) by
# lexicographic selection. If v1 beats v2, return c(1, 0, 0) to symbolise a win. 
# If v2 beats v1, return c(0, 1, 0) to symbolise a loss. If neither 
# candidate wins (identical in performance), return c(0, 0, 1) to symbolise a 
# tie. 
compareLX <- function(v1, v2, obj.ordering) {
  
  # Candidates are identical in performance: Tie.
  if (all(v1 == v2)) {
    return(c(0, 0, 1))
  }
  
  # Lexicographic selection.
  winner = selTournamentLX(fitness = t(rbind(v1, v2)),
                           n.select = 1,
                           obj.ordering = obj.ordering,
                           k = 2,
                           theta = 0.01)
  
  # V1 outperforms V2: Win.
  if (winner == 1) {
    return(c(1, 0, 0))
  }
  
  # V2 outperforms V1: Loss.
  return(c(0, 1, 0))
}

# Returns a vector of results (W-L-T) for Pareto-dominance comparisons 
# between a full set of Pareto counterfactuals and a single lexicographic 
# counterfactual.
competePD <- function(lexico.set,
                      pareto.set)
{
  checkmate::assert_data_frame(lexico.set)
  if (nrow(lexico.set) != 1) {
    error("Error: Number of lexicographic counterfactuals must be precisely 1")
  }
  
  checkmate::assert_data_frame(pareto.set)
  
  # W-L-T record.
  wlt = c(0, 0, 0)
  
  # For each Pareto counterfactual...
  for (i in 1:nrow(pareto.set)) {
    wlt = wlt + comparePD(lexico.set[1,], pareto.set[i,])
  }
  
  return(wlt)
}

# Returns a vector of results (W-L-T) for lexicographic comparisons 
# between a full set of Pareto counterfactuals and a single lexicographic 
# counterfactual.
competeLX <- function(lexico.set,
                      pareto.set,
                      obj.ordering)
{
  checkmate::assert_data_frame(lexico.set)
  if (nrow(lexico.set) != 1) {
    error("Error: Number of lexicographic counterfactuals must be precisely 1")
  }
  
  checkmate::assert_data_frame(pareto.set)
  checkmate::assert_list(obj.ordering)
  
  # W-L-T record.
  wlt = c(0, 0, 0)
  
  # For each Pareto counterfactual...
  for (i in 1:nrow(pareto.set)) {
    wlt = wlt + compareLX(lexico.set[1,], pareto.set[i,], obj.ordering)
  }
  
  return(wlt)
}