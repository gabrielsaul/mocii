# Set the working directory.
wdir = "C:\\Users\\Owner\\OneDrive\\Documents\\Academic\\UKC\\CS\\3\\All\\COMP6200 Research Project\\MONOMOC\\monomoc\\testing"
setwd(wdir)

# Load results.
res = readRDS("logs/results.rds")

# Print results.
par_res = res[[1]]
lex_res = res[[2]]

par_res
lex_res
