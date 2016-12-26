library(caTools)
path <- "/Users/jinyongshin/Desktop/Machine Learning/final_data/full_data_clust_lambda_1100.txt"
conn <- file(path,open="r")
lines <- readLines(conn)

bifold_partition <- function (k) {
  sample = sample.split(lines, SplitRatio = .80)
  train = subset(lines, sample == TRUE)
  test = subset(lines, sample == FALSE)
  cat("Sampled train data:", length(train), "\n")
  cat("Sampled test data:", length(test), "\n")
  train_set_name <- paste("full_clust_lambda_1100", k, ".train", sep="")
  write.table(train, train_set_name, quote = FALSE, col.names = FALSE, row.names = FALSE)
  test_set_name <- paste("full_clust_lambda_1100", k, ".test", sep="")
  write.table(test, test_set_name, quote = FALSE, col.names = FALSE, row.names = FALSE)
}

bifold_num <- 5
for (k in 1:bifold_num) {
  bifold_partition(k)
}
