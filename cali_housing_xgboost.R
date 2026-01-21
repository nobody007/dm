# ---------------------------------------------
# dafulat xgboost model fitting and test RMSE
# ---------------------------------------------

# packages
req_pkgs <- c("data.table", "xgboost")
to_install <- req_pkgs[!(req_pkgs %in% rownames(installed.packages()))]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
lapply(req_pkgs, library, character.only = TRUE)

# setup
data_path <- "fetch_california_housing.csv"
target_var <- "MedHouseVal"
set.seed(42)

# 1) data loading
dt <- data.table::fread(data_path)
stopifnot(target_var %in% names(dt))
dt <- dt[complete.cases(dt), ]

# 2) train/test 80:20
n <- nrow(dt)
idx <- sample.int(n)
train_ratio <- 0.8
n_train <- floor(n * train_ratio)

train_idx <- idx[1:n_train]
test_idx  <- idx[(n_train + 1):n]

train <- dt[train_idx]
test  <- dt[test_idx]

y_train <- train[[target_var]]
X_train <- as.matrix(train[, setdiff(names(train), target_var), with = FALSE])

y_test  <- test[[target_var]]
X_test  <- as.matrix(test[, setdiff(names(test), target_var), with = FALSE])

dtrain <- xgb.DMatrix(X_train, label = y_train)
dtest  <- xgb.DMatrix(X_test,  label = y_test)

# 3) use default parameters
# put objective and eval metrics 
# nrounds 100 is a good starting value
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse"
)

model <- xgboost(
  params = params,
  data = dtrain,
  nrounds = 100,   # number of trees (or seq samples)
  verbose = 0
)

# 4) prediction and RMSE
pred <- predict(model, dtest)
rmse <- sqrt(mean((pred - y_test)^2))

cat(sprintf("Test RMSE: %.6f\n", rmse))

# VIP
imp <- xgb.importance(model = model)
print(head(imp, 10))
xgb.save(model, "xgb_california_default.model")





# ------------------------------------------------------------
# XGBoost (eta, max_depth tuning) + nrounds=500 
# ------------------------------------------------------------

# packages
req_pkgs <- c("data.table", "xgboost")
to_install <- req_pkgs[!(req_pkgs %in% rownames(installed.packages()))]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
lapply(req_pkgs, library, character.only = TRUE)

# ===== 1) setup =====
data_path <- "fetch_california_housing.csv"
target_var <- "MedHouseVal"
seed <- 42; set.seed(seed)
nfold <- 5
nrounds <- 500

etas       <- c(0.03, 0.1, 0.3)
max_depths <- c(3, 6, 9)

# ===== 2) data =====
dt <- data.table::fread(data_path)
stopifnot(target_var %in% names(dt))
dt <- dt[complete.cases(dt), ]

y <- dt[[target_var]]
X <- as.matrix(dt[, setdiff(names(dt), target_var), with = FALSE])
dtrain <- xgb.DMatrix(data = X, label = y)

# ===== 3) CV  =====
results <- data.frame(
  eta = numeric(), max_depth = integer(),
  mean_rmse = numeric(), stringsAsFactors = FALSE
)

for (eta in etas) {
  for (md in max_depths) {
    params <- list(
      booster = "gbtree",
      objective = "reg:squarederror",
      eval_metric = "rmse",
      eta = eta,
      max_depth = md
    )

    cv <- xgb.cv(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      nfold = nfold,
      verbose = 0,
      maximize = FALSE,
      seed = seed
    )

    mean_rmse <- cv$evaluation_log$test_rmse_mean[nrounds]

    results <- rbind(
      results,
      data.frame(eta = eta, max_depth = md, mean_rmse = mean_rmse)
    )

    cat(sprintf("eta=%.3f, max_depth=%d -> RMSE=%.5f\n",
                eta, md, mean_rmse))
  }
}

best <- results[which.min(results$mean_rmse), ]
cat("\n==== best params====\n")
print(best)

# ------------------------------------------------------------
# XGBoost Random Search 
# California housing dataset
# ------------------------------------------------------------

req_pkgs <- c("data.table", "xgboost")
to_install <- req_pkgs[!(req_pkgs %in% rownames(installed.packages()))]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")
lapply(req_pkgs, library, character.only = TRUE)

# ===== 1) setup =====
data_path <- "fetch_california_housing.csv"
target_var <- "MedHouseVal"
set.seed(42)
nfold <- 5
n_trials <- 30              # number of trials
max_nrounds <- 3000
early_stopping_rounds <- 50
nthread <- max(1, parallel::detectCores() - 1)

# ===== 2) data =====
dt <- data.table::fread(data_path)
stopifnot(target_var %in% names(dt))
dt <- dt[complete.cases(dt), ]

y <- dt[[target_var]]
X <- as.matrix(dt[, setdiff(names(dt), target_var), with = FALSE])
dtrain <- xgb.DMatrix(data = X, label = y)

# ===== 3) Random parameter sampler =====
sample_params <- function() {
  list(
    booster = "gbtree",
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = runif(1, 0.01, 0.3),                # learning rate
    max_depth = sample(3:10, 1),
    min_child_weight = sample(1:10, 1),
    subsample = runif(1, 0.7, 1.0),
    colsample_bytree = runif(1, 0.7, 1.0),
    gamma = runif(1, 0, 1),
    lambda = 10^runif(1, -1, 2),              # regularization
    nthread = nthread
  )
}

# ===== 4) random search =====
results <- data.frame()

for (t in seq_len(n_trials)) {
  params <- sample_params()

  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nfold = nfold,
    nrounds = max_nrounds,
    early_stopping_rounds = early_stopping_rounds,
    maximize = FALSE,
    verbose = 0
  )

  best_it   <- cv$best_iteration
  best_rmse <- cv$evaluation_log$test_rmse_mean[best_it]

  results <- rbind(
    results,
    data.frame(
      trial = t,
      eta = params$eta,
      max_depth = params$max_depth,
      min_child_weight = params$min_child_weight,
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      gamma = params$gamma,
      lambda = params$lambda,
      best_iter = best_it,
      best_rmse = best_rmse
    )
  )

  cat(sprintf("[Trial %02d/%02d] eta=%.3f, depth=%2d, RMSE=%.5f\n",
              t, n_trials, params$eta, params$max_depth, best_rmse))
}

best_row <- results[which.min(results$best_rmse), ]
cat("\n==== best params ====\n")
print(best_row)

# ===== 5) refit all training data with best params =====
best_params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = best_row$eta,
  max_depth = best_row$max_depth,
  min_child_weight = best_row$min_child_weight,
  subsample = best_row$subsample,
  colsample_bytree = best_row$colsample_bytree,
  gamma = best_row$gamma,
  lambda = best_row$lambda,
  nthread = nthread
)

final_model <- xgboost(
  params = best_params,
  data = dtrain,
  nrounds = best_row$best_iter,
  verbose = 0
)

cat(sprintf("\n[Best] eta=%.3f, depth=%d, trees=%d, RMSE=%.6f\n",
            best_row$eta, best_row$max_depth, best_row$best_iter, best_row$best_rmse))

imp <- xgb.importance(model = final_model)
print(head(imp, 10))
xgb.save(final_model, "xgb_randomsearch_best.model")
data.table::fwrite(results, "xgb_randomsearch_results.csv")
