library(data.table)

# ======== unzip file and loading ========
zip_path <- "adult.zip"

info <- unzip(zip_path, list = TRUE)
target <- info$Name[grepl("adult", info$Name, ignore.case = TRUE) &
                    grepl("\\.(csv|data|txt)$", info$Name, ignore.case = TRUE)]
if (length(target) == 0) target <- info$Name[1]

exdir <- tempdir()
unzipped_path <- unzip(zip_path, files = target[1], exdir = exdir, overwrite = TRUE)

# comma/space (쉼표/공백 자동 감지)
first_line <- readLines(unzipped_path, n = 1)
sep_guess <- if (grepl(",", first_line)) "," else ""

adult <- fread(
  unzipped_path,
  header = FALSE,
  sep = sep_guess,
  na.strings = c("?", ""),
  strip.white = TRUE
)

# ======== colnames, missing handling (컬럼명, 공백, 결측 처리) ========
setnames(adult, c(
  "age","workclass","fnlwgt","education","education_num",
  "marital_status","occupation","relationship","race","sex",
  "capital_gain","capital_loss","hours_per_week","native_country","income"
))

# remove space and dot(공백 제거 + "." 제거)
char_cols <- names(adult)[vapply(adult, is.character, logical(1))]
for (c in char_cols) set(adult, j = c, value = trimws(adult[[c]]))
adult[, income := gsub("\\.", "", income)]
adult <- na.omit(adult)

# ======== transform target variable(타깃 변수 변환) ========
# income_num: 1 (>50K), 0 (<=50K)
adult[, income_num := as.integer(income == ">50K")]

# income_fac: factor (<=50K, >50K)
adult[, income_fac := factor(income_num, levels = c(0,1), labels = c("<=50K", ">50K"))]

# ======== Let catboost know which variables are categorical variables
#CatBoost용 범주형 변수 지정 ========
cat_cols <- c("workclass","education","marital_status","occupation",
              "relationship","race","sex","native_country")
for (c in cat_cols) {
  if (!is.factor(adult[[c]])) {
    set(adult, j = c, value = as.factor(adult[[c]]))
  }
}

# ========check the structure of dataset(최종 구조 확인) ========
str(adult)

# 4) Train/Test split (stratified by income_fac)
idx <- createDataPartition(adult$income_fac, p = 0.8, list = FALSE)
train <- adult[idx]
test  <- adult[-idx]

# Predictor sets
x_cols <- setdiff(names(train), c("income","income_num","income_fac"))

# ======================================================
# 5) Random Forest (formula, uses factors directly)
# ======================================================
rf_train <- copy(train[, c(x_cols, "income_fac"), with = FALSE])
rf_test  <- copy(test[,  c(x_cols, "income_fac"), with = FALSE])

set.seed(42)
rf_model <- randomForest(income_fac ~ ., data = rf_train, ntree = 300)
rf_pred  <- predict(rf_model, rf_test, type = "response")
rf_acc   <- mean(rf_pred == rf_test$income_fac)
cat(sprintf("RF   Test Accuracy: %.4f\n", rf_acc))

# ======================================================
# 6) XGBoost (dummy-encoded)
# ======================================================
dum <- dummyVars(~ ., data = train[, ..x_cols])
Xtr <- predict(dum, newdata = train[, ..x_cols])
Xte <- predict(dum, newdata = test[,  ..x_cols])
ytr <- train$income_num
yte <- test$income_num

dtrain <- xgb.DMatrix(data = as.matrix(Xtr), label = ytr)
dtest  <- xgb.DMatrix(data = as.matrix(Xte), label = yte)

params_xgb <- list(
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  nthread = nthread
)
set.seed(42)
xgb_model <- xgb.train(
  params = params_xgb, data = dtrain, nrounds = 2000,
  watchlist = list(valid = dtest),
  early_stopping_rounds = 50, verbose = 0
)
xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_prob > 0.5, 1L, 0L)
xgb_acc  <- mean(xgb_pred == yte)
cat(sprintf("XGB  Test Accuracy: %.4f (best_iter=%d)\n", xgb_acc, xgb_model$best_iteration))

# ======================================================
# 7) LightGBM (dummy-encoded; skip if not available)
# ======================================================
if (lgb_ready) {
  dtrain_lgb <- lgb.Dataset(as.matrix(Xtr), label = ytr, free_raw_data = FALSE)
  dvalid_lgb <- lgb.Dataset(as.matrix(Xte), label = yte, free_raw_data = FALSE)
  params_lgb <- list(
    objective = "binary",
    metric = "binary_error",
    learning_rate = 0.1,
    num_leaves = 63,
    max_depth = -1,
    feature_pre_filter = FALSE,   
    num_threads = nthread,
    verbosity = -1
  )
  lgb_model <- lgb.train(
    params = params_lgb, data = dtrain_lgb, nrounds = 3000,
    valids = list(valid = dvalid_lgb),
    early_stopping_rounds = 50, verbose = -1
  )
  lgb_prob <- predict(lgb_model, as.matrix(Xte))
  lgb_pred <- ifelse(lgb_prob > 0.5, 1L, 0L)
  lgb_acc  <- mean(lgb_pred == yte)
  cat(sprintf("LGB  Test Accuracy: %.4f (best_iter=%d)\n", lgb_acc, lgb_model$best_iter))
} else {
  lgb_acc <- NA_real_
  warning("lightgbm not available; skipping.")
}

# ======================================================
# 8) CatBoost (native categorical; no dummy encoding)
# ======================================================
if (cb_ready) {
  # 0-based categorical indices in the x_cols order
  cat_idx_0based <- which(x_cols %in% cat_cols) - 1L

  cb_train_pool <- catboost.load_pool(
    data = train[, ..x_cols],
    label = ytr,
    cat_features = cat_idx_0based
  )
  cb_test_pool <- catboost.load_pool(
    data = test[, ..x_cols],
    label = yte,
    cat_features = cat_idx_0based
  )

  params_cb <- list(
    loss_function = "Logloss",
    learning_rate = 0.1,
    depth = 8,
    iterations = 5000,     # upper bound (early stopping below)
    od_type = "Iter",
    od_wait = 50,
    random_seed = 42,
    thread_count = nthread,
    allow_writing_files = FALSE
  )
  cb_model <- catboost.train(cb_train_pool, params = params_cb, test_pool = cb_test_pool)
  cb_prob  <- catboost.predict(cb_model, cb_test_pool, prediction_type = "Probability")
  cb_pred  <- ifelse(cb_prob > 0.5, 1L, 0L)
  cb_acc   <- mean(cb_pred == yte)
  cat(sprintf("CAT  Test Accuracy: %.4f (best_iter≈%d)\n", cb_acc, cb_model$tree_count_))
} else {
  cb_acc <- NA_real_
  warning("catboost not available; skipping.")
}

# ======================================================
# 9) Summary table
# ======================================================
res <- data.frame(
  Model = c("Random Forest","XGBoost","LightGBM","CatBoost"),
  Test_Accuracy = c(rf_acc, xgb_acc, lgb_acc, cb_acc)
)
print(res[order(-res$Test_Accuracy), ], row.names = FALSE)


