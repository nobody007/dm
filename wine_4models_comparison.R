# ============================================================
# Boosting Models: XGBoost / LightGBM / CatBoost
# - Small tuning + CV + early stopping
# - Test error visualization
# - Best model: VIP + PDP
# Dataset: UCI Wine Quality (White)  – target: quality
# ============================================================

# 0) Packages (robust install)
base_pkgs <- c("data.table", "ggplot2", "xgboost")
for (p in base_pkgs) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(xgboost)
})

# LightGBM (CRAN or GitHub fallback)
if (!requireNamespace("lightgbm", quietly=TRUE)) {
  try(install.packages("lightgbm", repos="https://cloud.r-project.org"), silent = TRUE)
}
if (!requireNamespace("lightgbm", quietly=TRUE)) {
  if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org")
  try(remotes::install_github("microsoft/LightGBM", subdir="R-package", upgrade="never"), silent = TRUE)
}
lgb_ready <- requireNamespace("lightgbm", quietly=TRUE)
if (lgb_ready) library(lightgbm)

# CatBoost
if (!requireNamespace("catboost", quietly=TRUE)) {
  install.packages("catboost", repos="https://cloud.r-project.org")
}
cb_ready <- requireNamespace("catboost", quietly=TRUE)
if (cb_ready) library(catboost)

# PDP/Permutation-Importance (model-agnostic)
if (!requireNamespace("iml", quietly=TRUE)) install.packages("iml", repos="https://cloud.r-project.org")
suppressPackageStartupMessages(library(iml))

set.seed(42)
nthread <- max(1, parallel::detectCores() - 1)

# 1) Load data (UCI White Wine)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
wine <- read.csv(url, sep = ";")
stopifnot("quality" %in% names(wine))

y <- wine$quality
X <- as.matrix(wine[, setdiff(names(wine), "quality")])
feat_names <- colnames(X)

# Train/Test split (80/20)
n <- nrow(wine)
idx <- sample.int(n)
tr_n <- floor(0.8 * n)
tr_idx <- idx[1:tr_n]; te_idx <- idx[(tr_n+1):n]

Xtr <- X[tr_idx, , drop=FALSE]
ytr <- y[tr_idx]
Xte <- X[te_idx, , drop=FALSE]
yte <- y[te_idx]

# ============================================================
# 2) XGBOOST: tiny grid + CV + early stop
# ============================================================
dtrain_xgb <- xgb.DMatrix(Xtr, label = ytr)
dtest_xgb  <- xgb.DMatrix(Xte, label = yte)

etas       <- c(0.03, 0.1)
max_depths <- c(4, 6, 8)
nfold <- 5
max_nrounds <- 2000
esr <- 30

xgb_res <- data.frame(eta=numeric(), max_depth=integer(), best_iter=integer(), cv_rmse=numeric())
for (eta in etas) {
  for (md in max_depths) {
    params <- list(
      booster="gbtree", objective="reg:squarederror", eval_metric="rmse",
      eta=eta, max_depth=md, nthread=nthread
    )
    cv <- xgb.cv(
      params=params, data=dtrain_xgb, nfold=nfold,
      nrounds=max_nrounds, early_stopping_rounds=esr,
      maximize=FALSE, verbose=0
    )
    best_it <- cv$best_iteration
    best_rm <- cv$evaluation_log$test_rmse_mean[best_it]
    xgb_res <- rbind(xgb_res, data.frame(eta=eta, max_depth=md, best_iter=best_it, cv_rmse=best_rm))
    cat(sprintf("[XGB] eta=%.2f md=%d -> it=%d RMSE=%.4f\n", eta, md, best_it, best_rm))
  }
}
xgb_best <- xgb_res[which.min(xgb_res$cv_rmse), ]
xgb_model <- xgboost(
  params = list(objective="reg:squarederror", eval_metric="rmse",
                eta=xgb_best$eta, max_depth=xgb_best$max_depth, nthread=nthread),
  data = dtrain_xgb, nrounds = xgb_best$best_iter, verbose = 0
)
xgb_pred <- predict(xgb_model, dtest_xgb)
xgb_rmse <- sqrt(mean((xgb_pred - yte)^2))

# ============================================================
# 3) LIGHTGBM: tiny grid + CV + early stop (skip if not installed)
# ============================================================
if (lgb_ready) {
  dtrain_lgb <- lgb.Dataset(Xtr, label = ytr, free_raw_data = FALSE)
  lrates <- c(0.03, 0.1)
  leaves <- c(31, 63)
  depths <- c(-1, 6)
  lgb_res <- data.frame(lr=numeric(), num_leaves=integer(), max_depth=integer(),
                        best_iter=integer(), cv_rmse=numeric())
  for (lr in lrates) {
    for (nl in leaves) {
      for (md in depths) {
        params <- list(
          objective="regression", metric="rmse",
          learning_rate=lr, num_leaves=nl, max_depth=md,
          feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
          num_threads=nthread, verbosity=-1,
          feature_pre_filter = FALSE  # <- 중요! 튜닝 시 안전
        )
        cv <- lgb.cv(
          params=params, data=dtrain_lgb, nfold=nfold,
          nrounds=3000, early_stopping_rounds=esr,
          stratified=FALSE, verbose=-1
        )
        rmse_path <- unlist(cv$record_evals$valid$rmse$eval)
        best_it <- which.min(rmse_path)
        best_rm <- rmse_path[best_it]
        lgb_res <- rbind(lgb_res, data.frame(lr=lr, num_leaves=nl, max_depth=md,
                                             best_iter=best_it, cv_rmse=best_rm))
        cat(sprintf("[LGB] lr=%.2f leaves=%d md=%d -> it=%d RMSE=%.4f\n", lr, nl, md, best_it, best_rm))
      }
    }
  }
  lgb_best <- lgb_res[which.min(lgb_res$cv_rmse), ]
  lgb_model <- lgb.train(
    params = list(objective="regression", metric="rmse",
                  learning_rate=lgb_best$lr, num_leaves=lgb_best$num_leaves,
                  max_depth=lgb_best$max_depth, feature_fraction=0.9,
                  bagging_fraction=0.9, bagging_freq=1,
                  num_threads=nthread, verbosity=-1, feature_pre_filter=FALSE),
    data = dtrain_lgb, nrounds = lgb_best$best_iter, verbose = -1
  )
  lgb_pred <- predict(lgb_model, Xte)
  lgb_rmse <- sqrt(mean((lgb_pred - yte)^2))
} else {
  lgb_pred <- rep(NA_real_, length(yte))
  lgb_rmse <- NA_real_
  warning("lightgbm 패키지를 불러오지 못해 LightGBM 실험을 건너뜁니다.")
}

# ============================================================
# 4) CATBOOST: tiny grid + CV + early stop (no verbose param)
# ============================================================
if (cb_ready) {
  train_pool <- catboost.load_pool(Xtr, label = ytr)
  test_pool  <- catboost.load_pool(Xte, label = yte)
  depths <- c(4, 6, 8)
  lrates <- c(0.03, 0.1)
  cat_res <- data.frame(depth=integer(), lr=numeric(), best_iter=integer(), cv_rmse=numeric())

  for (dp in depths) {
    for (lr in lrates) {
      params <- list(
        loss_function = "RMSE",
        depth         = dp,
        learning_rate = lr,
        iterations    = 5000,      # upper bound
        od_type       = "Iter",
        od_wait       = esr,       # early stopping
        random_seed   = 42,
        allow_writing_files = FALSE,
        thread_count  = nthread
      )
      cv <- catboost.cv(
        train_pool,
        params = params,
        fold_count = nfold,
        partition_random_seed = 42
      )
      # 버전에 따라 "test-RMSE-mean" 혹은 "test.RMSE.mean"
      rmse_col  <- grep("test.*RMSE.*mean", names(cv$cv_results), value = TRUE)[1]
      rmse_path <- cv$cv_results[[rmse_col]]
      best_it   <- which.min(rmse_path)
      best_rm   <- rmse_path[best_it]

      cat_res <- rbind(cat_res, data.frame(depth=dp, lr=lr, best_iter=best_it, cv_rmse=best_rm))
      cat(sprintf("[CAT] lr=%.2f depth=%d -> it=%d RMSE=%.4f\n", lr, dp, best_it, best_rm))
    }
  }

  stopifnot(nrow(cat_res) > 0)
  cat_best <- cat_res[which.min(cat_res$cv_rmse), , drop=FALSE]
  # 스칼라 강제
  best_depth <- as.integer(cat_best$depth[1])
  best_lr    <- as.numeric(cat_best$lr[1])
  best_iter  <- as.integer(cat_best$best_iter[1])
  if (!is.finite(best_lr))    best_lr   <- 0.1
  if (!is.finite(best_depth)) best_depth <- 6
  if (!is.finite(best_iter))  best_iter  <- 500

  cat_model <- catboost.train(
    train_pool,
    params = list(
      loss_function="RMSE",
      depth=best_depth, learning_rate=best_lr,
      iterations=best_iter,
      random_seed=42, allow_writing_files=FALSE, thread_count=nthread
    )
  )
  cat_pred <- catboost.predict(cat_model, test_pool)
  cat_rmse <- sqrt(mean((cat_pred - yte)^2))
} else {
  cat_pred <- rep(NA_real_, length(yte))
  cat_rmse <- NA_real_
  warning("can't load the catboost package, skip CatBoost.")
}

# ============================================================
# 5) Compare on Test + Visualize Errors
# ============================================================
res <- data.frame(
  model = c("XGBoost", "LightGBM", "CatBoost"),
  test_RMSE = c(xgb_rmse, lgb_rmse, cat_rmse)
)
print(res)

# merge for visualization dataframe
pred_df <- rbind(
  data.frame(model="XGBoost",  y=yte, yhat=xgb_pred),
  data.frame(model="LightGBM", y=yte, yhat=lgb_pred),
  data.frame(model="CatBoost", y=yte, yhat=cat_pred)
)
pred_df$resid <- pred_df$yhat - pred_df$y

# (A) observed vs predicted scatter plot
p1 <- ggplot(pred_df, aes(x=y, y=yhat)) +
  geom_point(alpha=.4) +
  geom_abline(slope=1, intercept=0, linetype="dashed") +
  facet_wrap(~ model, scales="free") +
  labs(title="Predicted vs Actual (Test)", x="Actual", y="Predicted")
print(p1)

# (B) residual Distribution
p2 <- ggplot(pred_df, aes(x=resid)) +
  geom_histogram(bins=30) +
  facet_wrap(~ model, scales="free_y") +
  labs(title="Residual Distribution (Test)", x="Residual", y="Count")
print(p2)

# ============================================================
# 6) BEST model (min RMSE) → VIP & PDP
# ============================================================
# which model is the best?
avail <- is.finite(res$test_RMSE)
best_row <- res[avail, ][which.min(res$test_RMSE[avail]), , drop=FALSE]
best_model_name <- best_row$model[1]
cat(sprintf("\n[Best Model] %s (Test RMSE = %.4f)\n", best_model_name, best_row$test_RMSE))

# ---- VIP  ----
if (best_model_name == "XGBoost") {
  vip_tbl <- xgb.importance(model = xgb_model, feature_names = feat_names)
  vip_tbl <- vip_tbl[1:min(10, nrow(vip_tbl)), ]
  p_vip <- ggplot(vip_tbl, aes(x=reorder(Feature, Gain), y=Gain)) +
    geom_col() + coord_flip() + labs(title="XGBoost VIP (top 10)", x="", y="Gain")
  print(p_vip)

  # PDP with iml
  df_tr <- as.data.frame(Xtr); names(df_tr) <- feat_names
  pred_fun <- function(m, newdata) predict(m, newdata = xgb.DMatrix(as.matrix(newdata)))
  predictor <- Predictor$new(model = xgb_model, data = df_tr, y = ytr, predict.function = pred_fun)
} else if (best_model_name == "LightGBM" && lgb_ready) {
  vip_tbl <- lgb.importance(lgb_model)
  vip_tbl <- vip_tbl[1:min(10, nrow(vip_tbl)), ]
  p_vip <- ggplot(vip_tbl, aes(x=reorder(Feature, Gain), y=Gain)) +
    geom_col() + coord_flip() + labs(title="LightGBM VIP (top 10)", x="", y="Gain")
  print(p_vip)

  df_tr <- as.data.frame(Xtr); names(df_tr) <- feat_names
  pred_fun <- function(m, newdata) predict(lgb_model, as.matrix(newdata))
  predictor <- Predictor$new(model = lgb_model, data = df_tr, y = ytr, predict.function = pred_fun)
} else if (best_model_name == "CatBoost" && cb_ready) {
  # CatBoost VIP
  train_pool_full <- catboost.load_pool(Xtr, label = ytr)
  fi <- catboost.get_feature_importance(cat_model, train_pool_full, type = "FeatureImportance")
  vip_tbl <- data.frame(Feature = feat_names, Importance = fi)
  vip_tbl <- vip_tbl[order(-vip_tbl$Importance), ][1:min(10, nrow(vip_tbl)), ]
  p_vip <- ggplot(vip_tbl, aes(x=reorder(Feature, Importance), y=Importance)) +
    geom_col() + coord_flip() + labs(title="CatBoost VIP (top 10)", x="", y="Importance")
  print(p_vip)

  df_tr <- as.data.frame(Xtr); names(df_tr) <- feat_names
  pred_fun <- function(m, newdata) {
    tmp_pool <- catboost.load_pool(as.matrix(newdata))
    as.numeric(catboost.predict(cat_model, tmp_pool))
  }
  predictor <- Predictor$new(model = cat_model, data = df_tr, y = ytr, predict.function = pred_fun)
} else {
  stop("No best model available (check installed packages).")
}

# ---- PDP: top 2 variables PDP ----
# pick top variables from VIP table
if (best_model_name == "CatBoost") {
  top_feats <- vip_tbl$Feature[1:min(2, nrow(vip_tbl))]
} else {
  top_feats <- vip_tbl$Feature[1:min(2, nrow(vip_tbl))]
}
for (f in top_feats) {
  fe <- FeatureEffect$new(predictor, feature = f, method = "pdp")
  print(fe$plot() + ggtitle(paste0("PDP: ", f, " (", best_model_name, ")")))
}

cat("\nDone.\n")


## best model
print(res[order(res$test_RMSE), ])

## VIP
# ---- XGBoost VIP ----
vip_tbl <- xgb.importance(model = xgb_model, feature_names = feat_names)
vip_tbl <- vip_tbl[1:min(10, nrow(vip_tbl)), ]  # 상위 10개만 보기

# vip table
print(vip_tbl)

# visualization
library(ggplot2)
ggplot(vip_tbl, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "XGBoost Variable Importance (Top 10)",
    x = "",
    y = "Gain"
  )

### PDP
# =========================================
# top 2 variable pdp in one window
# =========================================

# packages
if (!requireNamespace("patchwork", quietly = TRUE)) {
  install.packages("patchwork", repos = "https://cloud.r-project.org")
}
library(ggplot2); library(xgboost); library(patchwork)

# 1) pick top 2 variables from vip table
if (!exists("vip_tbl")) {
  vip_tbl <- xgb.importance(model = xgb_model, feature_names = feat_names)
}
top_feats <- vip_tbl$Feature[1:min(2, nrow(vip_tbl))]

# 2) light pdp function with only means
pdp_fast <- function(model, X, feature, grid.size = 25) {
  stopifnot(feature %in% colnames(X))
  x <- as.data.frame(X)
  vals <- as.numeric(quantile(x[[feature]], probs = seq(0, 1, length.out = grid.size), na.rm = TRUE))
  out <- data.frame(value = vals, yhat = NA_real_)
  for (i in seq_along(vals)) {
    xx <- x
    xx[[feature]] <- vals[i]
    out$yhat[i] <- mean(predict(model, xgb.DMatrix(as.matrix(xx))))
  }
  out
}

# 3) pdp calculation for top 2 variables
p_list <- list()
for (f in top_feats) {
  df_pdp <- pdp_fast(xgb_model, Xtr, f, grid.size = 25)
  p <- ggplot(df_pdp, aes(x = value, y = yhat)) +
    geom_line() +
    labs(title = paste0("PDP (XGBoost): ", f), x = f, y = "Average prediction") +
    theme_minimal(base_size = 12)
  p_list[[f]] <- p
}

# 4) nrow=2, ncol=1 plot
# patchwork can merge top/bottom easily
if (length(p_list) == 2) {
  (p_list[[1]] / p_list[[2]]) + plot_layout(heights = c(1, 1))
} else {
  # if there is only one variable
  p_list[[1]]
}

# =================================================
# Smooth 2D PDP (heatmap + contour, stable version)
# =================================================
library(ggplot2)
library(xgboost)

# pick top 2 variables
if (!exists("vip_tbl")) vip_tbl <- xgb.importance(model = xgb_model, feature_names = feat_names)
top2 <- vip_tbl$Feature[1:2]; f1 <- top2[1]; f2 <- top2[2]

# smooth 2d pdp
pdp2d_smooth <- function(model, X, feat1, feat2, grid1 = 80, grid2 = 80, bg_n = 1500) {
  Xdf <- as.data.frame(X)
  set.seed(42)
  bg <- Xdf[sample(nrow(Xdf), min(bg_n, nrow(Xdf))), , drop = FALSE]
  
  v1 <- seq(min(Xdf[[feat1]], na.rm=TRUE), max(Xdf[[feat1]], na.rm=TRUE), length.out=grid1)
  v2 <- seq(min(Xdf[[feat2]], na.rm=TRUE), max(Xdf[[feat2]], na.rm=TRUE), length.out=grid2)
  
  grid <- expand.grid(v1, v2)
  names(grid) <- c(feat1, feat2)
  
  preds <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    tmp <- bg
    tmp[[feat1]] <- grid[[feat1]][i]
    tmp[[feat2]] <- grid[[feat2]][i]
    preds[i] <- mean(predict(model, xgb.DMatrix(as.matrix(tmp))))
  }
  grid$pred <- preds
  grid
}

# computation
df2d <- pdp2d_smooth(xgb_model, Xtr, f1, f2, grid1 = 80, grid2 = 80, bg_n = 1500)

# visualization (conti heatmap + contour line )
ggplot(df2d, aes_string(x = f1, y = f2, z = "pred")) +
  geom_raster(aes(fill = pred), interpolate = TRUE) +   # conti colored surface 
  geom_contour(color = "white", alpha = 0.5) +          # contour line
  scale_fill_viridis_c(option = "magma") +              # pick color 
  labs(
    title = paste0("2D PDP (XGBoost): ", f1, " × ", f2),
    fill = "Avg prediction"
  ) +
  theme_minimal(base_size = 12)

