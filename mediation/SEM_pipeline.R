## ====================== 0. LOAD PACKAGES ======================
library(psych)
library(ggplot2)
library(GGally)
library(corrplot)
library(semPlot)
library(lavaan)
library(dplyr)
library(semTools)
library(purrr)

set.seed(42)

## ====================== 1. SETUP ======================
data_path <- "/Users/zayleazhongjiekua/Documents/camCAN/manuscript/WMBAG_data.csv"
out_dir   <- "/Users/zayleazhongjiekua/Documents/camCAN/out"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

cat("================================================================================\n")
cat("COMBINED SEM ANALYSIS\n")
cat("Part 1: Primary SEM with Latent Speed Factor\n")
cat("Part 2: Individual Risk Factor Models (Parallel Mediation)\n")
cat("Primary model: age + sex adjusted for both mediators and outcome\n")
cat("Bootstrap: 5000 iterations with BCa confidence intervals\n")
cat("================================================================================\n\n")

## ====================== HELPER FUNCTIONS ======================

get_fit_df <- function(fit, model_label) {
  fm <- fitMeasures(fit,
                    c("chisq", "df", "pvalue", "cfi", "tli",
                      "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
                      "rmsea.pvalue", "srmr", "npar", "ntotal"))
  data.frame(Model     = model_label,
             Index     = names(fm),
             Value     = as.numeric(fm),
             row.names = NULL)
}

zscore <- function(x) as.numeric(scale(x))

## ====================== 2. LOAD DATA ======================

mydata <- read.csv(data_path, stringsAsFactors = FALSE)

## ---- Required columns ----
need_core <- c("WMBAG", "GMBAG", "age", "sex", "risk_10y", "education", "BMI",
               "RTsimple_RTtrim3mean", "RTsimple_RTsd",
               "RTchoice_RTtrim3mean_all", "RTchoice_RTsd_all")
missing_cols <- need_core[!need_core %in% names(mydata)]
if (length(missing_cols) > 0) {
  cat("WARNING: Missing columns:", paste(missing_cols, collapse = ", "), "\n")
  cat("Please check that column names are correct (e.g., 'WMBAG' not 'Ridge_WMBAG')\n\n")
}

stopifnot(all(need_core[need_core %in% names(mydata)] %in% names(mydata)))

cat("Sample size (total):", nrow(mydata), "\n\n")

## ====================== 3. PREPARE REACTION TIME VARIABLES ======================

mydata$RTsimple_RTsd[mydata$RTsimple_RTsd == 0]         <- NA
mydata$RTchoice_RTsd_all[mydata$RTchoice_RTsd_all == 0] <- NA

mydata$RTsimple_RTtrim3mean_log         <- log(mydata$RTsimple_RTtrim3mean)
mydata$RTsimple_RTsd_log                <- log(mydata$RTsimple_RTsd)
mydata$RTchoice_RTtrim3mean_all_log     <- log(mydata$RTchoice_RTtrim3mean_all)
mydata$RTchoice_RTsd_all_log            <- log(mydata$RTchoice_RTsd_all)

mydata$RTsimple_RTtrim3mean_log_inv     <- -1 * mydata$RTsimple_RTtrim3mean_log
mydata$RTsimple_RTsd_log_inv            <- -1 * mydata$RTsimple_RTsd_log
mydata$RTchoice_RTtrim3mean_all_log_inv <- -1 * mydata$RTchoice_RTtrim3mean_all_log
mydata$RTchoice_RTsd_all_log_inv        <- -1 * mydata$RTchoice_RTsd_all_log

## ====================== 4. DATA PREPARATION ======================

mydata$risk10_per10 <- as.numeric(mydata$risk_10y) / 10

sx <- mydata$sex
if (is.numeric(sx)) {
  if (!all(sx %in% c(0, 1), na.rm = TRUE)) {
    u  <- sort(unique(na.omit(sx)))
    sx <- ifelse(is.na(sx), NA, ifelse(sx == u[1], 0L, 1L))
  }
} else {
  low <- tolower(trimws(as.character(sx)))
  sx  <- ifelse(grepl("\\bfemale\\b", low), 1L,
                ifelse(grepl("\\bmale\\b", low), 0L, NA))
}
mydata$sex01 <- as.numeric(sx)

if (!is.numeric(mydata$education)) {
  edu_chr <- trimws(tolower(as.character(mydata$education)))
  mydata$edu_ord <- dplyr::case_when(
    edu_chr %in% c("secondary") ~ 0,
    edu_chr %in% c("preu", "pre-u", "pre u", "junior college", "jc") ~ 1,
    edu_chr %in% c("degree", "university", "bachelor", "bachelors") ~ 2,
    TRUE ~ NA_real_
  )
} else {
  mydata$edu_ord <- as.numeric(mydata$education)
}

## ====================== 5. PREPARE INDIVIDUAL RISK FACTORS ======================
## NOTE: Diabetes excluded from analysis.
## Models run for hypertension, obesity, and smoking only.
## Bonferroni correction applied across 3 tests.

## ---- Obesity: BMI >= 30 (WHO definition) ----
mydata$obesity <- ifelse(is.na(mydata$BMI), NA,
                         ifelse(mydata$BMI >= 30, 1L, 0L))

cat("Obesity prevalence:\n")
print(table(mydata$obesity, useNA = "ifany"))
cat("\n")

if ("hypertension" %in% names(mydata)) {
  cat("Hypertension prevalence:\n")
  print(table(mydata$hypertension, useNA = "ifany"))
  cat("\n")
}

if ("smoking" %in% names(mydata)) {
  cat("Smoking prevalence:\n")
  print(table(mydata$smoking, useNA = "ifany"))
  cat("\n")
}

## ====================== 6. Z-SCORE VARIABLES ======================

mydata$risk10_z <- zscore(mydata$risk10_per10)
mydata$WMBAG_z <- zscore(mydata$WMBAG)
mydata$GMBAG_z       <- zscore(mydata$GMBAG)

# z-score RT indicators after log + inversion, before CFA
mydata$RTsimple_RTtrim3mean_z     <- zscore(mydata$RTsimple_RTtrim3mean_log_inv)
mydata$RTsimple_RTsd_z            <- zscore(mydata$RTsimple_RTsd_log_inv)
mydata$RTchoice_RTtrim3mean_all_z <- zscore(mydata$RTchoice_RTtrim3mean_all_log_inv)
mydata$RTchoice_RTsd_all_z        <- zscore(mydata$RTchoice_RTsd_all_log_inv)

## ====================== 7. ANALYSIS DATASET ======================

need_vars <- c("risk10_z", "WMBAG_z", "GMBAG_z", "age", "sex01",
               "RTsimple_RTtrim3mean_z", "RTsimple_RTsd_z",
               "RTchoice_RTtrim3mean_all_z", "RTchoice_RTsd_all_z",
               "obesity")

need_vars_rf <- c(need_vars, "hypertension", "smoking")
need_vars_rf <- need_vars_rf[need_vars_rf %in% names(mydata)]

d_sem <- mydata %>%
  dplyr::select(dplyr::all_of(need_vars_rf))

cat("Number of complete cases across all SEM variables:", sum(complete.cases(d_sem)), "\n")
cat("SEM estimation uses FIML with fixed.x = FALSE\n\n")

## ====================== 8. PRIMARY SEM MODEL ======================

cat("================================================================================\n")
cat("PART 1: PRIMARY SEM - PARALLEL MEDIATION WITH CV RISK\n")
cat("================================================================================\n\n")

primary_sem_model <- '
  # ---- Measurement model ----
  Speed_latent =~ RTsimple_RTtrim3mean_z + RTsimple_RTsd_z +
                  RTchoice_RTtrim3mean_all_z + RTchoice_RTsd_all_z

  # Residual covariances
  RTsimple_RTtrim3mean_z ~~ RTsimple_RTsd_z
  RTchoice_RTtrim3mean_all_z ~~ RTchoice_RTsd_all_z
  RTsimple_RTtrim3mean_z ~~ RTchoice_RTtrim3mean_all_z

  # ---- Mediators ----
  WMBAG_z ~ a_w*risk10_z + age + sex01
  GMBAG_z       ~ a_g*risk10_z + age + sex01

  # Residual covariance between mediators
  WMBAG_z ~~ GMBAG_z

  # ---- Outcome ----
  Speed_latent ~ b_w*WMBAG_z + b_g*GMBAG_z + cp*risk10_z + age + sex01

  # ---- Indirect effects ----
  ind_w    := a_w * b_w
  ind_g    := a_g * b_g
  ind_diff := ind_w - ind_g
  total    := cp + ind_w + ind_g

  # ---- Percent mediated ----
  pm_total := 100 * (ind_w + ind_g) / total
  pm_wm    := 100 * ind_w / total
  pm_gm    := 100 * ind_g / total
'

cat("Fitting primary SEM with bootstrap (5000 iterations)...\n")
fit_primary <- sem(
  primary_sem_model,
  data      = d_sem,
  estimator = "ML",
  missing   = "fiml",
  fixed.x   = FALSE,
  se        = "bootstrap",
  bootstrap = 5000
)
cat("✓ Primary SEM fitting complete\n\n")

## ====================== 9. EXTRACT PRIMARY SEM RESULTS ======================

pe_primary <- parameterEstimates(
  fit_primary,
  ci           = TRUE,
  level        = 0.95,
  boot.ci.type = "bca",
  standardized = TRUE
)

primary_results <- pe_primary %>%
  dplyr::filter(
    label %in% c("a_w", "a_g", "b_w", "b_g", "cp") |
      (op == ":=" & lhs %in% c("ind_w", "ind_g", "ind_diff", "total",
                               "pm_total", "pm_wm", "pm_gm")) |
      (op == "~~" & lhs == "WMBAG_z" & rhs == "GMBAG_z")
  ) %>%
  dplyr::mutate(
    Path = dplyr::case_when(
      label == "a_w"         ~ "a_w (risk -> WMBAG)",
      label == "a_g"         ~ "a_g (risk -> GMBAG)",
      label == "b_w"         ~ "b_w (WMBAG -> Speed)",
      label == "b_g"         ~ "b_g (GMBAG -> Speed)",
      label == "cp"          ~ "c' (direct: risk -> Speed)",
      lhs == "ind_w"         ~ "Indirect: White Matter",
      lhs == "ind_g"         ~ "Indirect: Grey Matter",
      lhs == "ind_diff"      ~ "Difference: WM - GM",
      lhs == "total"         ~ "Total effect",
      lhs == "pm_total"      ~ "% mediated (total)",
      lhs == "pm_wm"         ~ "% mediated (WM only)",
      lhs == "pm_gm"         ~ "% mediated (GM only)",
      lhs == "WMBAG_z"       ~ "Correlation: WMBAG ~~ GMBAG",
      TRUE ~ paste(lhs, op, rhs)
    ),
    Significant_CI = ifelse(ci.lower > 0 | ci.upper < 0, "Yes", "No")
  ) %>%
  dplyr::select(Path, est, se, z, pvalue, ci.lower, ci.upper, std.all, Significant_CI)

write.csv(
  primary_results,
  file.path(out_dir, "Primary_SEM_CV_risk_key_paths.csv"),
  row.names = FALSE
)

primary_fit <- get_fit_df(fit_primary, "Primary SEM - CV Risk")

write.csv(
  primary_fit,
  file.path(out_dir, "Primary_SEM_CV_risk_fit_indices.csv"),
  row.names = FALSE
)

writeLines(
  capture.output(
    summary(
      fit_primary,
      ci = TRUE,
      standardized = TRUE,
      fit.measures = TRUE,
      rsquare = TRUE,
      fm.args = list(robust = FALSE)
    )
  ),
  con = file.path(out_dir, "Primary_SEM_CV_risk_lavaan_summary.txt")
)

cat("Primary SEM Fit Indices:\n")
print(primary_fit)
cat("\n")

## ====================== 10. CREATE PATH DIAGRAM FOR PRIMARY SEM ======================

cat("Creating primary SEM path diagram...\n")

png(
  filename = file.path(out_dir, "Primary_SEM_CV_risk_path_diagram.png"),
  width = 2600, height = 1800, res = 220
)

semPaths(
  fit_primary,
  what = "std",
  whatLabels = "std",
  style = "lisrel",
  layout = "tree2",
  rotation = 2,
  residuals = FALSE,
  intercepts = FALSE,
  edge.label.cex = 0.9,
  sizeMan = 7,
  sizeLat = 10,
  nCharNodes = 0,
  mar = c(8, 8, 8, 8),
  title = TRUE
)

title(main = "Primary SEM: Cardiovascular Risk", line = 2, cex.main = 1.2)

dev.off()

pdf(
  file = file.path(out_dir, "Primary_SEM_CV_risk_path_diagram.pdf"),
  width = 14, height = 10
)

semPaths(
  fit_primary,
  what = "std",
  whatLabels = "std",
  style = "lisrel",
  layout = "tree2",
  rotation = 2,
  residuals = FALSE,
  intercepts = FALSE,
  edge.label.cex = 0.8,
  sizeMan = 7,
  sizeLat = 10,
  nCharNodes = 0,
  mar = c(8, 8, 8, 8),
  title = TRUE
)

title(main = "Primary SEM: Cardiovascular Risk", line = 2, cex.main = 1.2)

dev.off()

cat("✓ Primary SEM path diagrams created\n\n")

## ====================== 11. INDIVIDUAL RISK FACTOR MODEL BUILDER ======================

build_parallel_model <- function(risk_factor) {

  model <- paste0(
    '
  # ---- Measurement model ----
  Speed_latent =~ RTsimple_RTtrim3mean_z + RTsimple_RTsd_z +
                  RTchoice_RTtrim3mean_all_z + RTchoice_RTsd_all_z

  # Correlated residuals within and across tasks
  RTsimple_RTtrim3mean_z     ~~ RTsimple_RTsd_z
  RTchoice_RTtrim3mean_all_z ~~ RTchoice_RTsd_all_z
  RTsimple_RTtrim3mean_z     ~~ RTchoice_RTtrim3mean_all_z

  # ---- Mediators (covariates: age, sex) ----
  WMBAG_z ~ a_w*', risk_factor, ' + age + sex01
  GMBAG_z       ~ a_g*', risk_factor, ' + age + sex01

  # Residual covariance between mediators
  WMBAG_z ~~ GMBAG_z

  # ---- Outcome (covariates: age, sex) ----
  Speed_latent ~ b_w*WMBAG_z + b_g*GMBAG_z +
                 cp*', risk_factor, ' + age + sex01

  # ---- Indirect effects ----
  ind_w    := a_w * b_w
  ind_g    := a_g * b_g
  ind_diff := ind_w - ind_g
  total    := cp + ind_w + ind_g
  '
  )
  return(model)
}

## ====================== 12. INDIVIDUAL RISK FACTOR FIT FUNCTION ======================

fit_parallel_mediator <- function(data, risk_factor, out_dir, n_tests = 3) {

  cat("--------------------------------------------------------------------------------\n")
  cat("Risk factor:", risk_factor, "\n")

  ## ---- Bonferroni correction settings ----
  alpha_bonf    <- 0.05 / n_tests
  ci_level_bonf <- 1 - alpha_bonf

  cat("  Bonferroni-adjusted alpha:", round(alpha_bonf, 5), "\n")
  cat("  Bonferroni-adjusted CI level:", round(ci_level_bonf, 4), "\n")

  ## ---- Handle missing data per risk factor ----
  if (risk_factor == "hypertension") {
    data_model <- data %>%
      dplyr::filter(!is.na(.data[[risk_factor]]))
    use_fiml <- FALSE
    cat("  Listwise deletion for hypertension:",
        nrow(data) - nrow(data_model), "cases removed\n")
    cat("  N for this model:", nrow(data_model), "\n")
  } else {
    data_model <- data
    use_fiml   <- TRUE
    cat("  N for this model:", nrow(data_model), "\n")
  }

  model_syntax <- build_parallel_model(risk_factor)

  ## ---- Fit model with 5000 bootstrap samples ----
  fit <- suppressWarnings(
    sem(
      model_syntax,
      data      = data_model,
      estimator = "ML",
      missing   = ifelse(use_fiml, "fiml", "listwise"),
      fixed.x   = FALSE,
      se        = "bootstrap",
      bootstrap = 5000,
      iseed     = 42
    )
  )

  cat("  Model fitted successfully\n")

  ## ---- Calculate alpha and CI level for standard (uncorrected) test ----
  alpha_standard <- 0.05
  ci_level_standard <- 1 - alpha_standard

  ## ---- Extract parameter estimates with both CI levels ----
  ## Standard 95% BCa CIs (uncorrected)
  pe_95 <- parameterEstimates(
    fit,
    ci           = TRUE,
    level        = ci_level_standard,
    boot.ci.type = "bca",
    standardized = TRUE
  )

  ## Bonferroni-adjusted BCa CIs (corrected for multiple comparisons)
  pe_bonf <- parameterEstimates(
    fit,
    ci           = TRUE,
    level        = ci_level_bonf,
    boot.ci.type = "bca",
    standardized = TRUE
  )

  ## ---- Merge both CI sets ----
  pe <- pe_95 %>%
    dplyr::left_join(
      pe_bonf %>%
        dplyr::select(
          lhs, op, rhs, label,
          ci.lower.bonf = ci.lower,
          ci.upper.bonf = ci.upper
        ),
      by = c("lhs", "op", "rhs", "label")
    )

  ## ---- Extract key results ----
  key_results <- pe %>%
    dplyr::filter(
      label %in% c("a_w", "a_g", "b_w", "b_g", "cp") |
        (op == ":=" & lhs %in% c("ind_w", "ind_g", "ind_diff", "total")) |
        (op == "~~" & lhs == "WMBAG_z" & rhs == "GMBAG_z")
    ) %>%
    dplyr::mutate(
      Risk_Factor      = risk_factor,
      N_model          = nrow(data_model),
      Missing_handling = ifelse(
        risk_factor == "hypertension",
        "Listwise deletion (n=4 removed)",
        "FIML"
      ),
      Path = dplyr::case_when(
        label == "a_w"         ~ paste0("a_w (", risk_factor, " -> WMBAG)"),
        label == "a_g"         ~ paste0("a_g (", risk_factor, " -> GMBAG)"),
        label == "b_w"         ~ "b_w (WMBAG -> Speed)",
        label == "b_g"         ~ "b_g (GMBAG -> Speed)",
        label == "cp"          ~ paste0("c' (direct: ", risk_factor, " -> Speed)"),
        lhs == "ind_w"         ~ "Indirect: White Matter",
        lhs == "ind_g"         ~ "Indirect: Grey Matter",
        lhs == "ind_diff"      ~ "Difference: WM - GM",
        lhs == "total"         ~ "Total effect",
        lhs == "WMBAG_z"       ~ "Correlation: WMBAG ~~ GMBAG",
        TRUE ~ paste(lhs, op, rhs)
      ),
      Significant_CI = ifelse(ci.lower > 0 | ci.upper < 0, "Yes", "No"),
      Survives_Bonferroni = ifelse(
        ci.lower.bonf > 0 | ci.upper.bonf < 0,
        "Yes",
        "No"
      )
    ) %>%
    dplyr::select(
      Risk_Factor, N_model, Missing_handling, Path,
      est, se, z, pvalue,
      ci.lower, ci.upper,
      ci.lower.bonf, ci.upper.bonf,
      std.all, Significant_CI, Survives_Bonferroni
    )

  ## ---- Fit indices ----
  fit_df <- get_fit_df(fit, risk_factor)

  ## ---- Save outputs ----
  safe_name <- gsub("[^A-Za-z0-9]+", "_", risk_factor)

  write.csv(
    key_results,
    file.path(out_dir, paste0(safe_name, "_paths.csv")),
    row.names = FALSE
  )

  write.csv(
    fit_df,
    file.path(out_dir, paste0(safe_name, "_fit_indices.csv")),
    row.names = FALSE
  )

  writeLines(
    capture.output(
      summary(
        fit, ci = TRUE, standardized = TRUE,
        fit.measures = TRUE, rsquare = TRUE
      )
    ),
    file.path(out_dir, paste0(safe_name, "_lavaan_summary.txt"))
  )

  cat("  Saved outputs for:", risk_factor, "\n")

  return(list(
    fit         = fit,
    key_results = key_results,
    fit_df      = fit_df
  ))
}

## ====================== 13. RUN INDIVIDUAL RISK FACTOR MODELS ======================
## NOTE: Diabetes excluded. Running for: hypertension, obesity, smoking

cat("================================================================================\n")
cat("PART 2: INDIVIDUAL RISK FACTOR MODELS - PARALLEL MEDIATION\n")
cat("NOTE: Diabetes excluded from analysis.\n")
cat("================================================================================\n\n")

risk_factors <- c("hypertension", "obesity", "smoking")

missing_rf <- risk_factors[!risk_factors %in% names(d_sem)]
if (length(missing_rf) > 0) {
  cat("WARNING: The following risk factors are missing from the data:\n")
  cat("  ", paste(missing_rf, collapse = ", "), "\n")
  cat("  Check column names with: names(mydata)\n\n")
  risk_factors <- risk_factors[risk_factors %in% names(d_sem)]
}

n_tests <- length(risk_factors)
alpha_bonf    <- 0.05 / n_tests
ci_level_bonf <- 1 - alpha_bonf

cat("Running models for:", paste(risk_factors, collapse = ", "), "\n")
cat("Bootstrap samples: 5000\n")
cat("Standard CIs: 95% BCa\n")
cat("Bonferroni-adjusted alpha:", round(alpha_bonf, 5), "\n")
cat("Bonferroni-adjusted CIs:", round(ci_level_bonf * 100, 2), "% BCa\n\n")

all_results <- list()
all_fit     <- list()

for (rf in risk_factors) {
  res <- fit_parallel_mediator(
    data        = d_sem,
    risk_factor = rf,
    out_dir     = out_dir,
    n_tests     = n_tests
  )
  all_results[[rf]] <- res$key_results
  all_fit[[rf]]     <- res$fit_df
}

## ====================== 14. COMBINE AND SAVE INDIVIDUAL RF RESULTS ======================

all_results_df <- bind_rows(all_results)
all_fit_df     <- bind_rows(all_fit)

write.csv(
  all_results_df,
  file.path(out_dir, "ALL_risk_factors_paths_combined.csv"),
  row.names = FALSE
)

write.csv(
  all_fit_df,
  file.path(out_dir, "ALL_risk_factors_fit_indices_combined.csv"),
  row.names = FALSE
)

## ====================== 15. SUMMARY TABLE FOR INDIVIDUAL RF ======================

summary_table <- all_results_df %>%
  dplyr::filter(Path == "Indirect: White Matter") %>%
  dplyr::mutate(
    CI_95   = sprintf("%.4f [%.4f, %.4f]", est, ci.lower, ci.upper),
    CI_bonf = sprintf("%.4f [%.4f, %.4f]", est, ci.lower.bonf, ci.upper.bonf),
    P       = signif(pvalue, 4)
  ) %>%
  dplyr::select(
    Risk_Factor, N_model, Missing_handling,
    CI_95, CI_bonf, P,
    Significant_CI, Survives_Bonferroni
  )

cat("\n================================================================================\n")
cat("SUMMARY: WM INDIRECT EFFECTS BY INDIVIDUAL RISK FACTOR\n")
cat("(GMBAG retained as parallel mediator to control for GM brain ageing)\n")
cat("Risk factors:", paste(risk_factors, collapse = ", "), "\n")
cat("Bootstrap: 5000 samples, BCa CIs\n")
cat("Standard CI: 95% | Bonferroni-adjusted CI:",
    round(ci_level_bonf * 100, 2), "%\n")
cat("================================================================================\n\n")
print(summary_table)

write.csv(
  summary_table,
  file.path(out_dir, "Summary_WM_indirect_by_risk_factor.csv"),
  row.names = FALSE
)

## ====================== 16. FINISH ======================

cat("\n================================================================================\n")
cat("COMBINED ANALYSIS COMPLETE\n")
cat("================================================================================\n\n")

cat("Part 1: Primary SEM (Cardiovascular Risk)\n")
cat("  - Primary_SEM_CV_risk_key_paths.csv\n")
cat("  - Primary_SEM_CV_risk_fit_indices.csv\n")
cat("  - Primary_SEM_CV_risk_lavaan_summary.txt\n")
cat("  - Primary_SEM_CV_risk_path_diagram.png & .pdf\n\n")

cat("Part 2: Individual Risk Factor Models (Hypertension, Obesity, Smoking)\n")
cat("  - hypertension_paths.csv\n")
cat("  - obesity_paths.csv\n")
cat("  - smoking_paths.csv\n")
cat("  - hypertension_fit_indices.csv\n")
cat("  - obesity_fit_indices.csv\n")
cat("  - smoking_fit_indices.csv\n")
cat("  - hypertension_lavaan_summary.txt\n")
cat("  - obesity_lavaan_summary.txt\n")
cat("  - smoking_lavaan_summary.txt\n")
cat("  - ALL_risk_factors_paths_combined.csv\n")
cat("  - ALL_risk_factors_fit_indices_combined.csv\n")
cat("  - Summary_WM_indirect_by_risk_factor.csv\n\n")

cat("Column guide for output CSVs:\n")
cat("  ci.lower / ci.upper           : standard 95% BCa CI (uncorrected)\n")
cat("  ci.lower.bonf / ci.upper.bonf : Bonferroni-adjusted BCa CI (corrected for multiple comparisons)\n")
cat("  Significant_CI                : 95% CI excludes zero\n")
cat("  Survives_Bonferroni           : Bonferroni-adjusted CI excludes zero\n\n")

cat("Files saved to:", out_dir, "\n")
cat("================================================================================\n")
