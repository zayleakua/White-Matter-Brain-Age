############################################################
## sem_pipeline.R
## SEM pipeline for:
##  (1) Main mediation: risk_10y → WMBAG → Speed / Fluid_Intelligence
##  (2) Sex-moderated mediation (multi-group by Sex)
##  (3) Single VRF → WMBAG → cognition (with FDR)
############################################################

library(lavaan)
library(dplyr)

set.seed(42)

## ====================== 0. Paths & data ======================

data_path <- "data/WMBAG_data.csv"   
out_dir   <- "results"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

df <- read.csv(data_path, stringsAsFactors = FALSE)

## ---- shared columns / checks ----
need_core <- c("WMBAG","speed","fluid_intelligence","age","sex","education","risk_10y")
stopifnot(all(need_core %in% names(df)))

has_GMBAG <- "GMBAG" %in% names(df)

# Risk per 10%-points (risk_10y is in %)
df$risk10_per10 <- as.numeric(df$risk_10y) / 10

# Education factor + treatment dummies (Secondary ref)
df$Education_f <- factor(df$education,
                         levels = c(0, 1, 2),
                         labels = c("Secondary", "PreU", "Degree"))
MM <- model.matrix(~ Education_f, data = df)
df$Edu_PreU   <- as.numeric(MM[, "Education_fPreU"])
df$Edu_Degree <- as.numeric(MM[, "Education_fDegree"])

# SexGrp factor for multi-group models (0 = Male, 1 = Female)
sx <- df$sex
if (is.numeric(sx)) {
  if (!all(sx %in% c(0,1), na.rm = TRUE)) {
    u <- sort(unique(na.omit(sx)))
    sx <- ifelse(is.na(sx), NA, ifelse(sx == u[1], 0L, 1L))
  }
} else {
  low <- tolower(as.character(sx))
  sx <- ifelse(grepl("\\bfemale\\b", low), 1L,
               ifelse(grepl("\\bmale\\b",   low), 0L, NA))
}
df$SexGrp <- factor(ifelse(sx == 0, "Male", "Female"),
                    levels = c("Male","Female"))

cat("Sex coding check (0=Male, 1=Female):\n")
print(table(SexNumeric = sx, SexGrp = df$SexGrp, useNA = "ifany"))


############################################################
## (1) MAIN MEDIATION MODELS:
##     risk10_per10 → WMBAG → Speed / LCF2
############################################################

run_main_outcome <- function(df, outcome_var) {
  predictor <- "risk10_per10"
  mediator  <- "WMBAG"
  has_GMBAG <- "GMBAG" %in% names(df)
  
  covs_A <- c("age","sex","Edu_PreU","Edu_Degree")
  covs_B <- if (has_GMBAG) c(covs_A, "GMBAG") else covs_A
  
  cov_str <- function(x) paste(x, collapse = " + ")
  
  model_string <- function(covs) {
    paste0(
      mediator, " ~ a*", predictor, " + ", cov_str(covs), "\n",
      outcome_var, " ~ b*", mediator, " + cp*", predictor, " + ", cov_str(covs), "\n",
      "indirect := a*b\n",
      "total    := cp + a*b\n",
      "pm       := 100*(a*b)/(cp + a*b)\n"
    )
  }
  
  model_A <- model_string(covs_A)
  model_B <- model_string(covs_B)
  
  need_A <- c(mediator, outcome_var, predictor, covs_A)
  dA <- df %>%
    dplyr::select(dplyr::all_of(need_A)) %>%
    dplyr::filter(complete.cases(.))
  
  if (has_GMBAG) {
    need_B <- c(mediator, outcome_var, predictor, covs_B)
    dB <- df %>%
      dplyr::select(dplyr::all_of(need_B)) %>%
      dplyr::filter(complete.cases(.))
  }
  
  fit_A <- sem(model_A, data = dA, se = "bootstrap", bootstrap = 5000,
               meanstructure = TRUE, fixed.x = TRUE)
  if (has_GMBAG) {
    fit_B <- sem(model_B, data = dB, se = "bootstrap", bootstrap = 5000,
                 meanstructure = TRUE, fixed.x = TRUE)
  }
  
  pull_keys <- function(fit, model_label) {
    parameterEstimates(fit, ci = TRUE, level = 0.95,
                       boot.ci.type = "bca.simple",
                       standardized = FALSE) %>%
      dplyr::filter(label %in% c("a","b","cp","indirect","total","pm")) %>%
      dplyr::transmute(
        Model    = model_label,
        Outcome  = outcome_var,
        Path     = dplyr::case_when(
          label == "a"        ~ "a (risk_10y→WMBAG)",
          label == "b"        ~ paste0("b (WMBAG→", outcome_var, ")"),
          label == "cp"       ~ paste0("c' (risk_10y→", outcome_var, ")"),
          label == "indirect" ~ "indirect (a*b)",
          label == "total"    ~ "total",
          label == "pm"       ~ "% mediated"
        ),
        Effect   = est,
        SE       = se,
        CI_lower = ci.lower,
        CI_upper = ci.upper,
        p_value  = pvalue
      )
  }
  
  pe_A <- pull_keys(fit_A, "age+sex+Education_f (Secondary ref)")
  pe_B <- if (has_GMBAG)
    pull_keys(fit_B, "age+sex+Education_f+GMBAG") else NULL
  
  res <- dplyr::bind_rows(pe_A, pe_B) %>%
    dplyr::mutate(
      dplyr::across(c(Effect, SE, CI_lower, CI_upper), ~ round(., 3)),
      p_value = signif(p_value, 6)
    )
  
  attr(res, "N_A") <- nrow(dA)
  if (has_GMBAG) attr(res, "N_B") <- nrow(dB)
  
  res
}

run_main_mediation <- function(df, out_file) {
  outcomes <- c("speed","fluid_intelligence")
  all_results <- dplyr::bind_rows(
    lapply(outcomes, run_main_outcome, df = df)
  )
  write.csv(all_results, out_file, row.names = FALSE)
  cat("Main mediation results saved to:", out_file, "\n")
}

main_out_csv <- file.path(
  out_dir,
  "sem_mediation_risk_WMBAG_Speed_FluidIntelligence.csv"
)
run_main_mediation(df, main_out_csv)


############################################################
## (2) SEX-MODERATED MEDIATION (multi-group by SexGrp)
##     risk10_per10 → WMBAG → speed / fluid_intelligence
##     Only LRT (χ²) tests, no Wald tests
############################################################

cov_tail_M1 <- " + age + Edu_PreU + Edu_Degree"
cov_tail_M2 <- paste0(cov_tail_M1, " + GMBAG")

make_model_unconstrained <- function(Y, cov_tail) {
  paste0(
    "WMBAG ~ c(a_m, a_f)*risk10_per10", cov_tail, "\n",
    Y,     " ~ c(b_m, b_f)*WMBAG + c(cp_m, cp_f)*risk10_per10", cov_tail, "\n",
    "ind_m := a_m*b_m\n",
    "ind_f := a_f*b_f\n",
    "diff_ind := ind_m - ind_f\n",
    "total_m := cp_m + ind_m\n",
    "total_f := cp_f + ind_f\n"
  )
}

make_model_constrain_a  <- function(Y, cov_tail)
  paste0(
    "WMBAG ~ c(a, a)*risk10_per10", cov_tail, "\n",
    Y,     " ~ c(b_m, b_f)*WMBAG + c(cp_m, cp_f)*risk10_per10", cov_tail, "\n"
  )

make_model_constrain_b  <- function(Y, cov_tail)
  paste0(
    "WMBAG ~ c(a_m, a_f)*risk10_per10", cov_tail, "\n",
    Y,     " ~ c(b, b)*WMBAG + c(cp_m, cp_f)*risk10_per10", cov_tail, "\n"
  )

make_model_constrain_cp <- function(Y, cov_tail)
  paste0(
    "WMBAG ~ c(a_m, a_f)*risk10_per10", cov_tail, "\n",
    Y,     " ~ c(b_m, b_f)*WMBAG + c(cp, cp)*risk10_per10", cov_tail, "\n"
  )

run_mg_outcome <- function(df, Y, cov_tail, include_gmbag = FALSE, boot_R = 5000) {
  need <- c("risk10_per10","WMBAG","age","Edu_PreU","Edu_Degree","SexGrp",Y)
  if (include_gmbag) need <- c(need, "GMBAG")
  d <- df[, need]
  d <- d[complete.cases(d), , drop = FALSE]
  
  # Unconstrained model for bootstrap estimates
  mod_un   <- make_model_unconstrained(Y, cov_tail)
  fit_boot <- sem(mod_un, data = d, group = "SexGrp",
                  se = "bootstrap", bootstrap = boot_R,
                  meanstructure = TRUE)
  pe_boot  <- parameterEstimates(fit_boot, ci = TRUE, level = 0.95,
                                 boot.ci.type = "bca.simple",
                                 standardized = FALSE)
  
  # Indirects + totals by sex
  defs_boot <- pe_boot %>%
    dplyr::filter(op == ":=",
                  lhs %in% c("ind_m","ind_f","diff_ind","total_m","total_f")) %>%
    dplyr::transmute(
      Outcome = Y, Param = lhs, Est = est, SE = se,
      CI_low = ci.lower, CI_high = ci.upper, p = pvalue
    )
  
  # a_m/a_f, b_m/b_f, cp_m/cp_f
  abcp_boot <- pe_boot %>%
    dplyr::filter(op == "~",
                  lhs %in% c("WMBAG", Y),
                  label %in% c("a_m","a_f","b_m","b_f","cp_m","cp_f")) %>%
    dplyr::transmute(
      Outcome = Y, Param = label, Est = est, SE = se,
      CI_low = ci.lower, CI_high = ci.upper, p = pvalue
    )
  
  # LRT (χ²) tests for equality of a, b, c'
  fit_ml <- sem(mod_un, data = d, group = "SexGrp",
                se = "standard", meanstructure = TRUE)
  
  fit_a  <- sem(make_model_constrain_a(Y,  cov_tail),
                data = d, group = "SexGrp",
                se = "standard", meanstructure = TRUE)
  fit_b  <- sem(make_model_constrain_b(Y,  cov_tail),
                data = d, group = "SexGrp",
                se = "standard", meanstructure = TRUE)
  fit_cp <- sem(make_model_constrain_cp(Y, cov_tail),
                data = d, group = "SexGrp",
                se = "standard", meanstructure = TRUE)
  
  lrt_a  <- anova(fit_ml, fit_a)
  lrt_b  <- anova(fit_ml, fit_b)
  lrt_cp <- anova(fit_ml, fit_cp)
  
  tidy_lrt <- function(anv, path_label) {
    data.frame(
      Outcome = Y,
      Path    = path_label,
      Df      = anv$Df,
      AIC     = anv$AIC,
      BIC     = anv$BIC,
      Chisq   = anv$Chisq,
      `Chisq diff` = c(NA, diff(anv$Chisq)),
      `Df diff`    = c(NA, diff(anv$Df)),
      `Pr(>Chisq)` = c(NA, pchisq(diff(anv$Chisq),
                                  df = diff(anv$Df),
                                  lower.tail = FALSE)),
      check.names = FALSE
    )
  }
  
  lrt_tbl <- dplyr::bind_rows(
    tidy_lrt(lrt_a,  "Equal a (risk10_per10→WMBAG)"),
    tidy_lrt(lrt_b,  paste0("Equal b (WMBAG→", Y, ")")),
    tidy_lrt(lrt_cp, paste0("Equal c' (risk10_per10→", Y, ")"))
  )
  
  list(
    outcome   = Y,
    N         = nrow(d),
    abcp_boot = abcp_boot,
    defs_boot = defs_boot,
    lrt_tbl   = lrt_tbl
  )
}

run_mg_model <- function(df, model_label, cov_tail,
                         include_gmbag, out_csv_path) {
  cat("\n================ ", model_label, " ================\n", sep = "")
  outcomes <- c("speed","fluid_intelligence")
  res_list <- lapply(outcomes,
                     function(Y) run_mg_outcome(df, Y, cov_tail, include_gmbag))
  
  # Save path estimates + defined effects
  out_tab <- dplyr::bind_rows(
    dplyr::bind_rows(lapply(res_list, `[[`, "abcp_boot")) %>%
      dplyr::mutate(Table = "Paths_by_Sex_BootCI"),
    dplyr::bind_rows(lapply(res_list, `[[`, "defs_boot")) %>%
      dplyr::mutate(Table = "Indirects_Totals_BootCI")
  )
  write.csv(out_tab, out_csv_path, row.names = FALSE)
  cat("Saved bootstrap path/indirect tables to:", out_csv_path, "\n")
  
  # Save LRT χ² tables only
  lrt_all <- dplyr::bind_rows(lapply(res_list, `[[`, "lrt_tbl"))
  lrt_file <- sub("\\.csv$", "_LRT.csv", out_csv_path)
  write.csv(lrt_all, lrt_file, row.names = FALSE)
  cat("Saved LRT (χ²) tests to:", lrt_file, "\n")
}

mg_out1 <- file.path(out_dir, "mg_mediation_WMBAG_sex_pathtest_MODEL1.csv")
mg_out2 <- file.path(out_dir, "mg_mediation_WMBAG_sex_pathtest_MODEL2_GMBAG.csv")

run_mg_model(df, "Model 1: age+education", cov_tail_M1,
             include_gmbag = FALSE, out_csv_path = mg_out1)

if (has_GMBAG) {
  run_mg_model(df, "Model 2: age+education+GMBAG", cov_tail_M2,
               include_gmbag = TRUE, out_csv_path = mg_out2)
} else {
  message("Skipping sex-MG Model 2: GMBAG not found in data.")
}



############################################################
## (3) SINGLE-VRF MEDIATION:
##     Hypertension / Smoking / Diabetes → WMBAG → speed / fluid_intelligence
##     (with FDR for indirect effects)
############################################################

run_single_vrf_mediation <- function(df, out_file) {
  has_GMBAG <- "GMBAG" %in% names(df)
  
  # Three VRFs:
  risk_factors <- c(
    Hypertension = "hypertension",  # 0/1
    Smoking      = "smoking",       # 0/1
    Diabetes     = "diabetes"       # 0/1
  )
  risk_factors <- risk_factors[risk_factors %in% names(df)]
  stopifnot(length(risk_factors) > 0)
  
  # Coercions
  df$age   <- as.numeric(df$age)
  df$sex   <- as.integer(df$sex)  # 0/1
  df$speed <- as.numeric(df$speed)
  df$fluid_intelligence <- as.numeric(df$fluid_intelligence)
  df$WMBAG <- as.numeric(df$WMBAG)
  if (has_GMBAG) df$GMBAG <- as.numeric(df$GMBAG)
  
  # Ensure VRFs are 0/1
  to01 <- function(x) {
    x <- as.numeric(x)
    ifelse(is.na(x), NA, ifelse(x != 0, 1, 0))
  }
  for (nm in c("hypertension","smoking","diabetes")) {
    if (nm %in% names(df)) df[[nm]] <- to01(df[[nm]])
  }
  
  make_model <- function(Y, P, covars) {
    paste0(
      "WMBAG ~ a*", P, " + ", paste(covars, collapse=" + "), "\n",
      Y,     " ~ b*WMBAG + cp*", P, " + ", paste(covars, collapse=" + "), "\n",
      "indirect := a*b\n",
      "total    := cp + a*b\n"
    )
  }
  
  fit_sem <- function(dsub, model_string, boot_R=5000) {
    sem(model_string, data=dsub, se="bootstrap", bootstrap=boot_R,
        meanstructure=TRUE, fixed.x=TRUE)
  }
  
  pull_defs <- function(fit, model_label, outcome_label, predictor_label) {
    parameterEstimates(fit, ci=TRUE, level=0.95,
                       boot.ci.type="perc",
                       standardized=FALSE) %>%
      dplyr::filter(op==":=", lhs %in% c("indirect","total")) %>%
      dplyr::transmute(
        Model     = model_label,
        Outcome   = outcome_label,
        Predictor = predictor_label,
        Path      = lhs,   # "indirect" or "total"
        Effect    = est,
        SE        = se,
        CI_lower  = ci.lower,
        CI_upper  = ci.upper,
        p_value   = pvalue
      )
  }
  
  pull_paths <- function(fit, model_label, outcome_label, predictor_label) {
    parameterEstimates(fit, ci=TRUE, level=0.95,
                       boot.ci.type="bca.simple",
                       standardized=FALSE) %>%
      dplyr::filter(label %in% c("a","b","cp")) %>%
      dplyr::transmute(
        Model     = model_label,
        Outcome   = outcome_label,
        Predictor = predictor_label,
        Path      = dplyr::case_when(
          label=="a"  ~ paste0("a (", predictor_label,"→WMBAG)"),
          label=="b"  ~ paste0("b (WMBAG→", outcome_label, ")"),
          label=="cp" ~ paste0("c' (", predictor_label,"→", outcome_label, ")")
        ),
        Effect    = est,
        SE        = se,
        CI_lower  = ci.lower,
        CI_upper  = ci.upper,
        p_value   = pvalue
      )
  }
  
  outcomes <- c("speed","fluid_intelligence")
  all_rows <- list()
  
  for (pred_label in names(risk_factors)) {
    P_col <- risk_factors[[pred_label]]
    P <- P_col  # no SBP or other transformations now
    
    for (Y in outcomes) {
      covs_M1 <- c("age","sex","Edu_PreU","Edu_Degree")
      covs_M2 <- if (has_GMBAG) c(covs_M1, "GMBAG") else covs_M1
      
      need1 <- c("WMBAG", Y, P, covs_M1)
      d1 <- df[, need1, drop=FALSE] %>% dplyr::filter(complete.cases(.))
      need2 <- c("WMBAG", Y, P, covs_M2)
      d2 <- df[, need2, drop=FALSE] %>% dplyr::filter(complete.cases(.))
      
      mod1 <- make_model(Y, P, covs_M1)
      mod2 <- make_model(Y, P, covs_M2)
      
      fit1 <- fit_sem(d1, mod1, boot_R=5000)
      fit2 <- fit_sem(d2, mod2, boot_R=5000)
      
      tab1 <- dplyr::bind_rows(
        pull_paths(fit1, "Model 1: age+sex+education", Y, pred_label),
        pull_defs(fit1,  "Model 1: age+sex+education", Y, pred_label)
      ) %>% dplyr::mutate(N = nrow(d1))
      
      tab2 <- dplyr::bind_rows(
        pull_paths(fit2, "Model 2: +GMBAG", Y, pred_label),
        pull_defs(fit2,  "Model 2: +GMBAG", Y, pred_label)
      ) %>% dplyr::mutate(N = nrow(d2))
      
      all_rows[[length(all_rows)+1]] <- dplyr::bind_rows(tab1, tab2)
    }
  }
  
  results <- dplyr::bind_rows(all_rows) %>%
    dplyr::mutate(
      dplyr::across(c(Effect, SE, CI_lower, CI_upper), ~ round(., 3)),
      p_value = signif(p_value, 6)
    )
  
  # FDR across ALL indirect effects (a*b)
  is_indirect <- results$Path == "indirect"
  results$FDR_q_indirect_all <- NA_real_
  if (any(is_indirect, na.rm = TRUE)) {
    results$FDR_q_indirect_all[is_indirect] <-
      p.adjust(results$p_value[is_indirect], method = "BH")
  }
  
  # Optional: FDR within Outcome × Model for indirects
  results$FDR_q_indirect_byModel <- NA_real_
  if (any(is_indirect, na.rm = TRUE)) {
    results <- results %>%
      dplyr::group_by(Outcome, Model) %>%
      dplyr::mutate(FDR_q_indirect_byModel = ifelse(
        Path == "indirect",
        p.adjust(p_value, method = "BH"),
        NA_real_
      )) %>%
      dplyr::ungroup()
  }
  
  fmt_p <- function(x) ifelse(is.na(x), NA_character_,
                              ifelse(x < 0.001, "<0.001", sprintf("%.3f", x)))
  results <- results %>%
    dplyr::mutate(
      p_fmt   = fmt_p(p_value),
      q_all   = fmt_p(FDR_q_indirect_all),
      q_byMod = fmt_p(FDR_q_indirect_byModel)
    )
  
  write.csv(results, out_file, row.names = FALSE)
  cat("\nSingle-VRF mediation (with FDR) saved to:", out_file, "\n")
}

vrf_out_csv <- file.path(
  out_dir,
  "sem_by_riskfactor_mediation.csv"
)
run_single_vrf_mediation(df, vrf_out_csv)
