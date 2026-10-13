# Independent re-computation #2: R (base R + jsonlite).
#
# Written separately from the Python code. Rebuilds the calendar, recomputes
# every checked figure from the raw CSV and the exported predictions, and
# refits the interpretable GLM with R's own glm().
#
#   Rscript verify/verify.R  ->  verify/out/r.json

suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly = FALSE)
root <- normalizePath(file.path(dirname(sub("--file=", "", args[grep("--file=", args)])), ".."))
p <- function(...) file.path(root, ...)

raw <- read.csv(p("data/raw/bank-additional-full.csv"), sep = ";", stringsAsFactors = FALSE)
months <- c("jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec")
mnum <- match(raw$month, months)
raw$year <- 2008 + cumsum(c(0, diff(mnum) < 0))
raw$ym <- raw$year * 100 + mnum
raw$yes <- as.integer(raw$y == "yes")
raw$client_id <- seq_len(nrow(raw)) - 1L
n <- nrow(raw)

out <- list(
  n_clients = n, n_yes = sum(raw$yes), total_calls = sum(raw$campaign),
  n_months = length(unique(raw$ym)))
for (y in 2008:2010) {
  out[[sprintf("year_%d_clients", y)]] <- sum(raw$year == y)
  out[[sprintf("year_%d_yes", y)]] <- sum(raw$yes[raw$year == y])
}
for (k in 1:9) {
  out[[sprintf("call_%d_made", k)]] <- sum(raw$campaign >= k)
  out[[sprintf("call_%d_yes", k)]] <- sum(raw$campaign == k & raw$yes == 1)
}
out[["call_10+_made"]] <- sum(pmax(raw$campaign - 9, 0))
out[["call_10+_yes"]] <- sum(raw$campaign >= 10 & raw$yes == 1)
out$call2_vs_call1 <- (out$call_2_yes / out$call_2_made) / (out$call_1_yes / out$call_1_made)

capped <- pmin(raw$campaign, 10)
for (c in 1:10) {
  out[[sprintf("naive_%d_clients", c)]] <- sum(capped == c)
  out[[sprintf("naive_%d_conversion", c)]] <- mean(raw$yes[capped == c])
}

# chronological eighths, first (n %% 8) blocks one row larger
sizes <- rep(n %/% 8, 8) + c(rep(1, n %% 8), rep(0, 8 - n %% 8))
ends <- cumsum(sizes); starts <- c(1, head(ends, -1) + 1)
for (b in 1:8) out[[sprintf("eighth_%d_conversion", b)]] <- mean(raw$yes[starts[b]:ends[b]])

wins <- list(train_initial = c(200805, 200812), validation = c(200903, 200912),
             holdout = c(201003, 201011), out_of_sample = c(200903, 201011))
for (w in names(wins)) {
  s <- raw$ym >= wins[[w]][1] & raw$ym <= wins[[w]][2]
  out[[sprintf("window_%s_clients", w)]] <- sum(s)
  out[[sprintf("window_%s_yes", w)]] <- sum(raw$yes[s])
  out[[sprintf("window_%s_calls", w)]] <- sum(raw$campaign[s])
}

# ---------------------------------------------- metrics from predictions
auc <- function(score, label) {          # Mann-Whitney, average ranks for ties
  rk <- rank(score, ties.method = "average")
  npos <- sum(label == 1); nneg <- sum(label == 0)
  (sum(rk[label == 1]) - npos * (npos + 1) / 2) / (npos * nneg)
}
yes_kept <- function(score, calls, yes, budget) {
  o <- order(-score, method = "radix")   # stable, like the Python side
  cx <- c(0, cumsum(calls[o])); cy <- c(0, cumsum(yes[o]))
  approx(cx, cy, xout = budget * tail(cx, 1), ties = "ordered")$y / tail(cy, 1)
}

calls <- read.csv(p("outputs/oos_call_predictions.csv"))
clients <- read.csv(p("outputs/oos_client_predictions.csv"))
calls$ym <- raw$ym[match(calls$client_id, raw$client_id)]
clients$ym <- raw$ym[match(clients$client_id, raw$client_id)]
cands <- sub("^p_", "", grep("^p_", names(calls), value = TRUE))
wfilter <- list(out_of_sample = function(v) v >= 200903,
                validation = function(v) v >= 200903 & v <= 200912,
                holdout = function(v) v >= 201003)
for (w in names(wfilter)) {
  wc <- calls[wfilter[[w]](calls$ym), ]
  wl <- clients[wfilter[[w]](clients$ym), ]
  for (cd in cands) out[[sprintf("%s_auc_%s", w, cd)]] <- auc(wc[[paste0("p_", cd)]], wc$event)
  m <- ave(wc$p_logistic, wc$client_id, FUN = cummin)   # running minimum per client
  tot_c <- sum(wl$campaign); tot_y <- sum(wl$y)
  caps <- 1:max(wl$campaign)
  xs <- c(0, sapply(caps, function(k) sum(pmin(wl$campaign, k)) / tot_c))
  ys <- c(0, sapply(caps, function(k) sum(wl$y == 1 & wl$campaign <= k) / tot_y))
  for (bu in c(0.25, 0.5, 0.75)) {
    key <- function(x) sprintf("%s_%s_yes_kept_%s", w, x, format(bu))
    out[[key("next_call")]] <- yes_kept(m, rep(1, length(m)), wc$event, bu)
    out[[key("client_ranking_logistic")]] <- yes_kept(wl$s_logistic, wl$campaign, wl$y, bu)
    out[[key("call_cap")]] <- approx(xs, ys, xout = bu, ties = "ordered")$y
  }
  if (w == "out_of_sample") oos_calls <- wc
}

# ------------------- safeguards + money rule (independent of the Python code)
wc <- oos_calls
months <- sort(unique(wc$ym))
first <- wc[wc$ym == months[1], ]
out$regime_break_calls <- nrow(first)
out$regime_break_observed <- mean(first$event)
out$regime_break_predicted <- mean(first$p_logistic)
qlogis_c <- function(p) qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))
wc$p_g <- wc$p_logistic
wc$active <- FALSE
for (i in seq_along(months)) {
  if (i == 1) next                                  # shadow mode
  prev <- wc[wc$ym == months[i - 1], ]
  z <- qlogis_c(prev$p_logistic)
  shift <- uniroot(function(d) mean(plogis(z + d)) - mean(prev$event),
                   c(-20, 20), tol = 1e-12)$root
  if (abs(shift) > log(3)) next                     # drift alarm
  sel <- wc$ym == months[i]
  wc$p_g[sel] <- plogis(qlogis_c(wc$p_logistic[sel]) + shift)
  wc$active[sel] <- TRUE
}
out$active_share_of_calls <- mean(wc$active)
run_min <- ave(wc$p_g, wc$client_id, FUN = cummin)
total_yes <- sum(wc$event)
for (ratio in c(10, 20, 30, 50)) {
  made <- run_min >= 1 / ratio | !wc$active
  out[[sprintf("money_%d_calls_made", ratio)]] <- sum(made)
  out[[sprintf("money_%d_yes_kept", ratio)]] <- sum(wc$event[made])
  out[[sprintf("money_%d_cut_success_rate", ratio)]] <- mean(wc$event[!made])
  out[[sprintf("money_%d_gain", ratio)]] <- (nrow(wc) - sum(made)) - ratio * (total_yes - sum(wc$event[made]))
}
# pilot sample sizes (design A: client A/B; design B: flagged-call holdback)
for (ratio in c(10, 20, 30)) {
  made <- run_min >= 1 / ratio | !wc$active
  ctrl <- tapply(ratio * wc$event - 1, wc$client_id, sum)
  trt <- tapply(ratio * wc$event * made - made, wc$client_id, sum)
  delta <- mean(trt) - mean(ctrl)
  z <- qnorm(0.975) + qnorm(0.80)
  out[[sprintf("pilot_%d_a_n", ratio)]] <- ceiling(z^2 * (var(ctrl) + var(trt)) / delta^2)
  p0 <- 1 / ratio; p1 <- mean(wc$event[!made])
  n <- ceiling(((qnorm(0.95) * sqrt(p0 * (1 - p0)) + qnorm(0.80) * sqrt(p1 * (1 - p1))) / (p0 - p1))^2)
  exact_power <- function(n) {
    k <- qbinom(0.05, n, p0)
    while (k >= 0 && pbinom(k, n, p0) > 0.05) k <- k - 1
    pbinom(k, n, p1)
  }
  while (exact_power(n) < 0.80) n <- n + 1
  out[[sprintf("pilot_%d_b_n", ratio)]] <- n
  out[[sprintf("pilot_%d_b_power", ratio)]] <- exact_power(n)
}
made <- run_min >= 0.05 | !wc$active
out$example_rule_calls_share <- mean(made)
out$example_rule_yes_kept <- sum(wc$event[made]) / total_yes

# ------------------------------------------------ interpretable GLM in R
g <- read.csv(p("outputs/glm_input.csv"), stringsAsFactors = TRUE)
g$age_group <- relevel(g$age_group, ref = "a_25_60")
fit <- glm(event ~ call + age_group + contact + poutcome + euribor, data = g, family = binomial())
co <- coef(fit)
out$glm_n <- nobs(fit)
for (nm in names(co)) out[[paste0("glm_", nm)]] <- unname(co[nm])

dir.create(p("verify/out"), showWarnings = FALSE, recursive = TRUE)
write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), p("verify/out/r.json"))
cat(sprintf("R: %d figures written\n", length(out)))
