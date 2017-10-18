

log = """
metric_adjusted_mutual_info_score_avg20 0.156813 metric_adjusted_rand_score_avg20 0.151745 metric_completeness_score_avg20 0.31238 metric_fowlkes_mallows_score_avg20 0.511218 metric_homogeneity_score_avg20 0.418947 metric_misclassification_rate_BV01_avg20 0.448438 metric_mutual_info_score_avg20 0.239717 metric_normalized_mutual_info_score_avg20 0.266477 metric_purity_score_avg20 0.67125 metric_v_measure_score_avg20 0.261955

"""

# Split the log
log = log.replace('.', ',') # This is just for german configured excels
log = log.strip().split(' ')
assert len(log) % 2 == 0

# Print it (the output just may be copy pasted to excel)
for i in range(0, len(log), 2):
    print("{}\t{}".format(log[i], log[i + 1]))

print()

# Print only the values
for i in range(0, len(log), 2):
    print("{}".format(log[i + 1]))
