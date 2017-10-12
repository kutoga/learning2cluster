

log = """
metric_adjusted_mutual_info_score_avg20 0.753292 metric_adjusted_rand_score_avg20 0.751749 metric_completeness_score_avg20 0.840444 metric_fowlkes_mallows_score_avg20 0.83808 metric_homogeneity_score_avg20 0.826981 metric_misclassification_rate_BV01_avg20 0.133313 metric_mutual_info_score_avg20 0.686789 metric_normalized_mutual_info_score_avg20 0.820655 metric_purity_score_avg20 0.893313 metric_v_measure_score_avg20 0.818292

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
