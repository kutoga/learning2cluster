

log = """
metric_adjusted_mutual_info_score_avg20 0.73957 metric_adjusted_rand_score_avg20 0.735231 metric_completeness_score_avg20 0.837835 metric_fowlkes_mallows_score_avg20 0.824322 metric_homogeneity_score_avg20 0.815902 metric_misclassification_rate_BV01_avg20 0.148571 metric_mutual_info_score_avg20 0.670186 metric_normalized_mutual_info_score_avg20 0.812734 metric_purity_score_avg20 0.879286 metric_v_measure_score_avg20 0.810111

"""
log = input('Log line: ')

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
print()

# Print only the values as a single row (for excel)
print("\t".join([str(log[i + 1]) for i in range(0, len(log), 2)]))
