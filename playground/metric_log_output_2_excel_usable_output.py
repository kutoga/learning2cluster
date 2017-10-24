

log = """
 metric_adjusted_mutual_info_score_avg20 0.740793 metric_adjusted_rand_score_avg20 0.739336 metric_completeness_score_avg20 0.82877 metric_fowlkes_mallows_score_avg20 0.836114 metric_homogeneity_score_avg20 0.82728 metric_misclassification_rate_BV01_avg20 0.135786 metric_mutual_info_score_avg20 0.669878 metric_normalized_mutual_info_score_avg20 0.806617 metric_purity_score_avg20 0.8935 metric_v_measure_score_avg20 0.804545

"""
log = input('Log line: ')

# Split the log
log = log.replace('.', ',') # This is just for german configured excels
log = list(filter(lambda l: len(l) > 0, log.strip().split(' ')))
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
