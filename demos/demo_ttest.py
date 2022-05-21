from scipy.stats import ttest_ind_from_stats

print(
      ttest_ind_from_stats(mean1=78.65, std1=1.19, nobs1=10,
                           mean2=77.68, std2=1.28, nobs2=10)[1]/2
    )