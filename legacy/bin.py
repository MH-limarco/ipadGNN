import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 模擬數據或替換為你自己的資料
np.random.seed(42)
num_samples = 1000
entropy_change = np.random.rand(num_samples)  # normalized entropy change
true_power = 3.0
sample_prob = 1 - entropy_change ** true_power
agreement = np.random.binomial(n=1, p=sample_prob)  # 1: match, 0: not match

num_bins = 20
bins = np.linspace(0, 1, num_bins + 1)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 預測一致
hist_true, _ = np.histogram(entropy_change[agreement == 1], bins=bins)
# 預測不一致
hist_false, _ = np.histogram(entropy_change[agreement == 0], bins=bins)
hist = hist_true / (hist_true + hist_false + 1e-4)  # 避免除以零

def curve_func(x, a):
    return 1 - x ** a

fit_mask = ~np.isnan(hist)
fitted_power, _ = curve_fit(curve_func, bin_centers[fit_mask], hist[fit_mask], p0=[2.0], maxfev=1000)
fit_curve = curve_func(bin_centers, fitted_power)


# --------- 畫圖 ---------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(bin_centers, hist_true, width=1/num_bins*0.9, color='#7bc96f', alpha=0.85)
plt.xlabel('Normalized Entropy Change')
plt.ylabel('Sample Count')
plt.title('Teacher-Student Match\n(hist_true)')

plt.subplot(1, 2, 2)
plt.bar(bin_centers, hist_false, width=1/num_bins*0.9, color='#f37c7c', alpha=0.85)
plt.xlabel('Normalized Entropy Change')
plt.ylabel('Sample Count')
plt.title('Teacher-Student Mismatch\n(hist_false)')
plt.tight_layout()
plt.show()


plt.bar(bin_centers, hist, width=1/num_bins*0.9, color='#6ca0dc', alpha=0.85, label='Empirical $p_k$ (bar plot)')
plt.plot(bin_centers, fit_curve, 'r-', lw=2, label=fr'Fitted $p = 1 - x^{{{fitted_power[0]:.2f}}}$')
plt.xlabel('Normalized Entropy Change')
plt.ylabel('Empirical Probability (hist)')
plt.title('MP-KRD Curriculum: Empirical Bar Plot & Fitted Power Curve')
plt.ylim(0, 1.05)

plt.legend()
plt.tight_layout()
plt.show()