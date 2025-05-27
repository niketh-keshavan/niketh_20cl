import pandas as pd
import matplotlib.pyplot as plt

# Load the synthetic spike data
df = pd.read_csv('data/spike_data.csv')

# Compute the timing difference
df['delta_t'] = df['time_B_s'] - df['time_A_s']

# Plot Δt vs time_A_s
plt.figure()
plt.plot(df['time_A_s'], df['delta_t'])
plt.xlabel('Time of Neuron A Spike (s)')
plt.ylabel('Spike Timing Difference Δt = t_B - t_A (s)')
plt.title('Spike Timing Difference Over Time')
plt.tight_layout()
plt.show()