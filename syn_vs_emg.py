import functions

# Dictionary mapping numbers to dataset paths
dataset_options = {
    '1': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep0_pinch.bag',
    '2': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep0_ulnar.bag',
    '3': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep0_power.bag',
    '4': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep1_pinch.bag',
    '5': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep1_ulnar.bag',
    '6': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep1_power.bag',
    '7': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep2_pinch.bag',
    '8': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep2_ulnar.bag',
    '9': 'C:/Users/ricca/Desktop/int_th/dataset/emg_signal/rep2_power.bag'
}
topic_name = 'emg_rms' # Found by looking at the dataset info

max_synergies = 8  # It has to be lower than 8, which is the muscle sensors

# NMF Variables initialization (for Sparse or Classical)
init = 'nndsvd'
max_iter = 1000
l1_ratio_sparse = 0.8
l1_ratio_classical = 0.0
alpha_S_m_sparse = 0.01
alpha_S_m_classical = 0.0
random_state = 42

###################################################################################################################################################################################
# Ask the user what dataset to analyze
bag_path = functions.select_dataset(dataset_options)

# Load timestamps and EMG data
emg_data, timestamps = functions.load_emg_data(bag_path, topic_name)

# Shape data: each column corresponds to a muscle (sensor) and each row to a timestamp
print(f"Shape of emg_data: {emg_data.shape}") #(samples, n° muscles/sensors) == (rows, columns) == (1000, 8)

###################################################################################################################################################################################
# Ask the user what kind of NMF to use: Classical or Sparse
l1_ratio, alpha_S_m = functions.get_nmf_approach(l1_ratio_classical, alpha_S_m_classical, l1_ratio_sparse, alpha_S_m_sparse)

###################################################################################################################################################################################
# Compute VAF and Frobenius Norm Analysis
VAF_values = functions.compute_vaf(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)
REC_errors, REC_percent = functions.compute_rec_err(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)

###################################################################################################################################################################################
# Ask the user how many synergies he wants to use to reconstruct the signal
n_synergies = functions.get_synergy_count()

# Plot the residuals in the reconstruction based on the synergy number selected
functions.plot_residuals(emg_data, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)


###################################################################################################################################################################################
# Apply NMF
S_m, U, E, rec_error = functions.apply_nmf(emg_data, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)
# Check Matrices shape after NMF application
print(f"Shape of S_m (Samples x Synergies): {S_m.shape}")
print(f"Shape of U (Synergies x Muscles): {U.shape}")

# Scale the synergy extracted to match the originl data
S_m_scaled = functions.scale_synergy_signal(S_m, emg_data)

# Store synergy activations in a dictionary
synergy_data = functions.store_synergy(S_m_scaled, n_synergies)

# Get user selection for synergies
selected_synergies = functions.get_synergy_selection(n_synergies)

# Plot the selected synergies
functions.plot_synergies(timestamps, emg_data, synergy_data, selected_synergies)

print("Session ended.")















