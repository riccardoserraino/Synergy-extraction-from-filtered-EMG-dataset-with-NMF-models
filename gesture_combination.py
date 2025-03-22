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
# Ask user to select datasets and their order
selected_paths = functions.select_datasets(dataset_options)

# Load EMG data 
emg_data_combined, timestamps = functions.load_combined_emg_data(selected_paths, topic_name)

###################################################################################################################################################################################
# Ask the user what kind of NMF to use: Classical or Sparse
l1_ratio, alpha_S_m = functions.get_nmf_approach(l1_ratio_classical, alpha_S_m_classical, l1_ratio_sparse, alpha_S_m_sparse)

###################################################################################################################################################################################
# VAF, Frobenius, Reconstruction Analysis
VAF_values = functions.compute_vaf(emg_data_combined, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)

REC_errors, REC_percent = functions.compute_rec_err(emg_data_combined, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)

###################################################################################################################################################################################
# Ask the user how many synergies he wants to use to reconstruct the signal
n_synergies = functions.get_synergy_count()

###################################################################################################################################################################################
# Residuals Analysis
functions.plot_residuals(emg_data_combined, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)

###################################################################################################################################################################################
# EMG Signal Analysis and Reconstruction
# Apply Sparse NMF
S_m, U, E, reconstruction_error = functions.apply_nmf(emg_data_combined, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)

# Reconstruct the signal starting from some elementar synergies
E_reconstructed = functions.reconstruct_signal(S_m, U, n_synergies)

# Scale the synergy signal to match the range of the EMG signa
S_m_scaled = functions.scale_synergy_signal(S_m, emg_data_combined)


print("Shape of E_reconstructed:", E_reconstructed.shape)  # (total_samples, num_channels)
print("Shape of S_m_scaled:", S_m_scaled.shape)  # (total_samples, n_synergies)
print("Shape of U:", U.shape)  # (n_synergies, num_channels)

# Plot results
functions.plot_results(emg_data_combined, E_reconstructed, S_m_scaled, U, n_synergies)

print("Session ended.")



