import functions

# Dictionary mapping numbers to dataset paths
dataset_options = {
    '1': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_pinch.bag',
    '2': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_ulnar.bag',
    '3': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep0_power.bag',
    '4': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_pinch.bag',
    '5': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_ulnar.bag',
    '6': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep1_power.bag',
    '7': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_pinch.bag',
    '8': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_ulnar.bag',
    '9': 'C:/Users/ricca/Desktop/th_unibo/dataset/emg_signal/rep2_power.bag'
}
topic_name = 'emg_rms' # Found by looking at the dataset info

# Initialization of synergy for a comparison analysis
n_synergies_sparse = 3  
n_synergies_classical = 3  
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
print(f"Shape of EMG Data: {emg_data.shape}") #(samples, n° muscles/sensors) == (rows, columns) == (1000, 8)


###################################################################################################################################################################################
# Ask the user what kind of NMF to use: Classical or Sparse or both
l1_ratio, alpha_S_m = functions.get_nmf_approach(l1_ratio_classical, alpha_S_m_classical, l1_ratio_sparse, alpha_S_m_sparse)

###################################################################################################################################################################################
# Compute VAF and Frobenius Norm Analysis
VAF_values = functions.compute_vaf(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)
REC_errors, REC_percent = functions.compute_rec_err(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_S_m, random_state)

###################################################################################################################################################################################
# Ask the user how many synergies he wants to use to reconstruct the signal
n_synergies = functions.get_synergy_count()

#Plot the reconstruction error based on the synergy number selection
functions.plot_residuals(emg_data, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)

###################################################################################################################################################################################
# Apply NMF
S_m, U, E, rec_error = functions.apply_nmf(emg_data, n_synergies, init, max_iter, l1_ratio, alpha_S_m, random_state)
# Check Matrices shape after NMF application
print(f"Shape of S_m (Samples x Synergies): {S_m.shape}")
print(f"Shape of U (Synergies x Muscles): {U.shape}")

# Scale the synergy extracted to match the originl data
S_m_scaled = functions.scale_synergy_signal(S_m, emg_data)

# Reconstruct the signal starting from some elementar synergies
E_reconstructed = functions.reconstruct_signal(S_m, U, n_synergies)


print("Shape of E_Reconstructed:", E_reconstructed.shape)  # (total_samples, num_channels)
print("Shape of S_m_scaled:", S_m_scaled.shape)  # (total_samples, n_synergies)
print("Shape of U:", U.shape)  # (n_synergies, num_channels)

# Plot results
functions.plot_results(emg_data, E_reconstructed, S_m_scaled, U, n_synergies)



###################################################################################################################################################################################
# Ask the user whether he wants to compare the NMF approachs

choice = functions.ask_for_comparison()

if choice == 'y':
    REC_errors = functions.compute_rec_err_side_by_side(emg_data, max_synergies, l1_ratio_classical, l1_ratio_sparse, init, max_iter, alpha_S_m_classical, alpha_S_m_sparse, random_state)
    VAF_values = functions.compute_vaf_side_by_side(emg_data, max_synergies, l1_ratio_classical, l1_ratio_sparse, init, max_iter, alpha_S_m_classical, alpha_S_m_sparse, random_state)
    functions.plot_residuals_side_by_side(emg_data, E_reconstructed, E_reconstructed)
    n_synergies = functions.get_synergy_count()


    # Sparse NMF
    S_m_s, U_s, E_s, reconstr_error_s = functions.apply_nmf(emg_data, n_synergies_sparse, init, max_iter, l1_ratio_sparse, alpha_S_m_sparse, random_state)
    E_reconstructed_s = functions.reconstruct_signal(S_m_s, U_s, n_synergies_sparse)
    S_m_scaled_s = functions.scale_synergy_signal(S_m_s, emg_data)

    print(f"Shape of E_Reconstructed Sparse NMF: {E_reconstructed_s.shape}")
    print(f"Shape of S_m_s (Samples x Synergies): {S_m_s.shape}")
    print(f"Shape of U_s (Synergies x Muscles): {U_s.shape}")

    
    # Classical NMF
    S_m, U, E, reconstr_error = functions.apply_nmf(emg_data, n_synergies_classical, init, max_iter, l1_ratio_classical, alpha_S_m_classical, random_state)
    E_reconstructed = functions.reconstruct_signal(S_m, U, n_synergies_classical)
    S_m_scaled = functions.scale_synergy_signal(S_m, emg_data)
    
    print(f"Shape of E_Reconstructed Classical Sparse: {E_reconstructed.shape}")
    print(f"Shape of S_m (Samples x Synergies): {S_m.shape}")
    print(f"Shape of U (Synergies x Muscles): {U.shape}")
    

    # Plot both results side by side for comparison
    functions.plot_results_side_by_side(emg_data, E_reconstructed_s, E_reconstructed, S_m_scaled_s, S_m_scaled, U_s, U, n_synergies_sparse, n_synergies_classical)
    
    print("Session ended.")

else: 
        print("Session ended.")








