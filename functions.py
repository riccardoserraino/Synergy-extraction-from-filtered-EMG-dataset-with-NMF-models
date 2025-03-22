'''This .py includes all useful functions for the semg extraction and analysis'''

import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

###################################################################################################################################################################################
# Loads EMG data from a ROS .bag file
def load_emg_data(bag_path, topic_name):
    timestamps = []
    emg_data = []
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, rawdata, timestamp in bag.read_messages(topics=[topic_name]):
            emg_data.append(rawdata.data)  # Extract EMG RMS values
            timestamps.append(timestamp.to_sec())  # Convert timestamp to seconds
            # Check if emg_data is shaped correctly
    # Convert the list to a NumPy array after all data is appended
    emg_data = np.array(emg_data)
    # Convert the list to a NumPy array after all data is appended
    timestamps = np.array(timestamps)        
    if emg_data.shape[0] < emg_data.shape[1]:  # If (n_muscles, n_samples), transpose it
        emg_data = emg_data.T  # Now it’s (n_samples, n_muscles)
    return emg_data, timestamps


###################################################################################################################################################################################
# Loads EMG data from combined ROS .bag file
def load_combined_emg_data(selected_paths, topic_name):
    # Initialize empty lists for EMG data and timestamps
    emg_data_combined = []
    timestamps_combined = []

    for bag_path in selected_paths:
        # Load EMG data and timestamps
        emg_data, timestamps = load_emg_data(bag_path, topic_name)
        
        # Check and reshape data if necessary
        if emg_data.shape[0] < emg_data.shape[1]:
            emg_data = emg_data.T
        
        # Append data to the lists
        emg_data_combined.append(emg_data)
        timestamps_combined.append(timestamps)

    # Concatenate data into single arrays
    emg_data_combined = np.vstack(emg_data_combined)
    timestamps_combined = np.concatenate(timestamps_combined)

    return emg_data_combined, timestamps_combined


###################################################################################################################################################################################
# Applies Sparse Non-Negative Matrix Factorization (NMF) to extract synergies
def apply_nmf(emg_data, n_components, init, max_iter, l1_ratio, alpha_W, random_state):
    nmf = NMF(n_components=n_components, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state) # Setting Sparse NMF parameters
    W = nmf.fit_transform(emg_data)  # Synergy activations
    H = nmf.components_  # Muscle patterns
    # Transpose W and H to match the correct shapes if needed
    if W.shape[0] != emg_data.shape[0]:
        W = W.T  # Ensure W has shape (n_samples, n_synergies)
    if H.shape[0] != n_components:
        H = H.T  # Ensure H has shape (n_synergies, n_muscles)
    Z = np.dot(W, H)  # Reconstructed signal
    rec_error = nmf.reconstruction_err_ # Reconstruction error
    return W, H, Z, rec_error


###################################################################################################################################################################################
# Reconstructs the EMG signal using a specified number of synergies
def reconstruct_signal(W, H, selected_synergies):
    W_selected = W[:, :selected_synergies]
    H_selected = H[:selected_synergies, :]
    Z_reconstructed = np.dot(W_selected, H_selected)
    return Z_reconstructed


###################################################################################################################################################################################
# Plots the residual error (original - reconstructed) over time for each muscle channel.
def plot_residuals(emg_data, n_synergies, init, max_iter, l1_ratio, alpha_W, random_state, alpha=0.7, use_subplots=False):
    W, H, Z, rec_err = apply_nmf(emg_data, n_synergies, init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state)
    Z_reconstructed = reconstruct_signal(W, H, n_synergies)

    # Compute errors signal
    error_signal = emg_data - Z_reconstructed

    n_muscles = error_signal.shape[1]  # Number of muscle channels

    if use_subplots:
        # Use subplots for better visualization
        fig, axes = plt.subplots(n_muscles, 1, figsize=(10, 2 * n_muscles), squeeze=False)
        fig.suptitle("Residual Error Over Time")
        for i in range(n_muscles):
            axes[i, 0].plot(error_signal[:, i], label=f"Muscle {i+1}", alpha=alpha)
            axes[i, 0].legend()
        axes[-1, 0].set_xlabel("Time (samples)")
    else:
        # Plot all channels in a single figure
        plt.figure(figsize=(8, 6))
        for i in range(n_muscles):
            plt.plot(error_signal[:, i], label=f"Muscle {i+1}", alpha=alpha)
        plt.title("Residual Error Over Time")
        plt.xlabel("Time (samples)")
        plt.ylabel("Residual Error")
        plt.legend()

    plt.tight_layout()
    plt.show()


###################################################################################################################################################################################
# Plots the residual errors side by side for classical NMF and sparse NMF.
def plot_residuals_side_by_side(emg_data, Z_reconstructed_classical, Z_reconstructed_sparse, alpha=0.7):


    # Compute residuals for classical NMF
    error_signal_classical = emg_data - Z_reconstructed_classical

    # Compute residuals for sparse NMF
    error_signal_sparse = emg_data - Z_reconstructed_sparse

    n_muscles = emg_data.shape[1]  # Number of muscle channels

    # Create a figure with two subplots side by side
    plt.figure(figsize=(12, 6))

    # Plot residuals for classical NMF
    plt.subplot(1, 2, 1)
    for i in range(n_muscles):
        plt.plot(error_signal_classical[:, i], label=f"Muscle {i+1}", alpha=alpha)
    plt.title("Residual Error Over Time (Classical NMF)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid(True)

    # Plot residuals for sparse NMF
    plt.subplot(1, 2, 2)
    for i in range(n_muscles):
        plt.plot(error_signal_sparse[:, i], label=f"Muscle {i+1}", alpha=alpha)
    plt.title("Residual Error Over Time (Sparse NMF)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

###################################################################################################################################################################################
# Scales the synergy signal S_m to match the range of the original EMG signal for a better visual
def scale_synergy_signal(W, emg_data):
    emg_min = np.min(emg_data)
    emg_max = np.max(emg_data)
    W_min = np.min(W)
    W_max = np.max(W)
    W_scaled = ((W - W_min) / (W_max - W_min)) * (emg_max - emg_min) + emg_min
    W_scaled = np.maximum(W_scaled, 0)  # Ensures W_scaled is non-negative
    return W_scaled


###################################################################################################################################################################################
# Function to plot EMG data
def plot_emg_data(timestamps, emg_data, title):
    plt.figure(figsize=(10, 8))
    plt.plot(emg_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('EMG RMS [mV]')
    plt.title(title)
    plt.grid()
    plt.show()


###################################################################################################################################################################################
# Function to calculate the VAF to determine the optimal number of synergies for the Sparse NMF
def compute_vaf(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_W, random_state):
    VAF_values = []

    for n in range(1, max_synergies + 1):
        W, H, Z, rec_err = apply_nmf(emg_data, n, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state)
        VAF = 1 - np.sum((emg_data - Z) ** 2) / np.sum(emg_data ** 2)
        VAF_values.append(VAF)
        print(f"VAF for {n} synergies: {VAF:.4f}")

    plt.plot(range(1, max_synergies+1), VAF_values, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('VAF')
    plt.title('VAF vs Number of Synergies')
    plt.show()
        
    return VAF_values


###################################################################################################################################################################################
# Function to plot the reconstruction error (Frobenius Norm) based on different number of synergies
def compute_rec_err(emg_data, max_synergies, l1_ratio, init, max_iter, alpha_W, random_state):
    reconstruction_errors = []
    percentage_errors = []

    # Compute the Frobenius norm of the original data for relative error calculation
    original_norm = np.linalg.norm(emg_data, 'fro')

    for n in range(1, max_synergies + 1):
        W, H, Z, reconstruction_error = apply_nmf(emg_data, n_components=n, init=init, max_iter=max_iter, l1_ratio=l1_ratio, alpha_W=alpha_W, random_state=random_state)
        # Calculate the reconstruction error
        reconstruction_errors.append(reconstruction_error)
        print(f"Reconstruction error for {n} synergies: {reconstruction_error:.4f}")

        # Calculate the percentage error
        percentage_error = (reconstruction_error / original_norm) * 100
        percentage_errors.append(percentage_error)
        print(f"Percentage error for {n} synergies: {percentage_error:.2f}%")
    
    # Plot reconstruction errors vs number of synergies
    plt.figure(figsize=(12, 6))

    # Plot Frobenius norm error
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_synergies + 1), reconstruction_errors, marker='o', color='b')
    plt.xlabel('Number of Synergies')
    plt.ylabel('Reconstruction Error (Frobenius Norm)')
    plt.title('Reconstruction Error vs Number of Synergies')

    # Plot percentage error
    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_synergies + 1), percentage_errors, marker='o', color='r')
    plt.xlabel('Number of Synergies')
    plt.ylabel('Percentage Error (%)')
    plt.title('Percentage Error vs Number of Synergies')

    plt.tight_layout()
    plt.show()

    return reconstruction_errors, percentage_errors


###################################################################################################################################################################################
# Function to plot the reconstruction error (Frobenius Norm) based on different number of synergies
def compute_rec_err_side_by_side(emg_data, max_synergies, l1_ratio_c, l1_ratio_s, init, max_iter, alpha_W_c, alpha_W_s, random_state):
    reconstruction_errors = []
    reconstruction_errors_s = []

    # Classical NMF REC errors
    for n in range(1, max_synergies + 1):
        W, H, Z, reconstruction_error = apply_nmf(emg_data, n_components=n, init=init, max_iter=max_iter, l1_ratio=l1_ratio_c, alpha_W=alpha_W_c, random_state=random_state)
        reconstruction_errors.append(reconstruction_error)
        print(f"Reconstruction error for {n} synergies: {reconstruction_error:.4f}")

    # Sparse NMF REC errors
    for n in range(1, max_synergies + 1):
        W, H, Z, reconstruction_error_s = apply_nmf(emg_data, n_components=n, init=init, max_iter=max_iter, l1_ratio=l1_ratio_s, alpha_W=alpha_W_s, random_state=random_state)
        reconstruction_errors_s.append(reconstruction_error_s)
        print(f"Reconstruction error for {n} synergies: {reconstruction_error_s:.4f}")

    #Plot the results
    plt.figure(figsize=(8, 6))

    # Classical NMF Plot
    plt.subplot(2, 1, 2)
    plt.plot(range(1, max_synergies+1), reconstruction_errors, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('Reconstruction Error (Frobenius Norm)')
    plt.title('Reconstruction Error vs Number of Synergies (Classical NMF)')

    # Sparse NMF Plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, max_synergies+1), reconstruction_errors_s, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('Reconstruction Error (Frobenius Norm)')
    plt.title('Reconstruction Error vs Number of Synergies (Sparse NMF)')

    plt.tight_layout()
    plt.show()

    return reconstruction_errors


###################################################################################################################################################################################
# Function to calculate the VAF to determine the optimal number of synergies for the NMF approach
def compute_vaf_side_by_side(emg_data, max_synergies, l1_ratio_c, l1_ratio_s, init, max_iter, alpha_W_c, alpha_W_s, random_state):
    VAF_values = []
    VAF_values_s = []
    
    # Classical NMF VAF Analysis
    for n in range(1, max_synergies + 1):
        W, H, Z, rec_err = apply_nmf(emg_data, n, init=init, max_iter=max_iter, l1_ratio=l1_ratio_c, alpha_W=alpha_W_c, random_state=random_state)
        VAF = 1 - np.sum((emg_data - Z) ** 2) / np.sum(emg_data ** 2)
        VAF_values.append(VAF)
        print(f"VAF for {n} synergies: {VAF:.4f}")
    
    # Sparse NMF VAF Analysis
    for n in range(1, max_synergies + 1):
        W_s, H_s, Z_s, rec_err = apply_nmf(emg_data, n, init=init, max_iter=max_iter, l1_ratio=l1_ratio_s, alpha_W=alpha_W_s, random_state=random_state)
        VAF_s = 1 - np.sum((emg_data - Z_s) ** 2) / np.sum(emg_data ** 2)
        VAF_values_s.append(VAF_s)
        print(f"VAF for {n} synergies: {VAF_s:.4f}")

    #Plot the results
    plt.figure(figsize=(8, 6))

    # Classical NMF Plot
    plt.subplot(2, 1, 2)
    plt.plot(range(1, max_synergies+1), VAF_values, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('VAF (Classical NMF)')
    plt.title('VAF vs Number of Synergies (Classical NMF)')

    # Sparse NMF Plot
    plt.subplot(2, 1, 1)
    plt.plot(range(1, max_synergies+1), VAF_values_s, marker='o')
    plt.xlabel('Number of Synergies')
    plt.ylabel('VAF (Sparse NMF)')
    plt.title('VAF vs Number of Synergies (Sparse NMF)')

    plt.tight_layout()
    plt.show()

    return VAF_values


###################################################################################################################################################################################
# Plots the original and reconstructed EMG signals, synergy activations, and muscle patterns
def plot_results(emg_data, Z_reconstructed, W_scaled, H, selected_synergies):
    plt.figure(figsize=(10, 8))
    
    # Original EMG Signal
    plt.subplot(4, 1, 1)
    plt.plot(emg_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('EMG RMS [mV]')
    plt.title('Original EMG Signal')
    
    # Reconstructed EMG Signal
    plt.subplot(4, 1, 2)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.xlabel('Time (samples)')
    plt.ylabel('Reconstructed EMG RMS [mV]')
    plt.title(f'Reconstructed EMG Signal ({selected_synergies} Synergies)')
    
    # Synergy Activations (W matrix)
    plt.subplot(4, 1, 3)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], label=f'Synergy {i+1}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Activation')
    plt.title('Activations of the Synergies')
    plt.legend()
    
    # Muscle Patterns (H matrix)
    plt.subplot(4, 1, 4)
    for i in range(selected_synergies):
        plt.plot(H[i, :], label=f'Synergy {i+1}')
    plt.xlabel('EMG Channels')
    plt.ylabel('Activation')
    plt.title('Muscle Patterns')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


###################################################################################################################################################################################
# Plot results of the signals reconstruction using Classical NMF approach or Sparse NMF approacch
def plot_results_side_by_side(emg_data, Z_reconstructed, Z_reconstructed_1, W_scaled, W_scaled_1, H, H_1, selected_synergies, selected_synergies_1):
    plt.figure(figsize=(10, 8))
    
    
    # Column 1: Sparse NMF Results
    # Original EMG Signal
    plt.subplot(4, 2, 1)
    plt.plot(emg_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('EMG RMS [mV]')
    plt.title('Original EMG Signal')
    
    # Reconstructed EMG Signal (Sparse NMF)
    plt.subplot(4, 2, 3)
    plt.plot(Z_reconstructed, linestyle='--')
    plt.xlabel('Time (samples)')
    plt.ylabel('Reconstructed EMG RMS [mV]')
    plt.title(f'Reconstructed EMG Signal (Sparse NMF, {selected_synergies} Synergies)')
    
    # Synergy Activations (W matrix) for Sparse NMF
    plt.subplot(4, 2, 5)
    for i in range(selected_synergies):
        plt.plot(W_scaled[:, i], label=f'Synergy {i+1}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Activation')
    plt.title('Activations of the Synergies (Sparse NMF)')
    plt.legend()
    
    # Muscle Patterns (H matrix) for Sparse NMF
    plt.subplot(4, 2, 7)
    for i in range(selected_synergies):
        plt.plot(H[i, :], label=f'Synergy {i+1}')
    plt.xlabel('EMG Channels')
    plt.ylabel('Activation')
    plt.title('Muscle Patterns (Sparse NMF)')
    plt.legend()
    
    # Column 2: Classical NMF Results
    # Original EMG Signal (same as Column 1, but can be omitted if redundant)
    plt.subplot(4, 2, 2)
    plt.plot(emg_data)
    plt.xlabel('Time (samples)')
    plt.ylabel('EMG RMS [mV]')
    plt.title('Original EMG Signal')
    
    # Reconstructed EMG Signal (Classical NMF)
    plt.subplot(4, 2, 4)
    plt.plot(Z_reconstructed_1, linestyle='--')
    plt.xlabel('Time (samples)')
    plt.ylabel('Reconstructed EMG RMS [mV]')
    plt.title(f'Reconstructed EMG Signal (Classical NMF, {selected_synergies_1} Synergies)')
    
    # Synergy Activations (W matrix) for Classical NMF
    plt.subplot(4, 2, 6)
    for i in range(selected_synergies_1):
        plt.plot(W_scaled_1[:, i], label=f'Synergy {i+1}')
    plt.xlabel('Time (samples)')
    plt.ylabel('Activation')
    plt.title('Activations of the Synergies (Classical NMF)')
    plt.legend()
    
    # Muscle Patterns (H matrix) for Classical NMF
    plt.subplot(4, 2, 8)
    for i in range(selected_synergies_1):
        plt.plot(H_1[i, :], label=f'Synergy {i+1}')
    plt.xlabel('EMG Channels')
    plt.ylabel('Activation')
    plt.title('Muscle Patterns (Classical NMF)')
    plt.legend()



    plt.tight_layout()
    plt.show()


###################################################################################################################################################################################
# Plots the selected synergies overlaid on EMG signals.
def plot_synergies(timestamps, emg_data, synergy_data, selected_synergies):
    emg_min, emg_max = np.min(emg_data), np.max(emg_data)  # Find min/max of EMG signals

    plt.figure(figsize=(10, 8))

    # Plot all EMG signals in light red
    for i in range(emg_data.shape[1]):
        plt.plot(timestamps, emg_data[:, i], alpha=0.3, color='red', label='EMG Signals' if i == 0 else "")

    # Define colors dynamically
    colors = ['blue', 'green', 'cyan', 'orange', 'yellow', 'magenta', 'brown', 'purple']
    synergy_colors = {f'Synergy_{i+1}': colors[i % len(colors)] for i in range(len(synergy_data))}

    # Plot selected synergies
    for syn in selected_synergies:
        if syn in synergy_data:
            synergy_signal = synergy_data[syn]
            synergy_min, synergy_max = np.min(synergy_signal), np.max(synergy_signal)
            synergy_scaled = ((synergy_signal - synergy_min) / (synergy_max - synergy_min)) * (emg_max - emg_min) + emg_min
            plt.plot(timestamps, synergy_scaled, color=synergy_colors[syn], label=f'{syn}')

    # Labels and formatting
    plt.xlabel('Time (samples)')
    plt.ylabel('Activation')
    plt.title('Synergy Overlaid on EMG Signals')
    plt.legend()
    plt.tight_layout()
    plt.show()


###################################################################################################################################################################################
# Function to store synergy activations in a dictionary
def store_synergy(S_m, n_synergies):
    return {f'Synergy_{i+1}': S_m[:, i] for i in range(n_synergies)}


###################################################################################################################################################################################
# Asks the user whether to plot one or multiple synergies.
def get_synergy_selection(n_synergies):
    plot_choice = input("Do you want to plot one synergy or multiple?: ")
    while plot_choice not in ['1', 'multiple']:
        plot_choice = input("Invalid input. Please enter '1' for one synergy or 'multiple' for more than 1: ")

    if plot_choice == '1':
        valid_options = [str(i+1) for i in range(n_synergies)]
        syn_to_plot = input(f"Enter the synergy number to plot ({', '.join(valid_options)}): ")
        while syn_to_plot not in valid_options:
            syn_to_plot = input(f"Invalid input. Please enter a valid synergy number ({', '.join(valid_options)}): ")
        return [f'Synergy_{syn_to_plot}']
    
    return [f'Synergy_{i+1}' for i in range(n_synergies)]  # Plot all available synergies


###################################################################################################################################################################################
# Make the user select the number of synergies to use for reconstructing the signal
def get_synergy_count():
    print("Select the number of synergies you want to use to reconstruct the signal:")
    
    while True:
        choice = input("Choose a number between 1 and 8: ")
        
        if choice in [str(i) for i in range(1, 9)]:
            return int(choice)  # Return the valid number
        
        print("Invalid choice. Please select a number between 1 and 8.")


###################################################################################################################################################################################
# Ask the user for the NMF approach desired
def get_nmf_approach(l1_ratio_classical, alpha_S_m_classical, l1_ratio_sparse, alpha_S_m_sparse):
    print("Select the NMF approach you want to use:")
    
    while True:
        choice = input("Enter 1 for Classical NMF or 2 for Sparse NMF: ")
        
        if choice == '1':
            return l1_ratio_classical, alpha_S_m_classical  # Return Classical NMF parameters
        elif choice == '2':
            return l1_ratio_sparse, alpha_S_m_sparse  # Return Sparse NMF parameters
        
        print("Invalid input. Please select between: 1 for Classical NMF, 2 for Sparse NMF.")


###################################################################################################################################################################################
# Ask the user if he wants to proceed with a NMF comparison
def ask_for_comparison():
    print("Do you want to compare the NMF approaches?")

    while True:
        choice = input("Enter 'y' for Yes or 'n' for No: ").strip().lower()
        
        if choice in ['y', 'n']:
            return choice 
        
        print("Invalid input. Please enter 'y' for Yes or 'n' for No.")



###################################################################################################################################################################################
# Ask user what dataset to combine and in what order, used for multiple gesture analysis
def select_datasets(dataset_options):
    print("Select the dataset you want to combine:")
    
    while True:
        selected_datasets = input("Enter the dataset numbers (from 1 to 9) in the desired order (e.g., '1 2 3' or '2 1'): ").split()
        
        # Validate input
        selected_paths = [dataset_options[num] for num in selected_datasets if num in dataset_options]
        
        if selected_paths:
            return selected_paths  # Return valid dataset paths
        
        print("Invalid selection. Please enter valid dataset numbers.")


###################################################################################################################################################################################
# Ask user what dataset to analysis, single gesture analysis
def select_dataset(dataset_options):
    print("Select a dataset:")

    while True:
        choice = input("Enter a dataset number (from 1 to 9): ").strip()
        
        if choice in dataset_options:
            return dataset_options[choice]  # Return the corresponding dataset path
        
        print("Invalid selection. Please enter a valid dataset number. (You may have put a space!)")
    

###################################################################################################################################################################################
