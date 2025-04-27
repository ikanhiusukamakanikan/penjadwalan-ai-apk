import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import datetime
import copy

# Set page config
st.set_page_config(page_title="AI Penjadwalan Peserta Didik", layout="wide")

# Cache data loaders for better performance
@st.cache_data
def load_data():
    """Load data from CSV files and preprocess"""
    try:
        # Load data from CSV
        peserta_didik = pd.read_csv("Data Dummy - Data Peserta Didik.csv")
        tanggal = pd.read_csv("Data Dummy - Data Pasien.csv")
        wahana = pd.read_csv("Data Dummy - Data Wahana.csv")
        
        # Preprocessing: Extract range of normal patients
        wahana['Min_Pasien'] = wahana['Jumlah Normal Pasien per Peserta Didik (batas)'].str.split('-').str[0].str.strip().astype(int)
        wahana['Max_Pasien'] = wahana['Jumlah Normal Pasien per Peserta Didik (batas)'].str.split('-').str[1].str.strip().astype(int)
        
        return peserta_didik, tanggal, wahana
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load data
peserta_didik, tanggal, wahana = load_data()
if peserta_didik is None or tanggal is None or wahana is None:
    st.error("Failed to load required data. Please check your data files.")
    st.stop()

# Initialize session state if not already initialized
if 'jadwal' not in st.session_state:
    st.session_state.jadwal = None
if 'metode' not in st.session_state:
    st.session_state.metode = "Constraint Planning"
if 'selected_peserta' not in st.session_state:
    st.session_state.selected_peserta = peserta_didik['ID Peserta'].tolist()
if 'selected_wahana' not in st.session_state:
    st.session_state.selected_wahana = wahana['Nama Wahana'].tolist()
if 'selected_date' not in st.session_state:
    # Default to May 1, 2025
    st.session_state.selected_date = "1/5/2025"

# Constraint Planning Algorithm with better station status preservation
def constraint_planning(peserta_list, wahana_df, patient_counts):
    """
    Improved constraint-based planning that checks all station status changes
    to make sure moves don't make normal stations become problematic
    """
    st.info("Running Constraint Planning algorithm...")
    progress_bar = st.progress(0)
    
    wahana_list = wahana_df['Nama Wahana'].tolist()
    n_wahana = len(wahana_list)
    
    # Initialize empty allocation (station -> list of participants)
    allocation = {wahana_name: [] for wahana_name in wahana_list}
    initial_allocation = {}

    # Step 1: Distribute participants evenly first
    wahana_index = 0
    for p in peserta_list:
        allocation[wahana_list[wahana_index]].append(p)
        wahana_index = (wahana_index + 1) % n_wahana
        
    initial_allocation = copy.deepcopy(allocation)
    progress_bar.progress(0.5)
    
    # Helper function to calculate station status
    def get_wahana_status(allocation, patient_counts):
        status = {}
        pasien_per_peserta = {}
        
        for i, wahana_name in enumerate(wahana_list):
            peserta_count = len(allocation[wahana_name]) or 1  # Avoid division by zero
            pasien_count = patient_counts[i]
            ratio = pasien_count / peserta_count
            pasien_per_peserta[wahana_name] = ratio
            
            min_normal = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Min_Pasien'].values[0]
            max_normal = wahana_df.loc[wahana_df['Nama Wahana'] == wahana_name, 'Max_Pasien'].values[0]
            
            if ratio < min_normal:
                status[wahana_name] = 'underutilized'
            elif ratio > max_normal:
                status[wahana_name] = 'overloaded'
            else:
                status[wahana_name] = 'normal'
        
        return status, pasien_per_peserta
    
    # Helper function to check if a move is beneficial
    def is_move_beneficial(old_status, new_status):
        # Count status changes
        improvements = 0
        deteriorations = 0
        
        for wahana, old_state in old_status.items():
            new_state = new_status[wahana]
            
            # If was problematic and now normal = improvement
            if old_state != 'normal' and new_state == 'normal':
                improvements += 1
            
            # If was normal and now problematic = deterioration
            elif old_state == 'normal' and new_state != 'normal':
                deteriorations += 2  # Penalize this more heavily
            
            # If problematic and still problematic but different = no clear benefit
            elif old_state != 'normal' and new_state != 'normal' and old_state != new_state:
                deteriorations += 1  # Slight penalty for changing problem types
        
        # The move is beneficial if it results in more improvements than deteriorations
        return improvements > deteriorations
    
    # Step 2: Optimize allocation through iterative balancing
    max_iterations = 100
    iteration = 0
    status_changed = True
    history = []
    
    # Save initial status
    initial_status, initial_ratio = get_wahana_status(allocation, patient_counts)
    history.append({
        'Iterasi': 0,
        'Status': copy.deepcopy(initial_status),
        'Rasio': copy.deepcopy(initial_ratio)
    })
    
    overloaded = [w for w, s in initial_status.items() if s == 'overloaded']
    underutilized = [w for w, s in initial_status.items() if s == 'underutilized']
    bothStatus = False    
    if overloaded and underutilized:
        bothStatus = True
    
    while status_changed or bothStatus and iteration < max_iterations:
        iteration += 1
        status_changed = False
        
        # Get current status
        current_status, current_ratio = get_wahana_status(allocation, patient_counts)
        
        # Create lists of stations by status
        overloaded = [w for w, s in current_status.items() if s == 'overloaded']
        underutilized = [w for w, s in current_status.items() if s == 'underutilized']
        normal = [w for w, s in current_status.items() if s == 'normal']

        # If no stations need adjustments, we're done
        if not overloaded and not underutilized:
            bothStatus = False
            break
        elif overloaded and underutilized:
            bothStatus = True
        else:
            bothStatus = False
        
        # Sort stations by how far they are from normal
        overloaded_scores = []
        for w in overloaded:
            idx = wahana_df.index[wahana_df['Nama Wahana'] == w].tolist()[0]
            max_pasien = wahana_df.loc[idx, 'Max_Pasien']
            score = (current_ratio[w] - max_pasien) / max_pasien
            overloaded_scores.append((w, score))

        underutilized_scores = []
        for w in underutilized:
            idx = wahana_df.index[wahana_df['Nama Wahana'] == w].tolist()[0]
            min_pasien = wahana_df.loc[idx, 'Min_Pasien']
            score = (min_pasien - current_ratio[w]) / min_pasien
            underutilized_scores.append((w, score))
        
        normal_scores = []
        for w in normal:
            idx = wahana_df.index[wahana_df['Nama Wahana'] == w].tolist()[0]
            min_pasien = wahana_df.loc[idx, 'Min_Pasien']
            max_pasien = wahana_df.loc[idx, 'Max_Pasien']
            # Calculate how far from boundaries (lower score = safer to move)
            current = current_ratio[w]
            # Distance to min boundary (normalized)
            dist_to_min = (current - min_pasien) / (max_pasien - min_pasien)
            # Distance to max boundary (normalized)
            dist_to_max = (max_pasien - current) / (max_pasien - min_pasien)
            # Use the smaller distance to boundary
            score = min(dist_to_min, dist_to_max)
            normal_scores.append((w, score))
        
        # Sort by most in need of adjustment
        overloaded_scores.sort(key=lambda x: x[1], reverse=True)
        underutilized_scores.sort(key=lambda x: x[1], reverse=True)
        # Sort normal stations by safety margin (LOWER score = SAFER to move)
        normal_scores.sort(key=lambda x: x[1])
        
        # Try all possible moves between stations
        
        # First try: Move from underutilized to overloaded
        for over_wahana, _ in overloaded_scores:
            move_successful = False
            
            # First try from underutilized stations
            for under_wahana, _ in underutilized_scores:
                if allocation[under_wahana]:
                    # Save current state for comparison
                    prev_status = current_status.copy()
                    
                    # Move first participant
                    peserta = allocation[under_wahana][0]
                    allocation[under_wahana].remove(peserta)
                    allocation[over_wahana].append(peserta)
                     
                    # Check if this move improves the situation
                    new_status, new_ratio = get_wahana_status(allocation, patient_counts)
                    
                    # Use the improved evaluation function
                    if is_move_beneficial(prev_status, new_status):
                        # Move successful, record changes
                        status_changed = True
                        move_successful = True
                        history.append({
                            'Iterasi': iteration,
                            'Status': copy.deepcopy(new_status),
                            'Rasio': copy.deepcopy(new_ratio)
                        })
                        break
                    else:
                        # Revert the move
                        allocation[over_wahana].remove(peserta)
                        allocation[under_wahana].append(peserta)
            
            # If no successful move from underutilized, try from normal stations
            if not move_successful and normal:
                for norm_wahana, score in normal_scores:
                    # Only consider normal stations with a good safety margin
                    if allocation[norm_wahana] and len(allocation[norm_wahana]) > 1:  # Ensure we don't empty normal stations
                        # Save current state for comparison
                        prev_status = current_status.copy()
                        
                        # Move first participant
                        peserta = allocation[norm_wahana][0]
                        allocation[norm_wahana].remove(peserta)
                        allocation[over_wahana].append(peserta)
                        
                        # Check if this move improves the situation
                        new_status, new_ratio = get_wahana_status(allocation, patient_counts)
                        
                        # Use the improved evaluation function
                        if is_move_beneficial(prev_status, new_status):
                            # Move successful, record changes
                            status_changed = True
                            move_successful = True
                            history.append({
                                'Iterasi': iteration,
                                'Status': copy.deepcopy(new_status),
                                'Rasio': copy.deepcopy(new_ratio)
                            })
                            break
                        else:
                            # Revert the move
                            allocation[over_wahana].remove(peserta)
                            allocation[norm_wahana].append(peserta)
        
        # If there are still underutilized stations, try to balance them with normal stations
        if underutilized and normal:
            for under_wahana, _ in underutilized_scores:
                move_successful = False
                
                for norm_wahana, score in normal_scores:
                    if allocation[norm_wahana] and len(allocation[norm_wahana]) > 1:  # Ensure we don't empty normal stations
                        # Save current state for comparison
                        prev_status = current_status.copy()
                        
                        # Move first participant
                        peserta = allocation[norm_wahana][0]
                        allocation[norm_wahana].remove(peserta)
                        allocation[under_wahana].append(peserta)
                        
                        # Check if this move improves the situation
                        new_status, new_ratio = get_wahana_status(allocation, patient_counts)
                        
                        # Use the improved evaluation function
                        if is_move_beneficial(prev_status, new_status):
                            # Move successful, record changes
                            status_changed = True
                            move_successful = True
                            history.append({
                                'Iterasi': iteration,
                                'Status': copy.deepcopy(new_status),
                                'Rasio': copy.deepcopy(new_ratio)
                            })
                            break
                        else:
                            # Revert the move
                            allocation[under_wahana].remove(peserta)
                            allocation[norm_wahana].append(peserta)
                
                if move_successful:
                    break
        
        # Update progress bar
        progress = 0.5 + 0.5 * (iteration / max_iterations)
        progress_bar.progress(progress)
    
    # Convert allocation result to desired return format
    assignment = {}
    for wahana_name, peserta_list in allocation.items():
        for p in peserta_list:
            assignment[p] = wahana_name
    
    # Display status change graph
    if history:
        normal_count = [sum(1 for s in h['Status'].values() if s == 'normal') for h in history]
        overloaded_count = [sum(1 for s in h['Status'].values() if s == 'overloaded') for h in history]
        underutilized_count = [sum(1 for s in h['Status'].values() if s == 'underutilized') for h in history]
        
        chart_data = pd.DataFrame({
            'Iterasi': list(range(1, len(history) + 1)),
            'Normal': normal_count,
            'Overloaded': overloaded_count,
            'Underutilized': underutilized_count
        })
        
        chart_data['Iterasi'] = pd.to_numeric(chart_data['Iterasi'], downcast='integer')

        st.write("### Grafik Perubahan Status Wahana:")
        st.line_chart(chart_data.set_index('Iterasi'))
    
    progress_bar.progress(1.0)
    st.success("Constraint Planning completed")

    return {
        'final': assignment,
        'initial': initial_allocation
    }

#Fungsi visualisasi untuk penempatan awal
def visualize_initial_allocation(initial_allocation, wahana_df, peserta_df, patient_counts=None):
    """Visualize the initial distribution with detailed status information"""
    
    # Convert to DataFrame
    initial_data = []
    for wahana_name, peserta_list in initial_allocation.items():
        for peserta in peserta_list:
            initial_data.append({
                'ID Peserta': str(peserta),  # Convert to string explicitly
                'Nama Wahana': wahana_name
            })
    
    initial_df = pd.DataFrame(initial_data)
    
    # Merge dengan data peserta untuk mendapatkan nama
    if peserta_df is not None:
        # Make sure both ID columns are strings
        peserta_df = peserta_df.copy()  # Create a copy to avoid modifying the original
        peserta_df['ID'] = peserta_df['ID Peserta'].astype(str).str.replace(',', '').str.strip()
        initial_df['ID Peserta'] = initial_df['ID Peserta'].astype(str).str.replace(',', '').str.strip()
        
        initial_df = pd.merge(
            initial_df,
            peserta_df[['ID', 'Nama Lengkap']],
            left_on='ID Peserta',
            right_on='ID',
            how='left'
        ).drop('ID', axis=1)
        initial_df.columns = ['ID Peserta', 'Nama Wahana', 'Nama Peserta']
    
    # Tampilkan daftar peserta dengan wahananya
    st.markdown("### Detail Penempatan Peserta")
    st.dataframe(initial_df)
    
    # Hitung jumlah peserta per wahana
    count_df = initial_df.groupby('Nama Wahana').size().reset_index(name='Jumlah Peserta')
    
    # Gabungkan dengan data wahana
    result_df = pd.merge(count_df, wahana_df, on='Nama Wahana')
    
    # Jika patient_counts tersedia, tambahkan informasi status
    if patient_counts is not None:
        # Ensure patient_counts is a list and correct length
        if isinstance(patient_counts, list) and len(patient_counts) == len(result_df):
            # Add patient counts
            result_df['Jumlah Pasien'] = patient_counts
            
            # Calculate patients per participant
            result_df['Pasien per Peserta'] = [
                patient_counts[wahana_df['Nama Wahana'].tolist().index(w)] / n if n > 0 else 0
                for w, n in zip(result_df['Nama Wahana'], result_df['Jumlah Peserta'])
            ]
            
            # Determine status
            result_df['Status'] = [
                'normal' if min_val <= p <= max_val else ('overloaded' if p > max_val else 'underutilized')
                for p, min_val, max_val in zip(
                    result_df['Pasien per Peserta'], 
                    result_df['Min_Pasien'], 
                    result_df['Max_Pasien']
                )
            ]
            
            # Round pasien per peserta for display
            result_df['Pasien per Peserta'] = result_df['Pasien per Peserta'].round(2)
            
            # Calculate statistics
            total_wahana = len(result_df)
            normal_count = sum(result_df['Status'] == 'normal')
            overloaded_count = sum(result_df['Status'] == 'overloaded')
            underutilized_count = sum(result_df['Status'] == 'underutilized')
    
    # Tampilkan tabel detail wahana
    st.markdown("### Detail Status Wahana")
    
    # Create a display DataFrame with all needed information
    display_df = result_df.copy()

    display_df['Pasien per Peserta'] = display_df['Pasien per Peserta'].round(0).astype(int)
    
    # Select columns based on whether patient_counts is available
    if patient_counts is not None:
        # Full version with status
        display_columns = [
            'Nama Wahana', 
            'Jumlah Peserta', 
            'Jumlah Pasien', 
            'Pasien per Peserta', 
            'Jumlah Normal Pasien per Peserta Didik (batas)', 
            'Min_Pasien', 
            'Max_Pasien',
            'Status'
        ]
        # Make sure all columns exist
        display_columns = [col for col in display_columns if col in display_df.columns]
    else:
        # Simplified version without status
        display_columns = [
            'Nama Wahana', 
            'Jumlah Peserta',
            'Jumlah Normal Pasien per Peserta Didik (batas)', 
            'Min_Pasien', 
            'Max_Pasien'
        ]
        # Make sure all columns exist
        display_columns = [col for col in display_columns if col in display_df.columns]
    
    # Display the table
    display_df = display_df[display_columns]
    
    # Apply color coding to Status column if it exists
    if 'Status' in display_df.columns:
        def color_status(val):
            colors = {'normal': 'green', 'overloaded': 'red', 'underutilized': 'orange'}
            color = colors.get(val, 'black')
            return f'background-color: {color}; color: white'
        
        st.dataframe(display_df.style.applymap(color_status, subset=['Status']))
    else:
        st.dataframe(display_df)

    # Display metrics
    st.markdown("### Statistik Status Wahana Awal")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Wahana Normal", value=f"{normal_count} ({normal_count/total_wahana*100:.1f}%)")
    with col2:
        st.metric(label="Wahana Overloaded", value=f"{overloaded_count} ({overloaded_count/total_wahana*100:.1f}%)")
    with col3:
        st.metric(label="Wahana Underutilized", value=f"{underutilized_count} ({underutilized_count/total_wahana*100:.1f}%)")

    st.markdown("---")
    
    return initial_df

def get_patient_counts(selected_wahana, selected_date):
    """Calculate estimated patient counts for selected stations and date"""
    # Parse selected date
    date_parts = selected_date.split('/')
    selected_date_obj = datetime.datetime(int(date_parts[2]), int(date_parts[1]), int(date_parts[0]))
    
    # Calculate dates for last 10 days for averaging
    date_range = []
    for i in range(10):
        prev_date = selected_date_obj - datetime.timedelta(days=i)
        prev_date_str = f"{prev_date.day}/{prev_date.month}/{prev_date.year}"
        date_range.append(prev_date_str)
    
    # Get average patient counts for the selected stations
    filtered_wahana = wahana[wahana['Nama Wahana'].isin(selected_wahana)]
    patient_counts = []
    
    for wahana_name in filtered_wahana['Nama Wahana']:
        count_sum = 0
        days_with_data = 0
        
        for date_str in date_range:
            date_data = tanggal[tanggal['Tanggal'] == date_str]
            if not date_data.empty and wahana_name in date_data.columns:
                count_sum += date_data[wahana_name].values[0]
                days_with_data += 1
        
        # Use average or default if no data
        if days_with_data > 0:
            patient_counts.append(count_sum / days_with_data)
        else:
            # Try to use estimated value or default to 50
            if 'Estimasi Pasien Harian' in filtered_wahana.columns:
                est_value = filtered_wahana.loc[filtered_wahana['Nama Wahana'] == wahana_name, 'Estimasi Pasien Harian'].values[0]
                patient_counts.append(est_value)
            else:
                patient_counts.append(50)  # Default value if no data
    
    # Convert to integers
    patient_counts = [int(x) if isinstance(x, str) else int(x) for x in patient_counts]
    return patient_counts

def generate_jadwal(metode, selected_peserta, selected_wahana, selected_date):
    """Generate schedule based on selected method"""
    filtered_wahana = wahana[wahana['Nama Wahana'].isin(selected_wahana)]
    patient_counts = get_patient_counts(selected_wahana, selected_date)
    
    if metode == "Constraint Planning":
        result = constraint_planning(selected_peserta, filtered_wahana, patient_counts)
        return {
            'final': result['final'],
            'initial': result['initial']
        }
    else:
        st.error(f"Method '{metode}' not implemented")
        return None

def visualize_jadwal(jadwal, wahana_df, patient_counts, peserta_df=None):
    """Visualize the schedule results with proper participant ID formatting and names"""
    if not jadwal:
        st.warning("No schedule to visualize")
        return
    
    # Convert schedule to DataFrame
    jadwal_df = pd.DataFrame(list(jadwal.items()), columns=['ID Peserta', 'Nama Wahana'])
    
    # Format ID Peserta - remove commas and clean up
    jadwal_df['ID Peserta'] = jadwal_df['ID Peserta'].astype(str).str.replace(',', '').str.strip()
    
    # If peserta data is available, merge to get names
    if peserta_df is not None:
        # Clean and format ID in peserta_df
        if 'ID Peserta' in peserta_df.columns:
            peserta_df['ID'] = peserta_df['ID Peserta'].astype(str).str.replace(',', '').str.strip()
        else:
            peserta_df['ID'] = peserta_df['ID'].astype(str).str.replace(',', '').str.strip()
        
        # Merge with participant data
        jadwal_df = pd.merge(
            jadwal_df,
            peserta_df[['ID', 'Nama Lengkap']],
            left_on='ID Peserta',
            right_on='ID',
            how='left'
        ).drop('ID', axis=1)
        
        # Reorder and rename columns
        jadwal_df = jadwal_df[['ID Peserta', 'Nama Lengkap', 'Nama Wahana']]
        jadwal_df.columns = ['ID Peserta', 'Nama Peserta', 'Nama Wahana']
    else:
        # If no participant data, just show IDs
        jadwal_df = jadwal_df[['ID Peserta', 'Nama Wahana']]
    
    # Count participants per station
    peserta_per_wahana = jadwal_df.groupby('Nama Wahana').size().reset_index(name='Jumlah Peserta')
    
    # Merge with station data
    result_df = pd.merge(peserta_per_wahana, wahana_df, on='Nama Wahana')
    
    # Calculate patients per participant
    result_df['Pasien per Peserta'] = [
        patient_counts[wahana_df['Nama Wahana'].tolist().index(w)] / n 
        for w, n in zip(result_df['Nama Wahana'], result_df['Jumlah Peserta'])
    ]
    
    # Determine status
    result_df['Status'] = [
        'normal' if min_val <= p <= max_val else ('overloaded' if p > max_val else 'underutilized')
        for p, min_val, max_val in zip(
            result_df['Pasien per Peserta'], 
            result_df['Min_Pasien'], 
            result_df['Max_Pasien']
        )
    ]
    
    #membulatkan nilai pasien per peserta
    result_df['Pasien per Peserta'] = result_df['Pasien per Peserta'].round(0).astype(int)

    # Calculate metrics
    total_wahana = len(result_df)
    normal_count = sum(result_df['Status'] == 'normal')
    overloaded_count = sum(result_df['Status'] == 'overloaded')
    underutilized_count = sum(result_df['Status'] == 'underutilized')
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Wahana Normal", value=f"{normal_count} ({normal_count/total_wahana*100:.1f}%)")
    with col2:
        st.metric(label="Wahana Overloaded", value=f"{overloaded_count} ({overloaded_count/total_wahana*100:.1f}%)")
    with col3:
        st.metric(label="Wahana Underutilized", value=f"{underutilized_count} ({underutilized_count/total_wahana*100:.1f}%)")
    
    # Plot distribution
    st.subheader("Distribusi Pasien per Peserta Didik")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Chart 1: Participants per station
    sns.barplot(x='Nama Wahana', y='Jumlah Peserta', hue='Status', data=result_df, ax=ax1)
    ax1.set_title('Jumlah Peserta Didik per Wahana')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    
    # Chart 2: Patients per participant with min/max limits
    sns.barplot(x='Nama Wahana', y='Pasien per Peserta', hue='Status', data=result_df, ax=ax2)
    for i, (_, row) in enumerate(result_df.iterrows()):
        ax2.plot([i-0.4, i+0.4], [row['Min_Pasien'], row['Min_Pasien']], 'k--', alpha=0.7)
        ax2.plot([i-0.4, i+0.4], [row['Max_Pasien'], row['Max_Pasien']], 'k--', alpha=0.7)
    ax2.set_title('Pasien per Peserta Didik')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show detailed schedule
    st.subheader("Detail Jadwal")
    st.dataframe(jadwal_df)
    
    # Station status summary
    st.subheader("Status Wahana")
    stats_df = result_df[['Nama Wahana', 'Jumlah Peserta', 'Pasien per Peserta', 'Status']].copy()
    stats_df['Pasien per Peserta'] = stats_df['Pasien per Peserta'].round(2)
    
    # Apply color coding
    status_colors = {'normal': 'green', 'overloaded': 'red', 'underutilized': 'orange'}
    def color_status(val):
        color = status_colors.get(val, 'black')
        return f'background-color: {color}; color: white'
    st.dataframe(stats_df.style.applymap(color_status, subset=['Status']))
    
    # Detailed assignments per station
    st.subheader("Detail Penugasan per Wahana")
    wahana_assignments = {
        wahana_name: [p for p, w in jadwal.items() if w == wahana_name]
        for wahana_name in wahana_df['Nama Wahana']
    }
    
    # Create display DataFrame
    max_peserta = max(len(v) for v in wahana_assignments.values())
    detailed_data = []
    
    for i in range(max_peserta):
        row = {}
        for wahana_name in wahana_assignments:
            participants = wahana_assignments[wahana_name]
            if i < len(participants):
                participant_id = str(participants[i]).replace(',', '').strip()
                if peserta_df is not None:
                    try:
                        participant_name = peserta_df.loc[
                            peserta_df['ID'].astype(str).str.replace(',', '').str.strip() == participant_id,
                            'Nama'
                        ].values[0]
                        row[wahana_name] = f"{participant_id} - {participant_name}"
                    except:
                        row[wahana_name] = participant_id
                else:
                    row[wahana_name] = participant_id
            else:
                row[wahana_name] = ''
        detailed_data.append(row)
    
    st.dataframe(pd.DataFrame(detailed_data))
    
    # Download option
    csv = jadwal_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="Download Jadwal sebagai CSV",
        data=csv,
        file_name="Jadwal_Peserta_Didik.csv",
        mime="text/csv",
    )

# UI STREAMLIT

st.title("Artificial Intelligence Penjadwalan Peserta Didik di Wahana Kesehatan")
st.markdown("""
Sistem ini menggunakan teknik klasik dalam AI yaitu Planning lebih tepatnya Constraint Planning untuk mengoptimalkan penjadwalan peserta didik 
di wahana kesehatan berdasarkan kebutuhan pasien.
""")

# Sidebar for configurations
st.sidebar.header("Konfigurasi")

# Select algorithm
metode = st.sidebar.selectbox(
    "Pilih Metode AI",
    ["Constraint Planning"],
    index=0
)

# Date selection
tanggal['Tanggal'] = pd.to_datetime(tanggal['Tanggal'], format='%d/%m/%Y')
last_date = tanggal['Tanggal'].max()
next_5_days = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]

all_dates = [d.strftime('%#d/%#m/%Y') for d in next_5_days]
selected_date = st.sidebar.selectbox("Pilih Tanggal", all_dates, index=all_dates.index("1/5/2025") if "1/5/2025" in all_dates else 0)

# Tab for selecting peserta didik and wahana
tab1, tab2, tab3 = st.tabs(["Peserta Didik", "Wahana", "Jadwal"])

with tab1:
    st.header("Pilih Peserta Didik")
    
    # Ensure session state for selected_peserta exists
    if 'selected_peserta' not in st.session_state:
        st.session_state.selected_peserta = []

    # Add range selection option
    selection_method = st.radio(
        "Cara Pemilihan Peserta",
        ["Pilih Individual", "Rentang NPM"]
    )
    
    if selection_method == "Pilih Individual":
        if st.button("Pilih Semua Peserta"):
            st.session_state.selected_peserta = peserta_didik['ID Peserta'].tolist()
        if st.button("Batal Pilih Semua Peserta"):
            st.session_state.selected_peserta = []
                
        # Individual selection
        selected_peserta = st.multiselect(
            "Pilih Peserta Didik",
            options=peserta_didik['ID Peserta'].tolist(),
            default=st.session_state.selected_peserta
        )
        
    elif selection_method == "Rentang NPM":
        # Get min and max NPM values
        min_npm = int(peserta_didik['ID Peserta'].min())
        max_npm = int(peserta_didik['ID Peserta'].max())
        
        # Range slider for NPM selection
        npm_range = st.slider(
            "Pilih Rentang NPM",
            min_value=min_npm,
            max_value=max_npm,
            value=(min_npm, min_npm + 10),
            step=1
        )
        
        # Get peserta in range
        selected_peserta = peserta_didik[
            (peserta_didik['ID Peserta'] >= npm_range[0]) & 
            (peserta_didik['ID Peserta'] <= npm_range[1])
        ]['ID Peserta'].tolist()
    
    st.session_state.selected_peserta = selected_peserta

    # Show selected participants
    if selected_peserta:
        st.markdown(f"**{len(selected_peserta)} Peserta Dipilih**")
        selected_peserta_df = peserta_didik[peserta_didik['ID Peserta'].isin(selected_peserta)]
        temp_df = selected_peserta_df.copy()
        temp_df['ID Peserta'] = temp_df['ID Peserta'].astype(str)
        st.dataframe(temp_df, use_container_width=True)

with tab2:
    st.header("Pilih Wahana")
    
    # Add selection method
    wahana_select_method = st.radio(
        "Cara Pemilihan Wahana",
        ["Pilih Individual", "Berdasarkan Spesialis", "Jumlah Random"]
    )
    
    if wahana_select_method == "Pilih Individual":
        # Filter options
        spesialis_wahana_filter = st.multiselect(
            "Filter berdasarkan Spesialis Penyakit",
            sorted(wahana['Spesialist Penyakit'].unique().tolist()),
            default=sorted(wahana['Spesialist Penyakit'].unique().tolist())
        )
        
        # Apply filters
        filtered_wahana = wahana[wahana['Spesialist Penyakit'].isin(spesialis_wahana_filter)]
        
        # Filter session state untuk menghapus wahana yang tidak lagi dalam daftar opsi
        available_wahana = filtered_wahana['Nama Wahana'].tolist()
        if 'selected_wahana' in st.session_state:
            st.session_state.selected_wahana = [w for w in st.session_state.selected_wahana if w in available_wahana]
        
        if st.button("Pilih Semua Wahana"):
            st.session_state.selected_wahana = filtered_wahana['Nama Wahana'].tolist()
        if st.button("Batal Pilih Semua Wahana"):
            st.session_state.selected_wahana = []
        
        # Show multi-select for wahana
        selected_wahana = st.multiselect(
            "Pilih Wahana Kesehatan",
            options=filtered_wahana['Nama Wahana'].tolist(),
            default=st.session_state.selected_wahana
        )
        
    elif wahana_select_method == "Berdasarkan Spesialis":
        # Select specialists
        specialists = st.multiselect(
            "Pilih Spesialis Penyakit",
            sorted(wahana['Spesialist Penyakit'].unique().tolist()),
            default=[wahana['Spesialist Penyakit'].unique()[0]]
        )
        
        # Get all wahana from selected specialists
        selected_wahana = wahana[wahana['Spesialist Penyakit'].isin(specialists)]['Nama Wahana'].tolist()
        
    else:  # Random number
        # Select random number of wahana
        num_wahana = st.slider(
            "Jumlah Wahana", 
            min_value=1, 
            max_value=len(wahana), 
            value=min(5, len(wahana))
        )
        
        # Get random wahana
        if st.button("Acak Wahana"):
            all_wahana = wahana['Nama Wahana'].tolist()
            selected_indices = np.random.choice(len(all_wahana), size=num_wahana, replace=False)
            selected_wahana = [all_wahana[i] for i in selected_indices]
        else:
            selected_wahana = st.session_state.selected_wahana if 'selected_wahana' in st.session_state else []
    
    st.session_state.selected_wahana = selected_wahana
    
    # Show details of selected wahana
    if selected_wahana:
        st.markdown(f"**{len(selected_wahana)} Wahana Dipilih**")
        selected_wahana_df = wahana[wahana['Nama Wahana'].isin(selected_wahana)]
        st.dataframe(selected_wahana_df, use_container_width=True)

with tab3:
    # Show warning if no selections
    if not st.session_state.selected_peserta:
        st.warning("Silakan pilih peserta didik di tab Peserta Didik")
    if not st.session_state.selected_wahana:
        st.warning("Silakan pilih wahana di tab Wahana")
    
    # Generate initial allocation (without optimization) when participants and stations are selected
    if st.session_state.selected_peserta and st.session_state.selected_wahana:
        st.session_state.metode = metode
        st.session_state.selected_date = selected_date
        
        # Create initial allocation preview only when button is clicked
        if st.button("Buat Penempatan Awal", type="secondary"):
            # Build initial allocation (without optimization)
            wahana_list = st.session_state.selected_wahana
            n_wahana = len(wahana_list)
            
            # Initialize empty allocation
            preview_allocation = {wahana_name: [] for wahana_name in wahana_list}
            
            # Distribute participants evenly
            wahana_index = 0
            for p in st.session_state.selected_peserta:
                preview_allocation[wahana_list[wahana_index]].append(p)
                wahana_index = (wahana_index + 1) % n_wahana
            
            st.session_state.preview_allocation = preview_allocation
        
        # Display initial allocation preview only if it exists
        if 'preview_allocation' in st.session_state:
            st.subheader("Penempatan Awal Sebelum Optimasi")
            try:
                # Filter wahana to only selected ones
                filtered_wahana = wahana[wahana['Nama Wahana'].isin(st.session_state.selected_wahana)]
                
                # Get patient counts
                patient_counts = get_patient_counts(st.session_state.selected_wahana, selected_date)
                
                # Display the visualization with patient counts
                visualize_initial_allocation(
                    st.session_state.preview_allocation,
                    filtered_wahana,
                    peserta_didik,
                    patient_counts=patient_counts
                )
            except Exception as e:
                st.error(f"Error menampilkan preview: {e}")
        
        # Button to generate final schedule
        if st.button("Buat Jadwal Optimasi", type="primary"):
            with st.spinner(f"Membuat jadwal dengan metode {metode}..."):
                start_time = time.time()
                result = generate_jadwal(metode, st.session_state.selected_peserta, 
                                      st.session_state.selected_wahana, selected_date)
                end_time = time.time()
                
                st.session_state.jadwal = result['final']
                st.session_state.initial_allocation = result['initial']
                
                st.success(f"Jadwal berhasil dibuat dalam {end_time - start_time:.2f} detik!")
    
    # Display optimized jadwal if available
    if 'jadwal' in st.session_state and st.session_state.jadwal:
        st.header("Hasil Optimasi Jadwal")
        
        # Filter wahana to only selected ones
        filtered_wahana = wahana[wahana['Nama Wahana'].isin(st.session_state.selected_wahana)]
        
        # Get patient counts
        patient_counts = get_patient_counts(st.session_state.selected_wahana, selected_date)
        
        # Display the visualization with peserta_df parameter
        visualize_jadwal(
            st.session_state.jadwal, 
            filtered_wahana, 
            patient_counts,
            peserta_df=peserta_didik  # Pass peserta data to include names
        )


# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Copyright © 2025, 140810230011, 140810230031, 140810230045, 140810230057. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
