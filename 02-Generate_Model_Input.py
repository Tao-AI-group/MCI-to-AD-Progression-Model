import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import pickle as pkl
from collections import defaultdict
from csv import reader, writer
from typing import Dict, Set, Tuple, List, Optional
from sklearn.model_selection import train_test_split

np.random.seed(41)


# Constants

AGE_RANGES = [
    (40, 45), (45, 50), (50, 55), (55, 60), (60, 65),
    (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95)
]

# Mapping of prediction windows to their corresponding label column indices
WINDOW_LABEL_MAP = {
    365: 11,   # 1-365 days label
    730: 12,   # 1-730 days label
    1095: 13,  # 1-1095 days label
    1825: 14,  # 1-1825 days label
}

# File type configurations
FILE_CONFIGS = {
    'age': {'cols': (1, 0, 16), 'prefix': 'A_', 'source': 'encounters'},
    'gender': {'cols': (1, 0, 4), 'prefix': 'G_', 'source': 'encounters'},
    'diagnosis': {'cols': (0, 2, 3), 'prefix': '', 'source': 'diagnosis'},
    'medication': {'cols': (0, 2, 3), 'prefix': '', 'source': 'medications'},
    'procedure': {'cols': (0, 2, 3), 'prefix': '', 'source': 'procedures'}
}

def get_common_path(relative_path: str) -> str:
    """
    Generates a full path to a file relative to the directory containing this script.
    """
    cur_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(cur_path, relative_path)
    return file_path

def get_qualified_patients(input_path: str, prediction_start_gap: int) -> tuple[set[str], dict]:
   """
   Identifies qualified patients based on their last record time relative to prediction start day.
   A patient is qualified if their last record time is >= prediction start day.
   Returns (qualified_patients, patient_daygaps).
   """
   qualified_patients = set()
   all_patients = set()
   patient_daygaps = defaultdict(list)

   try:
       with open(input_path, 'r') as input_file:
           csv_reader = reader(input_file)
           header = next(csv_reader)
           
           if not header:
               raise ValueError("Empty input file")

           for row in csv_reader:
               patient_id = row[1]  # Patient ID
               day_gap = row[15]    # Day gap for encounter
               
               patient_daygaps[patient_id].append(int(day_gap))
               all_patients.add(patient_id)

       for patient_id, gap_list in patient_daygaps.items():
           last_gap = max(gap_list)
           if last_gap >= prediction_start_gap:
               qualified_patients.add(patient_id)

       print(f"Patient Selection: {len(qualified_patients)} qualified out of {len(all_patients)} total "
              f"(start gap: {prediction_start_gap} days)")
       return qualified_patients, patient_daygaps

   except Exception as e:
        print(f"Error in get_qualified_patients: {str(e)}")
        raise

def get_qualified_input_encid_dic(outcome_input_path: str, qualified_patients: set[str], days_before_index: int) -> dict:
    """
    Retrieves qualified encounter IDs and their dates for patients before the index date.
    Returns dictionary of qualified encounter IDs and dates for patients.
    """
    pt_qualified_input_encid_dic = defaultdict(dict)
    
    try:
        with open(outcome_input_path, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)  # Skip header

            for row in csv_reader:
                patient_id = row[1]
                # Only process records for qualified patients
                if patient_id not in qualified_patients:
                    continue
                    
                day_gap = int(row[15])
                # Check if encounter is within the lookback(observation) window
                if -days_before_index <= day_gap < 0:
                    encounter_id = row[0]
                    encounter_date = row[21].split(' ')[0]
                    pt_qualified_input_encid_dic[patient_id][encounter_id] = encounter_date

        print(f'Found {len(pt_qualified_input_encid_dic)} patients with qualified encounters\n')
        return pt_qualified_input_encid_dic

    except Exception as e:
        print(f"Error in get_qualified_input_encid_dic: {str(e)}")
        raise

def get_case_control_set(
    outcome_input_path: str,
    qualified_patients: set[str],
    prediction_start_daygap: int,
    prediction_end_daygap: int,
    patient_encounters_daygaps_list_dic: Dict[str, List[str]]
) -> Tuple[Set[str], Set[str]]:
    
    """
    Classifies patients into case and control groups based on outcome labels within 
    a specified prediction window.
    """

    if prediction_start_daygap != 1 or prediction_end_daygap not in WINDOW_LABEL_MAP:
        raise ValueError(f"Invalid prediction window: start={prediction_start_daygap}, "
                      f"end={prediction_end_daygap}")

    # Initialize sets and lists for classification
    case_pt_set = set()
    control_pt_set = set()
    case_list = []
    label_column = WINDOW_LABEL_MAP[prediction_end_daygap]

    try:
        with open(outcome_input_path, 'r') as f:
            csv_reader = reader(f)
            next(csv_reader)

            for row in csv_reader:
                patient_id = row[1]
                if patient_id not in qualified_patients:
                    continue

                day_gap = int(row[15])
                if prediction_start_daygap <= day_gap <= prediction_end_daygap:
                    if row[label_column] == '1':
                        case_patients.add(patient_id)

        # Add remaining qualified patients to controls
        control_patients = qualified_patients - case_patients

        # Remove censored controls
        control_patients -= {
            patient_id for patient_id, daygap_list in patient_encounters_daygaps_list_dic.items()
            if patient_id in control_patients and max(daygap_list) <= prediction_end_daygap
        }

        print(f'Classification results:\n'
              f'  Cases: {len(case_patients)}\n'
              f'  Controls: {len(control_patients)}\n')

        return case_patients, control_patients

    except Exception as e:
        print(f"Error in get_case_control_set: {str(e)}")
        raise


def map_age_to_code(age: int) -> Optional[str]:
    """Maps age to predefined age group code."""
    age_ranges = [(40, 45), (45, 50), (50, 55), (55, 60), (60, 65),
                 (65, 70), (70, 75), (75, 80), (80, 85), (85, 90), (90, 95)]
    
    for i, (start, end) in enumerate(age_ranges):
        if start <= age < end:
            return f"{start}_{end-1}"
    return None

def get_specific_file_input_dic(
    file_type: str,
    input_file_path: str,
    qualified_patients: set[str],
    qualified_encounters: Dict[str, Dict[str, str]]
) -> List[List[str]]:
    """
    Extracts medical record features from specified file type.
    
    Args:
        file_type: Type of medical record to process
        input_file_path: Path to input CSV file
        qualified_patients: Set of qualified patient IDs
        qualified_encounters: Mapping of patient IDs to encounter data
    
    Returns:
        List of [patient_id, code, encounter_date] records
    """
    if file_type not in FILE_CONFIGS:
        raise ValueError(f"Invalid file type: {file_type}")
        
    config = FILE_CONFIGS[file_type]
    pat_idx, enc_idx, code_idx = config['cols']
    data_lists = []
    
    try:
        with open(input_file_path) as f:
            csv_reader = reader(f)
            next(csv_reader)  # Skip header
            
            for row in csv_reader:
                patient_id = row[pat_idx]
                if patient_id not in qualified_patients:
                    continue
                    
                enc_dict = qualified_encounters[patient_id]
                encounter_id = row[enc_idx]
                if encounter_id not in enc_dict:
                    continue
                
                # Process code based on file type
                if file_type == 'age':
                    age = int(float(row[code_idx]))
                    code = map_age_to_code(age)
                    if not code:
                        continue
                    code = f"{config['prefix']}{code}"
                else:
                    code = f"{config['prefix']}{row[code_idx]}"
                
                data_lists.append([patient_id, code, enc_dict[encounter_id]])
                
        print(f"Processed {len(data_lists)} {file_type} records")
        return data_lists
        
    except Exception as e:
        print(f"Error processing {file_type} data: {str(e)}")
        raise


def write_data_lists_to_tsv(data_lists: List[List[List[str]]], output_path: str) -> None:
    """
    Writes combined data lists to TSV file.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wt') as f:
            csv_writer = writer(f, delimiter='\t')
            csv_writer.writerow(['Pt_id', 'ICD', 'Time'])
            for feature_list in data_lists:
                csv_writer.writerows(feature_list)
        print(f"Wrote {sum(len(x) for x in data_lists)} records to {output_path}")
    except Exception as e:
        print(f"Error writing to {output_path}: {str(e)}")
        raise

def deduplicate_tsv(input_path: str, output_path: str) -> None:
    """
    Removes duplicate rows from TSV file.
    """
    try:
        df = pd.read_csv(input_path, sep='\t')
        initial_rows = len(df)
        df_dedup = df.drop_duplicates()
        df_dedup.to_csv(output_path, sep='\t', index=False)
        print(f"Deduplicated {initial_rows - len(df_dedup)} rows from {input_path}")
    except Exception as e:
        print(f"Error deduplicating {input_path}: {str(e)}")
        raise

def get_split_file_paths(base_dir: str) -> Tuple[str, str, str, str, str, str]:
    """
    Sets up file paths for train/test/validation splitting.
    
    Args:
        base_dir: Directory containing data files
        
    Returns:
        Tuple of (case_file, control_file, type_file, out_file, cls_type, pts_file_pre)
    """
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")
        
    return (
        os.path.join(base_dir, 'dedup_case.tsv'),
        os.path.join(base_dir, 'dedup_control.tsv'),
        'NA',  # type_file
        os.path.join(base_dir, 'ad'),  # out_file
        'binary',  # cls_type
        'NA'  # pts_file_pre
    )

def process_and_split_data(
    case_file: str,
    control_file: str,
    type_file: str,
    out_file: str,
    cls_type: str = 'binary',
    pts_file_pre: str = 'NA'
) -> None:
    """
    Processes case/control data and creates train/test/validation splits.
    """
    def load_and_process_data(file_path: str, label: int) -> pd.DataFrame:
        """Loads and processes a single data file."""
        data = pd.read_table(file_path)
        columns = ["Pt_id", "ICD", "Time", "tte"] if cls_type == 'surv' else ["Pt_id", "ICD", "Time"]

        # Ensure all required columns exist
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in {file_path}")

        data = data[columns]
        data['Label'] = label
        print(f"{'Case' if label == 1 else 'Control'} counts: {data['Pt_id'].nunique()}")
        return data

    def process_patient_visits(group: pd.DataFrame, types: Dict[str, int]) -> Tuple[List[List[str]], List[List[int]], List[int]]:
        """Processes visits for a single patient."""
        
        data_i_c = []
        data_dt_c = []
        
        # Process visits - Sort by time in descending order (most recent first)
        for time, subgroup in group.sort_values(['Time'], ascending=False).groupby('Time', sort=False):
            data_i_c.append(np.array(subgroup['ICD']).tolist())
            data_dt_c.append(dt.strptime(time, '%Y-%m-%d'))

        # Calculate duration between visits
        v_dur_c = [0] if len(data_dt_c) <= 1 else [
            0 if i == 0 else (data_dt_c[i-1] - data_dt_c[i]).days
            for i in range(len(data_dt_c))
        ]

        # Recode diagnoses to integers
        newPatient_c = []
        max_type_id = max(types.values()) if types else 0
        for visit in data_i_c:
            newVisit_c = []
            for code in visit:
                if code not in types:
                    max_type_id += 1
                    types[code] = max_type_id
                newVisit_c.append(types[code])
            newPatient_c.append(newVisit_c)

        return data_i_c, newPatient_c, v_dur_c


    try:
        # Load and combine data
        print("Loading cases and controls")
        data_case = load_and_process_data(case_file, 1)
        data_control = load_and_process_data(control_file, 0)
        data_combined = pd.concat([data_case, data_control])
        print(f"Total counts: {data_combined['Pt_id'].nunique()}")

        # Load or initialize types dictionary
        if type_file == 'NA':
            types = {"Zeropad": 0}
        else:
            with open(type_file, 'rb') as f:
                types = pkl.load(f)

        # Process all patients to convert their visits into a structured format
        print("Processing patients")
        pt_list, label_list, newVisit_list, dur_list = [], [], [], []
        
        for (pt_id, group) in data_combined.groupby('Pt_id'):
            data_i_c, newPatient_c, v_dur_c = process_patient_visits(group, types)
            
            if data_i_c:  # Only save non-empty entries
                label = [group.iloc[0]['Label'], group.iloc[0]['tte']] if cls_type == 'surv' else group.iloc[0]['Label']
                pt_list.append(pt_id)
                label_list.append(label)
                newVisit_list.append(newPatient_c)
                dur_list.append(v_dur_c)

        # Save types mapping
        with open(f"{out_file}.types", 'wb') as f:
            pkl.dump(types, f, -1)

        # Create train/test/validation splits
        print("Creating data splits")
        if pts_file_pre == 'NA':
            X_train_val, X_test, y_train_val, y_test, train_val_idx, test_idx = train_test_split(
                pt_list, label_list, range(len(pt_list)), 
                test_size=0.2, random_state=7, stratify=label_list
            )
            train_idx, valid_idx = train_test_split(
                train_val_idx, test_size=0.125, random_state=7, 
                stratify=y_train_val
            )
        else:
            # Load existing splits
            splits = {
                'train': pkl.load(open(f"{pts_file_pre}.train", 'rb')),
                'valid': pkl.load(open(f"{pts_file_pre}.valid", 'rb')),
                'test': pkl.load(open(f"{pts_file_pre}.test", 'rb'))
            }
            train_idx, valid_idx, test_idx = [
                np.intersect1d(pt_list, splits[key], assume_unique=True, return_indices=True)[1]
                for key in ['train', 'valid', 'test']
            ]

        # Save splits
        for subset, indices in [('train', train_idx), ('valid', valid_idx), ('test', test_idx)]:
            subset_pts = [pt_list[i] for i in indices]
            pkl.dump(subset_pts, open(f"{out_file}.pts.{subset}", 'wb'), protocol=2)

        # Create and save combined datasets
        print("Creating combined datasets")
        fset = [
            [pt_list[i], label_list[i], [
                [[dur_list[i][v]], newVisit_list[i][v]]
                for v in range(len(newVisit_list[i]))
            ]]
            for i in range(len(pt_list))
        ]

        for subset, indices in [('train', train_idx), ('valid', valid_idx), ('test', test_idx)]:
            subset_data = [fset[i] for i in indices]
            pkl.dump(subset_data, open(f"{out_file}.combined.{subset}", 'wb'), -1)

        print("Processing complete")

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise

if __name__ == "__main__":
    # 1. Setup paths
    outcome_input_common_path = get_common_path('../../Data/')
    input_files = {
        'encounters': outcome_input_common_path + 'patients_encounters_Dataset_final_new.csv',
        'diagnosis': outcome_input_common_path + 'phewas_diagnosis_Dataset_final_new.csv',
        'medications': outcome_input_common_path + 'mod_drug_Dataset_final_new.csv',
        'procedures': outcome_input_common_path + 'mod_proc_Dataset_final_new.csv'
    }

    # 2. Setup output paths
    output_folder_name = 'Results_15_final_new'
    output_base = get_common_path('../../Results_new_no_censored/') + output_folder_name + '/'
    
    # 3. Define processing parameters
    days_before_index = 1825
    prediction_windows = {
        '1_365': [1, 365],
        '1_730': [1, 730],
        '1_1095': [1, 1095],
        '1_1825': [1, 1825]
    }

    # 4. Process each prediction window
    count_file = 0
    for window_name, window_params in prediction_windows.items():
        start_gap, end_gap = window_params
        start_time = dt.now()
        count_file += 1
        
        # Log processing start
        print(f"{count_file}. Starting processing {window_name}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Window: {start_gap}d to {end_gap}d\n")
        
        # Create output directory for this window
        window_dir = f"{output_base}/{days_before_index}/Results_{start_gap}d_{end_gap}d_window/"
        if not os.path.exists(window_dir):
            os.makedirs(window_dir)
            
        # 5. Get qualified patients and their encounters
        qualified_patients, patient_encounters_daygaps_list_dic = get_qualified_patients(
            input_files['encounters'], 
            start_gap
        )
        
        qualified_encounters = get_qualified_input_encid_dic(
            input_files['encounters'],
            qualified_patients,
            days_before_index
        )
        
        # 6. Identify case and control patients
        case_patients, control_patients = get_case_control_set(
            input_files['encounters'],
            qualified_patients,
            start_gap,
            end_gap,
            patient_encounters_daygaps_list_dic
        )
        
        # 7. Process features for cases and controls
        patient_groups = [
            (case_patients, 'case.tsv'),
            (control_patients, 'control.tsv')
        ]
        
        for idx1, (patients, output_filename) in enumerate(patient_groups):
            print(f"{count_file}-{idx1+1}. Start generating {output_filename} for Results_{start_gap}d_{end_gap}d_window")
            
            # Extract and combine features
            features = {
                'age': get_specific_file_input_dic('age', input_files['encounters'], patients, qualified_encounters),
                'gender': get_specific_file_input_dic('gender', input_files['encounters'], patients, qualified_encounters),
                'diagnosis': get_specific_file_input_dic('diagnosis', input_files['diagnosis'], patients, qualified_encounters),
                'medications': get_specific_file_input_dic('medication', input_files['medications'], patients, qualified_encounters),
                'procedures': get_specific_file_input_dic('procedure', input_files['procedures'], patients, qualified_encounters)
            }
            
            all_features = list(features.values())
            
            # Save and deduplicate
            output_path = os.path.join(window_dir, output_filename)
            dedup_output_path = os.path.join(window_dir, f"dedup_{output_filename}")
            
            print(f"{count_file}-{idx1+1}-1. Start writing {output_filename}")
            write_data_lists_to_tsv(all_features, output_path)
            
            print(f"{count_file}-{idx1+1}-2. Start deduplicating {output_filename}")
            deduplicate_tsv(output_path, dedup_output_path)
            
            print(f"{count_file}-{idx1+1}-3. Finished deduplicating {output_filename}\n")
        
        end_time_case_ctl = dt.now()
        print(f"Time for case/control processing: {end_time_case_ctl - start_time}\n")

        
        # 8. Create train/valid/test splits
        print(f"Start train/test/valid split at {dt.now()}")
        print(f"Generating train/test/valid for Results_{start_gap}d_{end_gap}d_window")
        
        case_file, control_file, type_file, out_file, cls_type, pts_file_pre = get_split_file_paths(window_dir)
        
        process_and_split_data(case_file, control_file, type_file, out_file, cls_type, pts_file_pre)
        print(f"{count_file}-{idx1+1}. Finished generating train/test/valid splits\n")

    
        # 9. Log completion times
        end_time_train_test = dt.now()
        print(f"Time for train/test/valid split: {end_time_train_test - end_time_case_ctl}")
        print(f"Total processing time: {end_time_train_test - start_time}\n")