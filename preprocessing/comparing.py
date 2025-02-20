import pickle
import pandas as pd
import numpy as np

# 1.0에서 제외되어야 할 데이터
exclude_1_0 = [51, 259, 598, 554, 1334]

# Internal Raw Dicom 제외 대상
exclude_internal = [98, 141, 161, 228, 501, 748, 1592, 1769]

# LAT Detection box 추가 데이터
add_lat = [808, 1175, 1363, 1587]


# 파일 로드
pkl_path = '/workspace/FNF/nas/FNF/Final_data/datasets/pickle_data_1733/FNF_Classification_data.pkl'
excel_path = '/workspace/FNF/labeling_detection_2to0_1to0_exception.xlsx'

excel_data = pd.read_excel(excel_path, usecols=['serial No.', '정답label(by CT)', 'exclusion'])
excel_data = excel_data.dropna(how='all')
excel_data = excel_data[pd.notna(excel_data['serial No.'])]

with open(pkl_path, 'rb') as f:
    pkl_data = pickle.load(f)
# PKL의 모든 serial number 수집
pkl_serials = set()
for fold_data in pkl_data.values():
    pkl_serials.update(fold_data['Serial'])

print("=== 데이터 체크 ===")

print("\n1. 1.0에서 제외되어야 할 데이터 체크:")
for serial in exclude_1_0:
    in_pkl = serial in pkl_serials
    excel_row = excel_data[excel_data['serial No.'] == serial]
    excel_info = f"Type: {excel_row['정답label(by CT)'].iloc[0]}, Exclusion: {excel_row['exclusion'].iloc[0]}" if not excel_row.empty else "Not found"
    print(f"Serial {serial:4d}: {'있음 ❌' if in_pkl else '없음 ✓'} (Excel: {excel_info})")

print("\n2. Internal Raw Dicom 제외 대상 체크:")
for serial in exclude_internal:
    in_pkl = serial in pkl_serials
    excel_row = excel_data[excel_data['serial No.'] == serial]
    excel_info = f"Type: {excel_row['정답label(by CT)'].iloc[0]}, Exclusion: {excel_row['exclusion'].iloc[0]}" if not excel_row.empty else "Not found"
    print(f"Serial {serial:4d}: {'있음 ❌' if in_pkl else '없음 ✓'} (Excel: {excel_info})")

print("\n3. LAT Detection box 추가 데이터 체크:")
for serial in add_lat:
    in_pkl = serial in pkl_serials
    excel_row = excel_data[excel_data['serial No.'] == serial]
    excel_info = f"Type: {excel_row['정답label(by CT)'].iloc[0]}, Exclusion: {excel_row['exclusion'].iloc[0]}" if not excel_row.empty else "Not found"
    print(f"Serial {serial:4d}: {'있음 ✓' if in_pkl else '없음 ❌'} (Excel: {excel_info})")

# Garden Type 분포 확인
print("\n=== Garden Type 분포 ===")
print("\nPKL Garden Type 분포:")
pkl_types = []
for fold_data in pkl_data.values():
    pkl_types.extend([t + 1 for t in fold_data['Garden_Type']])
pkl_type_counts = pd.Series(pkl_types).value_counts().sort_index()
print(pkl_type_counts)

print("\nExcel Garden Type 분포 (exclusion=0):")
excel_type_counts = excel_data[excel_data['exclusion'] == 0]['정답label(by CT)'].value_counts().sort_index()
print(excel_type_counts)