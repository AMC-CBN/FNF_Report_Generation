import pickle
import pandas as pd
import numpy as np

# 파일 존재 확인
import os
pickle_path = '/workspace/FNF/preprocessed_data/datasets/pickle_data_exception5/FNF_Classification_data.pkl'
print(f"Pickle file exists: {os.path.exists(pickle_path)}")

# 파일 크기 확인
print(f"Pickle file size: {os.path.getsize(pickle_path) / (1024*1024):.2f} MB")

print("Loading pickle file...")
try:
    with open(pickle_path, 'rb') as f:
        print("File opened successfully")
        pickle_data = pickle.load(f)
        print("Pickle data loaded successfully")
        print(f"Pickle data type: {type(pickle_data)}")
        if isinstance(pickle_data, dict):
            print(f"Number of keys: {len(pickle_data)}")
            print(f"Keys: {list(pickle_data.keys())}")
except Exception as e:
    print(f"Error loading pickle file: {str(e)}")

# Pickle 데이터의 serial number와 garden type 추출
print("Extracting pickle data...")
pkl_info = {}
pkl_serials = set()
for fold_name, fold_data in pickle_data.items():
    for serial, garden_type in zip(fold_data['Serial'], fold_data['Garden_Type']):
        pkl_info[serial] = garden_type + 1
        pkl_serials.add(serial)
print(f"Number of serials in pickle: {len(pkl_serials)}")

# Excel 파일 확인
excel_path = '/workspace/FNF/labeling_detection_2to0_1to0.xlsx'
print(f"Excel file exists: {os.path.exists(excel_path)}")

# Excel 데이터 읽기
print("Loading excel file...")
excel_data = pd.read_excel(excel_path)
print(f"Total rows in excel: {len(excel_data)}")

# NaN이 아닌 serial number만 선택
excel_data = excel_data.dropna(subset=['serial No.'])
print(f"Valid rows in excel (with serial numbers): {len(excel_data)}")

print("불일치 목록:")
print("Serial No. | PKL Type | PKL Excl | Label Type | Label Excl | 불일치 유형")
print("-" * 85)

# 모든 시리얼 번호에 대해 비교
print("Starting comparison...")
all_serials = sorted(set(list(pkl_serials) + list(excel_data['serial No.'])))
print(f"Total unique serials to compare: {len(all_serials)}")

for i, serial in enumerate(all_serials):
    if i % 100 == 0:  # 진행상황 표시
        print(f"Processing {i}/{len(all_serials)} serials...")
        
    # Pickle 데이터 확인
    pkl_type = pkl_info.get(serial, "없음")
    pkl_excl = 0 if serial in pkl_serials else "없음"  # pkl에 있으면 exclusion=0으로 간주
    
    # Excel 데이터 확인
    excel_row = excel_data[excel_data['serial No.'] == serial]
    if not excel_row.empty:
        excel_label = excel_row['정답label(by CT)'].iloc[0]
        excel_excl = excel_row['exclusion'].iloc[0]
        
        # 불일치 유형 확인
        type_mismatch = False
        excl_mismatch = False
        
        if pd.notna(excel_label):
            excel_label_str = str(excel_label)
            if pkl_type != "없음":
                if ('?' in excel_label_str) or (str(pkl_type) != excel_label_str):
                    type_mismatch = True
            
            if (pkl_excl == 0 and excel_excl == 2) or (pkl_excl == "없음" and excel_excl == 0):
                excl_mismatch = True
            
            if type_mismatch or excl_mismatch:
                mismatch_type = []
                if type_mismatch: mismatch_type.append("Type")
                if excl_mismatch: mismatch_type.append("Excl")
                
                print(f"{serial} | {pkl_type} | {pkl_excl} | {excel_label} | {excel_excl} | {'+'.join(mismatch_type)}")

print("\n참고:")
print("- PKL Type: Pickle 파일의 Garden Type")
print("- PKL Excl: Pickle 파일 포함 여부 (포함=0, 미포함='없음')")
print("- Label Type: labeling.xlsx의 정답label(by CT)")
print("- Label Excl: labeling.xlsx의 exclusion")
print("- 불일치 유형: Type (Garden Type 불일치), Excl (Exclusion 불일치)")

# 파일 경로
current_pkl = '/workspace/FNF/preprocessed_data/datasets/pickle_data_2to0_1to0/FNF_Classification_data.pkl'
answer_pkl = '/workspace/FNF/nas/FNF/Final_data/datasets/pickle_data_1733/FNF_Classification_data.pkl'

print("\nLoading files...")
try:
    # pkl 파일들 로드
    with open(current_pkl, 'rb') as f:
        current_data = pickle.load(f)
    print("Current pickle loaded")
    
    with open(answer_pkl, 'rb') as f:
        answer_data = pickle.load(f)
    print("Answer pickle loaded")
    
    # Excel 파일 로드
    excel_data = pd.read_excel(excel_path)
    print("Excel file loaded")
    
    # 각 pkl 파일의 serial number 세트 만들기
    current_serials = set()
    answer_serials = set()
    
    # Garden Type 매핑 딕셔너리 생성
    answer_types = {}
    
    for fold in current_data.keys():
        current_serials.update(current_data[fold]['Serial'])
        for serial, garden_type in zip(answer_data[fold]['Serial'], answer_data[fold]['Garden_Type']):
            answer_serials.add(serial)
            answer_types[serial] = garden_type
    
    # Serial number 차이 분석
    only_in_answer = answer_serials - current_serials
    
    print("\nAnalyzing serials only in answer pkl:")
    print("Serial | Garden Type | Excel Label | Excel Exclusion")
    print("-" * 55)
    
    for serial in sorted(only_in_answer):
        # Excel 데이터에서 해당 serial 정보 찾기
        excel_row = excel_data[excel_data['serial No.'] == serial]
        if not excel_row.empty:
            excel_label = excel_row['정답label(by CT)'].iloc[0]
            excel_excl = excel_row['exclusion'].iloc[0]
        else:
            excel_label = "Not found"
            excel_excl = "Not found"
            
        print(f"{serial:6d} | {answer_types[serial]:11d} | {excel_label!s:11} | {excel_excl!s:15}")
    
    # Garden Type 분포 비교
    print("\nGarden Type Distribution:")
    print("Source | Type 0 | Type 1 | Type 2 | Type 3")
    print("-" * 45)
    
    def count_types(data):
        all_types = []
        for fold in data.keys():
            all_types.extend(data[fold]['Garden_Type'])
        return np.bincount(all_types, minlength=4)
    
    current_counts = count_types(current_data)
    answer_counts = count_types(answer_data)
    
    print(f"Current | {current_counts[0]:7d} | {current_counts[1]:7d} | {current_counts[2]:7d} | {current_counts[3]:7d}")
    print(f"Answer  | {answer_counts[0]:7d} | {answer_counts[1]:7d} | {answer_counts[2]:7d} | {answer_counts[3]:7d}")
    print(f"Diff    | {current_counts[0]-answer_counts[0]:+7d} | {current_counts[1]-answer_counts[1]:+7d} | {current_counts[2]-answer_counts[2]:+7d} | {current_counts[3]-answer_counts[3]:+7d}")

    # 각 fold별로 어디에 속해있는지 확인
    print("\nFold distribution of different serials:")
    print("Serial | Fold | Garden Type")
    print("-" * 35)
    
    for serial in sorted(only_in_answer):
        for fold in answer_data.keys():
            if serial in answer_data[fold]['Serial']:
                idx = answer_data[fold]['Serial'].index(serial)
                garden_type = answer_data[fold]['Garden_Type'][idx]
                print(f"{serial:6d} | {fold:4} | {garden_type:11d}")

    print("Answer pkl의 누락된 시리얼 정보:")
    print("Serial | Garden Type | Excel Label | Excel Exclusion | PKL Exclusion | Fold")
    print("-" * 80)

    for serial in [617, 742, 2194, 2277, 2353]:
        # Excel 데이터에서 해당 serial 정보 찾기
        excel_row = excel_data[excel_data['serial No.'] == serial]
        excel_label = excel_row['정답label(by CT)'].iloc[0] if not excel_row.empty else "Not found"
        excel_excl = excel_row['exclusion'].iloc[0] if not excel_row.empty else "Not found"
        
        # Answer pkl에서 fold와 garden type, exclusion 찾기
        for fold in answer_data.keys():
            if serial in answer_data[fold]['Serial']:
                idx = answer_data[fold]['Serial'].index(serial)
                garden_type = answer_data[fold]['Garden_Type'][idx]
                pkl_excl = answer_data[fold].get('Exclusion', [0]*len(answer_data[fold]['Serial']))[idx]  # exclusion이 없으면 0으로 가정
                print(f"{serial:6d} | {garden_type:11d} | {excel_label!s:11} | {excel_excl!s:15} | {pkl_excl!s:13} | {fold:4}")

except Exception as e:
    print(f"Error: {str(e)}")