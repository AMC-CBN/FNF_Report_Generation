import os
import pickle
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 원본 pickle 데이터가 있는 폴더
#dir_path = "C:/Users/고효진/Workspace/FNF/preprocessed_data/datasets/pickle_data_1733"
dir_path = "/workspace/FNF/preprocessed_data/datasets/pickle_data_exception5"# 변환된 엑셀 파일을 저장할 폴더
output_dir = "/workspace/FNF/preprocessed_data/datasets/excel_output"

os.makedirs(output_dir, exist_ok=True)

def get_file_type(garden_type):
    """Garden Type을 파일의 Type 번호로 변환"""
    mapping = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
    return mapping[garden_type]

def analyze_detection_data(df, view_type):
    """Detection 데이터 분석 (Hip Joint Detection용)"""
    stats = {
        'sample_count': len(df),
        'image_count': len(df[f'Detection_image_{view_type}']),
        'has_ratio': sum(1 for x in df[f'Ratio_{view_type}_list'] if isinstance(x, list) and len(x) > 0),
        'has_pad': sum(1 for x in df[f'Pad_{view_type}_list'] if isinstance(x, list) and len(x) > 0),
        'has_xml': df[f'Xml_path_{view_type}'].notna().sum()
    }
    return stats

def analyze_classification_data(df):
    """Classification 데이터 분석 (Garden Classification용)"""
    stats = {
        'sample_count': len(df),
        'ap_images': len(df) * 2 if 'Crop_AP_Left_image' in df.columns else len(df),
        'lat_images': len(df) if 'Crop_LAT_image' in df.columns or 'Dicom_image_path_LAT' in df.columns else 0
    }
    
    if 'Garden_Type' in df.columns:
        # Garden Type 매핑 (파일의 Type -> 논문의 Garden Type)
        garden_dist = {
            'I': len(df[df['Garden_Type'] == 0]),    # Type 0 -> Garden I
            'II': len(df[df['Garden_Type'] == 1]),   # Type 1 -> Garden II
            'III': len(df[df['Garden_Type'] == 2]),  # Type 2 -> Garden III
            'IV': len(df[df['Garden_Type'] == 3])    # Type 3 -> Garden IV
        }
        stats['garden_type_dist'] = garden_dist
    
    return stats

def analyze_image_properties(df, view_type):
    """이미지 속성 상세 분석"""
    stats = {
        'width': [], 'height': [], 'mean': [], 'std': [],
        'aspect_ratio': [], 'size': []
    }
    
    images = df[f'Detection_image_{view_type}']
    for img in images:
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            stats['width'].append(w)
            stats['height'].append(h)
            stats['aspect_ratio'].append(w/h)
            stats['size'].append(w*h)
            stats['mean'].append(np.mean(img))
            stats['std'].append(np.std(img))
    
    return stats

def plot_garden_distribution(total_garden, output_dir):
    """Garden Type 분포 시각화"""
    plt.figure(figsize=(12, 6))
    
    # 현재 데이터
    x = np.arange(4)
    current_values = [total_garden[t] for t in ['I', 'II', 'III', 'IV']]
    total = sum(current_values)
    current_percentages = [v/total*100 for v in current_values]
    
    # 논문 데이터
    paper_values = [378, 68, 477, 665]
    paper_total = sum(paper_values)
    paper_percentages = [v/paper_total*100 for v in paper_values]
    
    # 막대 그래프
    width = 0.35
    plt.bar(x - width/2, current_percentages, width, label='Current Data')
    plt.bar(x + width/2, paper_percentages, width, label='Paper Data')
    
    plt.title('Garden Type Distribution: Current vs Paper')
    plt.xlabel('Garden Type')
    plt.ylabel('Percentage (%)')
    plt.xticks(x, ['Type I', 'Type II', 'Type III', 'Type IV'])
    plt.legend()
    
    # 값 표시
    for i, (c, p) in enumerate(zip(current_percentages, paper_percentages)):
        plt.text(i - width/2, c, f'{current_values[i]}\n({c:.1f}%)', ha='center', va='bottom')
        plt.text(i + width/2, p, f'{paper_values[i]}\n({p:.1f}%)', ha='center', va='bottom')
    
    plt.savefig(os.path.join(output_dir, 'garden_type_comparison.png'))
    plt.close()

def plot_image_analysis(stats, view_type, output_dir):
    """이미지 분석 결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{view_type} View Image Analysis')
    
    # 이미지 크기 분포
    axes[0,0].scatter(stats['width'], stats['height'])
    axes[0,0].set_title('Image Dimensions')
    axes[0,0].set_xlabel('Width')
    axes[0,0].set_ylabel('Height')
    
    # 종횡비 분포
    sns.histplot(data=stats['aspect_ratio'], ax=axes[0,1])
    axes[0,1].set_title('Aspect Ratio Distribution')
    
    # 이미지 크기 분포
    sns.histplot(data=stats['size'], ax=axes[1,0])
    axes[1,0].set_title('Image Size Distribution')
    
    # 픽셀값 통계
    sns.boxplot(data=[stats['mean'], stats['std']], ax=axes[1,1])
    axes[1,1].set_xticklabels(['Mean', 'Std'])
    axes[1,1].set_title('Pixel Value Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{view_type}_image_analysis.png'))
    plt.close()

print("\n=== 데이터 분석 ===")

print("\n1. Hip Joint Detection 데이터 분석")
print("=" * 50)

# Detection 데이터 분석
for file in os.listdir(dir_path):
    if 'Detection' in file and file.endswith('.pkl'):
        view_type = 'AP' if 'AP' in file else 'LAT'
        try:
            with open(os.path.join(dir_path, file), 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n📊 {file} 분석:")
            print("-" * 40)
            total_stats = {'sample_count': 0, 'image_count': 0, 'has_ratio': 0, 'has_pad': 0, 'has_xml': 0}
            
            for fold_name, fold_data in data.items():
                df = pd.DataFrame(fold_data)
                stats = analyze_detection_data(df, view_type)
                print(f"• {fold_name}:")
                print(f"  - Sample 수: {stats['sample_count']}")
                print(f"  - 검출된 이미지: {stats['image_count']}")
                print(f"  - Ratio 정보 있음: {stats['has_ratio']}")
                print(f"  - Padding 정보 있음: {stats['has_pad']}")
                print(f"  - XML 파일 있음: {stats['has_xml']}")
                
                for key in total_stats:
                    total_stats[key] += stats[key]
            
            print(f"\n📈 {view_type} View 전체 통계:")
            print(f"  - 전체 Sample 수: {total_stats['sample_count']}")
            print(f"  - 전체 검출된 이미지: {total_stats['image_count']}")
            print(f"  - Ratio 정보 있는 이미지: {total_stats['has_ratio']}")
            print(f"  - Padding 정보 있는 이미지: {total_stats['has_pad']}")
            print(f"  - XML 파일 있는 이미지: {total_stats['has_xml']}")
            
            stats = analyze_image_properties(df, view_type)
            plot_image_analysis(stats, view_type, output_dir)
            
        except Exception as e:
            print(f"❌ {file} 분석 실패: {e}")

print("\n2. Garden Classification 데이터 분석")
print("=" * 50)

# Classification 데이터 분석
for file in os.listdir(dir_path):
    if 'Classification' in file or 'Paper' in file:
        try:
            with open(os.path.join(dir_path, file), 'rb') as f:
                data = pickle.load(f)
            
            print(f"\n📊 {file} 분석:")
            print("-" * 40)
            total_stats = {'sample_count': 0, 'ap_images': 0, 'lat_images': 0}
            total_garden = {'I': 0, 'II': 0, 'III': 0, 'IV': 0}
            
            for fold_name, fold_data in data.items():
                df = pd.DataFrame(fold_data)
                stats = analyze_classification_data(df)
                print(f"• {fold_name}:")
                print(f"  - Sample 수: {stats['sample_count']}")
                if 'garden_type_dist' in stats:
                    print("  - Garden Type 분포:")
                    for gtype, count in stats['garden_type_dist'].items():
                        print(f"    Garden Type {gtype} (파일 Type {get_file_type(gtype)}): {count}")
                        total_garden[gtype] += count
                print(f"  - AP view 이미지: {stats['ap_images']}")
                print(f"  - Lateral view 이미지: {stats['lat_images']}")
                
                total_stats['sample_count'] += stats['sample_count']
                total_stats['ap_images'] += stats['ap_images']
                total_stats['lat_images'] += stats['lat_images']
            
            print(f"\n📈 전체 통계:")
            print(f"  - 전체 Sample 수: {total_stats['sample_count']}")
            if total_garden:
                total = sum(total_garden.values())
                print("\n  [Garden Type 분포]")
                print("  현재 데이터:")
                for gtype, count in total_garden.items():
                    print(f"    Garden Type {gtype} (파일 Type {get_file_type(gtype)}): {count} ({count/total*100:.1f}%)")
                
                print("\n  [논문 데이터와 비교]")
                print("  논문:")
                print("    Garden Type I:   378 (23.8%)")
                print("    Garden Type II:   68 (4.3%)")
                print("    Garden Type III: 477 (30.0%)")
                print("    Garden Type IV:  665 (41.9%)")
                print("  현재:")
                print(f"    Garden Type I:   {total_garden['I']} ({total_garden['I']/total*100:.1f}%)")
                print(f"    Garden Type II:  {total_garden['II']} ({total_garden['II']/total*100:.1f}%)")
                print(f"    Garden Type III: {total_garden['III']} ({total_garden['III']/total*100:.1f}%)")
                print(f"    Garden Type IV:  {total_garden['IV']} ({total_garden['IV']/total*100:.1f}%)")
            
            print(f"\n  - 전체 AP view 이미지: {total_stats['ap_images']}")
            print(f"  - 전체 Lateral view 이미지: {total_stats['lat_images']}")
            
            if total_garden:
                plot_garden_distribution(total_garden, output_dir)
            
        except Exception as e:
            print(f"❌ {file} 분석 실패: {e}")

print("\n분석이 완료되었습니다.")