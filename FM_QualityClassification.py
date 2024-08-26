import pandas as pd
import os

def create_classification_file(excel_file_path, binary=True):
    """Create classification file from Excel data."""
    if binary:
        output_file_name = 'FM_QualityClassification_Binary.txt'
    else:
        output_file_name = 'FM_QualityClassification_ThreeClass.txt'

    output_file_path = os.path.join(r'C:\GitRepositories\Master_Thesis', output_file_name)

    df = pd.read_excel(excel_file_path, usecols=[0, 10])
    df.iloc[:, 1] = df.iloc[:, 1].astype(str).str.strip().replace({r'\*': ''}, regex=True)

    classification_dict = {}
    for i, row in df.iterrows():
        base_id = ''.join(filter(str.isdigit, str(row.iloc[0]).strip()))
        fm_quality = str(row.iloc[1]).strip()
        if fm_quality.lower() == 'nan' or fm_quality == '':
            continue
        if binary:
            fm_quality = "FM+" if "FM+" in fm_quality or "FM++" in fm_quality else "FM-" if "FM-" in fm_quality else None
        if fm_quality in ["FM-", "FM+", "FM++"]:
            classification_dict[base_id] = fm_quality
    
    with open(output_file_path, 'w') as file:
        for child_id, fm_quality in classification_dict.items():
            file.write(f"ID: {child_id}, FM Quality: {fm_quality}\n")
    print(f"Classification file created: {output_file_path}")
