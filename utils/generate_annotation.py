import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.logger import setup_logger, create_progress_bar, get_device_info

# Setup logger
logger = setup_logger('annotation_generator', 'logs/annotation_generator.log')
logger.info(f"Using device: {get_device_info()}")

ground_truth_dir = 'data/Audios/Ground truth/'
annotations = []

default_duration = 0.05  # segundos

# Lista de arquivos para processar
files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('_GT.xlsx')])
logger.info(f"Found {len(files)} ground truth files to process")

for filename in create_progress_bar(files, desc="Processing ground truth files"):
    audio_file = filename.replace('_GT.xlsx', '.WAV')
    file_path = os.path.join(ground_truth_dir, filename)

    try:
        df = pd.read_excel(file_path)
        df = df.dropna()
        if 'Start_time' not in df.columns or 'GT' not in df.columns:
            logger.warning(f'Expected columns missing in {filename}. Skipping.')
            continue
        
        start_times = df['Start_time'].tolist()
        labels = df['GT'].tolist()

        for i in range(len(start_times)):
            start_time = start_times[i]
            if i < len(start_times) - 1:
                end_time = start_times[i + 1]
            else:
                end_time = start_time + default_duration

            label = labels[i]
            
            annotations.append({
                'audio_file': audio_file,
                'start_time': round(start_time, 6),
                'end_time': round(end_time, 6),
                'label_str': label.strip().strip("'")
            })
    except Exception as e:
        logger.error(f"Error processing {filename}: {str(e)}")

logger.info(f"Processing {len(annotations)} annotations")
annotations_df = pd.DataFrame(annotations)
label_encoder = LabelEncoder()
annotations_df['label'] = label_encoder.fit_transform(annotations_df['label_str'])

# Save the annotations
output_path = 'data/annotations.csv'
annotations_df.to_csv(output_path, index=False)
logger.info(f'✅ Annotations saved to {output_path}')
logger.info(f'Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}')
