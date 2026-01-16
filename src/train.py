import pandas as pd
import logging
import warnings
from src.data_loader import DataLoader
from src.model import ForecastEngine
from src.evaluate import Evaluator

# Config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

TRAIN_END_DATE = '2017-07-15'
FORECAST_HORIZON = 30
CSV_PATH = 'train.csv'

if __name__ == "__main__":
    try:
        # 1. Load Data
        loader = DataLoader(CSV_PATH)
        df = loader.load_and_clean()

        # 2. Split Data (Train / Validation)
        train_data = df[df['date'] <= TRAIN_END_DATE]
        
        # Validation set must align with forecast horizon
        valid_data = df[df['date'] > TRAIN_END_DATE]
        valid_data = valid_data[valid_data['date'] <= pd.to_datetime(TRAIN_END_DATE) + pd.Timedelta(days=FORECAST_HORIZON)]

        # 3. Train Models (Parallel)
        engine = ForecastEngine(train_data, FORECAST_HORIZON)
        results = engine.train_all_categories()

        # 4. Evaluate
        wmape, metrics_df = Evaluator.calculate_metrics(results, valid_data)
        
        print(f"\nFINAL RESULTS:")
        print(f"Global WMAPE: {wmape:.4%}")
        print("\nTop 5 Best Performing Categories:")
        print(metrics_df.head(5))

    except Exception as e:
        logging.error(f"Pipeline Failed: {e}")
