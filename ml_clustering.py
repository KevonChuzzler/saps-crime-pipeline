import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DB_URL = "sqlite:///saps_crime.db"

def run_ml_clustering():
    print("Connecting to database to fetch crime data...")
    engine = create_engine(DB_URL)
    
    # 1. Load data from the database
    df = pd.read_sql("SELECT * FROM crime_stats", engine)
    
    # 2. Isolate the most recent year's data for current risk profiling
    latest_year = df['year_start'].max()
    print(f"Profiling stations based on {latest_year} data...")
    df_latest = df[df['year_start'] == latest_year].copy()
    
    # Define columns to use for clustering (exclude geographic and time metadata)
    metadata_cols = ['year', 'year_start', 'station', 'loc_mn', 'dc_mn', 'longitude', 'latitude']
    crime_cols = [col for col in df_latest.columns if col not in metadata_cols]
    
    # 3. Standardize the data
    # K-Means is distance-based, so scaling ensures all crime types weigh equally
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_latest[crime_cols])
    
    # 4. Train the K-Means Model
    # We specify 4 clusters to represent different risk tiers (e.g., Low, Medium, High, Extreme)
    print("Training K-Means clustering algorithm...")
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init="auto")
    
    # Assign the predicted cluster label (0, 1, 2, or 3) to each police station
    df_latest['risk_cluster'] = kmeans_model.fit_predict(scaled_features)
    
    # 5. Save the output back to the database as a new table
    print("Saving risk profiles to the database...")
    df_latest.to_sql('station_risk_profiles', con=engine, if_exists='replace', index=False)
    
    print(f"\nSuccess! Clustered {len(df_latest)} stations into 4 risk tiers.")
    print("Sample of clustered stations:")
    print(df_latest[['station', 'murder', 'carjacking', 'risk_cluster']].head(10))

if __name__ == "__main__":
    run_ml_clustering()