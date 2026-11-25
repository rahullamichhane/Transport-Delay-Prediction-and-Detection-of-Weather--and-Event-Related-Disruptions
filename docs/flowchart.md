# System Flowchart

```mermaid
flowchart TD
    %% Start
    Start([Start]) --> Config[Load Configuration]

    %% Phase 1: Vehicle Classes
    Config --> VC[Vehicle Class Hierarchy]
    VC --> Bus[Bus Class<br/>base_delay = 5 min]
    VC --> Streetcar[Streetcar Class<br/>base_delay = 7 min]
    VC --> Subway[Subway Class<br/>base_delay = 3 min]

    %% Phase 2: Data Generation
    Bus --> Gen[TransportDataGenerator]
    Streetcar --> Gen
    Subway --> Gen
    Gen --> Hist{Mode?}
    
    Hist -->|Historical| HistGen[generate_historical n=6000<br/>Toronto TTC records]
    Hist -->|Live Stream| StreamGen[stream_records n<br/>one record at a time]
    
    HistGen --> Prep[Preprocessing Pipeline]
    StreamGen --> StreamPred[Live Streaming Detection]

    %% Phase 3: Preprocessing
    Prep --> Drop[Drop IDs: record_id, timestamp, vehicle_id, route_id, stop_id, delay_minutes]
    Drop --> Split[Stratified Train/Test Split 80/20]
    Split --> Num[Numeric: hour, day_of_week, is_peak_hour, is_weekend, temperature, humidity<br/>Median impute + StandardScaler]
    Split --> Cat[Categorical: vehicle_class, route_type, weather_condition, event_type<br/>Most-frequent impute + OneHotEncoder]
    Num --> Models[Model Training]
    Cat --> Models

    %% Phase 4: Model Training (6 models)
    Models --> LR[Logistic Regression]
    Models --> RF[Random Forest]
    Models --> GB[Gradient Boosting]
    Models --> KNN[K-Nearest Neighbors]
    Models --> NB[Naive Bayes]
    Models --> XGB[XGBoost]

    LR --> Eval[Evaluate on Test Set<br/>Accuracy, Precision, Recall, F1, ROC-AUC]
    RF --> Eval
    GB --> Eval
    KNN --> Eval
    NB --> Eval
    XGB --> Eval

    Eval --> SaveModels[Save pipelines to output/models/<br/>+ preprocessor.pkl]
    SaveModels --> SaveTable[Save model_comparison.csv to output/tables/]
    SaveTable --> SaveImages[Save confusion matrices, ROC curves,<br/>comparison charts to output/images/]

    %% Phase 5: Live Streaming Detection
    StreamPred --> LoadModels[Load trained pipelines from output/models/]
    LoadModels --> Predictor[StreamingPredictor rolling window = 200]
    Predictor --> Register[Register all 6 models]
    
    Register --> Loop{For each live record}
    Loop --> Row[Build 1-row DataFrame from record]
    Row --> Pred1[pipeline.predict(record)<br/>for every model]
    Pred1 --> Append[Append record + predictions to history]
    Append --> DelayRate[Compute per-model predicted delay rate]
    DelayRate --> Loop

    %% Phase 6: Output
    SaveImages --> OutputDone[Output Complete]
    DelayRate --> SaveDet[Save delay_detections.csv to output/detections/]
    SaveDet --> OutputDone

    %% Dashboard
    OutputDone --> Dashboard[Streamlit Dashboard<br/>dashboard/app.py]
    Dashboard --> KPI[KPIs: records, delayed, delay rate, avg delay]
    Dashboard --> Charts[Charts: delay over time,<br/>model delay share, vehicle class breakdown]
    Dashboard --> Table[Recent detections table]
    Charts --> End([End])
    Table --> End

    %% Styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px
    classDef phase fill:#1f3b57,stroke:#1f3b57,color:#fff,stroke-width:3px
    classDef model fill:#e8f4fd,stroke:#1f77b4,stroke-width:2px
    classDef output fill:#e8f8e8,stroke:#2ca02c,stroke-width:2px
    
    class VC,Gen,Prep,Models,Eval,StreamPred phase
    class LR,RF,GB,KNN,NB,XGB model
    class SaveModels,SaveTable,SaveImages,SaveDet,OutputDone output
```

## Flow Summary

```
Vehicle Classes → Data Generator → Preprocessing → 6 Model Training → 
Live Streaming Detection → Output (tables, images, detections) → Dashboard
```

### Key Decision Points
- **Historical vs Live mode:** Historical data trains the models; live data produces real-time predictions
- **Delay threshold:** `delay_minutes >= 10` → `delay_label = 1` (significantly delayed)
- **Vehicle class:** Bus / Streetcar / Subway each have different base delays and weather sensitivity
