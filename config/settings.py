# config/settings.py
import os

BASE_DIR_SETUP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(BASE_DIR_SETUP, "data", "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

class ProjectConfig:
    # --- PATHS ---
    BASE_DIR = BASE_DIR_SETUP
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts", "part1")
    
    OUT_DIR_PART2 = os.path.join(BASE_DIR, "data", "artifacts", "part2")
    OUT_DIR_FORECAST = os.path.join(BASE_DIR, "data", "artifacts", "forecasting", "fixed")
    OUT_DIR_DIAGNOSTICS = os.path.join(BASE_DIR, "data", "artifacts", "diagnostics")
    OUT_DIR_PROCUREMENT = os.path.join(BASE_DIR, "data", "artifacts", "procurement")
    OUT_DIR_LOGISTICS = os.path.join(BASE_DIR, "data", "artifacts", "vrp_route_maps")
    OUT_DIR_ANALYSIS = os.path.join(BASE_DIR, "data", "artifacts", "analysis_report")

    # --- GLOBAL --- 
    SEED = 2025
    HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
    HOURS = list(range(6, 22))
    KEEP_SUBSET = True
    VOLUME_SHARE = 0.50
    # Reduced for feasibility of local integrated optimization
    PAIR_LIMIT = 12500 
    CO_FALLBACK = 1.0 
    GLOBAL_NUM_STORES = 20
    
    # --- PART 2: DEMAND RECONSTRUCTION ---
    COV_MIN_DAYS = 90
    COV_MIN_WEEKS = 8
    COV_MIN_NONOOS = 60
    COV_MIN_PROMO = 30
    COV_VERBOSE = True
    FLAG_STOCKOUT_VAL = 1
    KS_THR_HARD = 0.20
    KS_THR_SOFT = 0.65
    MIN_WEEKS_FOR_L1 = 2
    MIN_WEEKS_KEY = 1    
    MIN_DAYS_KEY = 6
    SHRINK_K_L1 = 4
    SHRINK_K_L2 = 12
    CDF_MIN_CLIP = 0.15
    USE_PROMO_KEY = True  
    USE_EVENT_KEY = True  
    HOURLY_FLOOR_PCTL = 5
    RECENSOR_SAMPLE_N = 5000
    RECENSOR_PEAK_Q = 0.75

    # --- PART 3: FORECASTING ---
    SEQ_LEN = 28
    HORIZON = 7
    MAX_PAIRS = 1000
    MAX_SAMPLES = 200_000
    MIN_DAYS_PAIR = 90
    LGB_PARAMS = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "lambda_l1": 1.0, "verbosity": -1, "seed": 2025, "device": "cpu"}

    # --- PART 4: INVENTORY PLANNING ---
    ECONOMICS_MODE = "proxy"
    DAILY_HOLDING_RATE_PCT = 0.1 
    HOLDING_COST_MULTIPLIER = 1.0 
    
    # Inventory Policy (Tau mapping)
    TAU_BY_OOS = [(0.00, 0.10, 0.95), (0.10, 0.30, 0.85), (0.30, 0.50, 0.75), (0.50, 1.01, 0.60)]
    
    # Supply Chain Params
    TARGET_SUPPLY_DEMAND_RATIO = 1.2 
    MOQ_RANGE_UNITS = (50, 200)
    UNIT_WEIGHT = {101: 0.25, 102: 0.125, 201: 1.0, 202: 0.06}
    MIN_STD_FRACTION = 0.10
    
    # Synthetic Geo Data
    CENTER_LAT = 31.2304
    CENTER_LON = 121.4737
    STORE_RADIUS_KM = (1, 15)
    SPEED_KMPH = 35.0 
    
    # [High Contrast Zones]
    SUPPLIER_ZONES = [
        (2, 5, 50, 1.60, 3.0, "Wholesaler_Near"), 
        (2, 50, 150, 1.00, 1.0, "Distributor_Mid"), 
        (2, 150, 400, 0.40, 0.0, "Farm_Far")       
    ]
    
    PRODUCT_CATEGORIES = [
        (101, 1, "Fresh Strawberries", 0.40, 0.0018),
        (102, 1, "Fresh Blueberries", 0.38, 0.0016),
        (201, 2, "Whole Milk", 0.15, 0.0010),
        (202, 2, "Eggs", 0.12, 0.0014),
    ]
    SHELF_LIFE_BY_CAT = {1: 7, 2: 12} 
    PRICE_RANGE_BY_PRODUCT = {101: (4.5, 7.0), 102: (6.0, 9.0), 201: (1.1, 1.6), 202: (1.8, 3.0)}
    DEMAND_RANGE_BY_PRODUCT = {101: (50, 150), 102: (40, 120), 201: (200, 500), 202: (150, 400)}
    # --- [FIX] MISSING CONFIGS ADDED HERE ---
    SUPPLIERS_PER_CAT_MAX = 6
    ELAPSED_RANGE_BY_CAT = {1: (0, 2), 2: (0, 4)}
    # --- PART 5: PROCUREMENT OPTIMIZATION ---
    PROCURE_REVIEW_DAYS = 7.0
    SERVICE_LEVEL = 0.95 
    SHORTAGE_COST = 500.0
    FIXED_ORDER_COST = 50.0 
    TRANSPORT_COST_PER_KG_KM = 0.015 
    MAX_SOLVE_TIME_S = 900
    GAP_REL = 0.02
    ROUND_Q_TO_INT = True
    ALLOW_SHORTAGE = True
    VERBOSE_SOLVER = True
    
    # --- PART 6: LOGISTICS VRP ---
    VRP_MAX_ROUTE_DISTANCE_KM = 600 
    VRP_SEARCH_TIME_LIMIT_SEC = 60
    
    VEHICLE_FLEET_DEFINITIONS = [
        {"type": "Small",  "capacity": 1000,   "fixed_cost": 300.0,  "cost_km": 0.5, "count": 15},
        {"type": "Medium", "capacity": 3000,   "fixed_cost": 500.0, "cost_km": 0.8, "count": 10},
        {"type": "Large",  "capacity": 8000,   "fixed_cost": 900.0, "cost_km": 1.2, "count": 5},
    ]
    VEHICLES_CHECK = [("Small", 1000.0), ("Medium", 3000.0), ("Large", 8000.0)]
    N_SUPPLIERS = 6
    
    # Time Windows
    STORE_OPEN_WINDOW = (360, 600)    
    SUPPLIER_OPEN_WINDOW = (420, 960) 
    WAREHOUSE_WINDOW = (240, 1560)    
    
    SERVICE_TIME_STORE_MINS = 20
    SERVICE_TIME_SUPPLIER_MINS = 45
    SERVICE_TIME_CROSSDOCK_MINS = 90

    # --- PART 7: INTEGRATED MODEL ---
    FRESHNESS_PENALTY_PER_DAY = 0.05 
    
    STRATEGIC_SCENARIOS = [
        {"name": "Hyper-Fresh", "p": 2, "u": 2, "desc": "High Cost, Max Freshness"},
        {"name": "Local-Batch", "p": 2, "u": 5, "desc": "Local Sourcing, Min Logistics"},
        {"name": "Balanced", "p": 3, "u": 3, "desc": "Mid-range Sourcing"},
        {"name": "Bulk-Farm", "p": 4, "u": 3, "desc": "Farm Sourcing, Optimized Logistics"},
        {"name": "JIT-Farm", "p": 5, "u": 2, "desc": "Farm Sourcing, High Frequency"}
    ]
    
    # --- DATASET ANALYSIS CONFIG ---
    SBC_ADI_THRESHOLD = 1.32
    SBC_CV2_THRESHOLD = 0.49
    INTRA_HOUR_START = 6
    INTRA_HOUR_END = 21
    SIMULATION_VOL_THRESHOLDS = [1, 2, 3, 5, 8, 10]