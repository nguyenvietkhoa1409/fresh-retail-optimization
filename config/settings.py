# config/settings.py
import os

# --- PATH SETUP ---
BASE_DIR_SETUP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(BASE_DIR_SETUP, "data", "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR

class ProjectConfig:
    BASE_DIR = BASE_DIR_SETUP
    ARTIFACTS_DIR = os.path.join(BASE_DIR, "data", "artifacts", "part1")
    
    # --- Global ---
    SEED = 2025
    HF_DATASET = "Dingdong-Inc/FreshRetailNet-50K"
    HOURS = list(range(6, 22))
    KEEP_SUBSET = True
    VOLUME_SHARE = 0.50
    PAIR_LIMIT = 1200
    
    # --- Coverage ---
    COV_MIN_DAYS = 90
    COV_MIN_WEEKS = 8
    COV_MIN_NONOOS = 60
    COV_MIN_PROMO = 30
    COV_VERBOSE = True
    
    # --- Economics ---
    ECONOMICS_MODE = "proxy"
    CO_FALLBACK = 1.0
    TAU_BY_OOS = [(0.00, 0.10, 0.70), (0.10, 0.30, 0.75), (0.30, 0.50, 0.82), (0.50, 1.01, 0.90)]
    
    # --- Synthetic Data ---
    CENTER_LAT = 31.2304
    CENTER_LON = 121.4737
    STORE_RADIUS_KM = (1, 15)
    SPEED_KMPH = 35.0
    N_SUPPLIERS = 25
    SUPPLIER_RINGS = [(5, 20, 50), (10, 50, 150), (10, 150, 300)]
    
    # Products (Base Params)
    PRODUCT_CATEGORIES = [
        (101, 1, "Fresh Strawberries", 0.40, 0.0018),
        (102, 1, "Fresh Blueberries", 0.38, 0.0016),
        (201, 2, "Whole Milk", 0.15, 0.0010),
        (202, 2, "Eggs", 0.12, 0.0014),
    ]
    SHELF_LIFE_BY_CAT = {1: 5, 2: 10}
    PRICE_RANGE_BY_PRODUCT = {101: (4.5, 7.0), 102: (6.0, 9.0), 201: (1.1, 1.6), 202: (1.8, 3.0)}
    
    # --- (3) INCREASE HOLDING COST ---
    # Multiplier to simulate High Risk/High Opportunity Cost of Capital for Fresh Food
    HOLDING_COST_MULTIPLIER = 5.0 

    # --- (1) ADJUST DEMAND SCALE (20x) ---
    # "Goldilocks" Zone: Big enough for trucks, small enough that fixed costs matter.
    DEMAND_RANGE_BY_PRODUCT = {
        101: (40, 160),   
        102: (40, 140),   
        201: (160, 400),  
        202: (100, 300)   
    }
    
    SUPPLIERS_PER_CAT_MAX = 10
    ELAPSED_RANGE_BY_CAT = {1: (0, 2), 2: (0, 4)}
    VEHICLES = [("LightVan", 1000, 8.0, 45.0, 0.85), ("MediumTruck", 2000, 16.0, 80.0, 1.10)]

    # --- Part 2 & 3 ---
    OUT_DIR_PART2 = os.path.join(BASE_DIR, "data", "artifacts", "part2")
    FLAG_STOCKOUT_VAL = 1
    KS_THR_HARD = 0.20
    KS_THR_SOFT = 0.65
    MIN_WEEKS_FOR_L1 = 2
    MIN_WEEKS_KEY = 1    
    MIN_DAYS_KEY = 6
    SHRINK_K_L1 = 4
    SHRINK_K_L2 = 12
    CDF_MIN_CLIP = 0.25
    USE_PROMO_KEY = True
    USE_EVENT_KEY = True
    HOURLY_FLOOR_PCTL = 5
    RECENSOR_SAMPLE_N = 5000
    RECENSOR_PEAK_Q = 0.75

    OUT_DIR_FORECAST = os.path.join(BASE_DIR, "data", "artifacts", "forecasting", "fixed")
    SEQ_LEN = 28
    HORIZON = 7
    MAX_PAIRS = 1000
    MAX_SAMPLES = 200_000
    MIN_DAYS_PAIR = 90
    LGB_PARAMS = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "lambda_l1": 1.0, "verbosity": -1, "seed": 2025, "device": "cpu"}

    # --- Part 4: Inventory ---
    OUT_DIR_DIAGNOSTICS = os.path.join(BASE_DIR, "data", "artifacts", "diagnostics")
    TARGET_SUPPLY_DEMAND_RATIO = 1.5
    MOQ_RANGE_UNITS = (1, 8)
    UNIT_WEIGHT = {101: 0.25, 102: 0.125, 201: 1.0, 202: 0.06}
    REVIEW_PERIOD_DAYS = 7.0
    MIN_STD_FRACTION = 0.10
    VEHICLES_CHECK = [("LightVan", 1000.0), ("MediumTruck", 2000.0)]

    # --- Part 5: Procurement ---
    OUT_DIR_PROCUREMENT = os.path.join(BASE_DIR, "data", "artifacts", "procurement")
    PROCURE_REVIEW_DAYS = 7.0
    SERVICE_LEVEL = 0.98
    SHORTAGE_COST = 500.0
    
    # Fixed Cost: $80 is roughly 8-10% of a typical 20x order. Healthy balance.
    FIXED_ORDER_COST = 80.0 
    
    # Transport: High to ensure VRP efficiency matters.
    TRANSPORT_COST_PER_KG_KM = 0.04 
    
    MAX_SOLVE_TIME_S = 900
    GAP_REL = 0.02
    ROUND_Q_TO_INT = True
    ALLOW_SHORTAGE = True
    VERBOSE_SOLVER = True

    # --- Part 6: Logistics VRP ---
    OUT_DIR_LOGISTICS = os.path.join(BASE_DIR, "data", "artifacts", "vrp_route_maps")
    
    VRP_MAX_ROUTE_DISTANCE_KM = 400
    
    # FLEET MIX: Optimized for 20x scale (100kg - 2000kg loads)
    VEHICLE_FLEET_DEFINITIONS = [
        {"type": "Small",  "capacity": 1000,   "fixed_cost": 80.0,  "cost_km": 0.8, "count": 25},
        {"type": "Medium", "capacity": 3000,   "fixed_cost": 120.0, "cost_km": 1.2, "count": 20},
        {"type": "Large",  "capacity": 8000,   "fixed_cost": 180.0, "cost_km": 1.6, "count": 10},
        {"type": "Extra",  "capacity": 15000,  "fixed_cost": 250.0, "cost_km": 2.2, "count": 5}
    ]
    
    VRP_SEARCH_TIME_LIMIT_SEC = 60

    # --- Part 7: Reporting ---
    OUT_DIR_ANALYSIS = os.path.join(BASE_DIR, "data", "artifacts", "analysis_report")