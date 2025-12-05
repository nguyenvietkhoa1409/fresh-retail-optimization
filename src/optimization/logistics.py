import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
from config.settings import ProjectConfig as Cfg
from src.utils.geo import GeoUtils

class IntegratedVRPSolver:
    def __init__(self, depot_loc, stops_df, role="Inbound", earliest_start_time=0):
        self.depot_loc = depot_loc
        self.stops_df = stops_df
        self.role = role
        self.earliest_start_time = int(earliest_start_time)
        
        # Flatten Fleet Config
        self.vehicles = []
        for v in Cfg.VEHICLE_FLEET_DEFINITIONS:
            self.vehicles.extend([v] * v['count'])
        self.vehicles.sort(key=lambda x: x['capacity']) # Sort small to large

    def solve(self):
        self.data = self._create_data()
        self.manager = pywrapcp.RoutingIndexManager(len(self.data['dm']), len(self.vehicles), 0)
        self.routing = pywrapcp.RoutingModel(self.manager)
        
        # --- 1. Dimension: Distance ---
        def dist_cb(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return int(self.data['dm'][from_node][to_node])
            
        transit_callback_index = self.routing.RegisterTransitCallback(dist_cb)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        self.routing.AddDimension(
            transit_callback_index, 0, int(Cfg.VRP_MAX_ROUTE_DISTANCE_KM * 1000), True, "Distance"
        )
        
        # --- 2. Dimension: Capacity ---
        def cap_cb(from_index):
            from_node = self.manager.IndexToNode(from_index)
            return int(self.data['demands'][from_node])
            
        cap_callback_index = self.routing.RegisterUnaryTransitCallback(cap_cb)
        self.routing.AddDimensionWithVehicleCapacity(
            cap_callback_index, 0, [int(v['capacity']) for v in self.vehicles], True, "Capacity"
        )
        
        # --- 3. Dimension: Time ---
        def time_cb(from_index, to_index):
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            val = self.data['tm'][from_node][to_node] + self.data['service'][from_node]
            return int(val)
            
        time_callback_index = self.routing.RegisterTransitCallback(time_cb)
        self.routing.AddDimension(time_callback_index, 2880, 2880, False, "Time")
        time_dim = self.routing.GetDimensionOrDie("Time")
        
        # Time Windows
        for i, (start, end) in enumerate(self.data['tw']):
            if i == 0: continue
            index = self.manager.NodeToIndex(i)
            time_dim.CumulVar(index).SetRange(int(start), int(end))
            
        # Depot Constraints
        depot_idx = self.manager.NodeToIndex(0)
        depot_open, depot_close = int(self.data['tw'][0][0]), int(self.data['tw'][0][1])
        
        if self.role == "Outbound":
            actual_start = max(depot_open, self.earliest_start_time)
            if actual_start > depot_close:
                depot_close = actual_start + 60 
            time_dim.CumulVar(depot_idx).SetRange(actual_start, depot_close)
        else:
            time_dim.CumulVar(depot_idx).SetRange(depot_open, depot_close)

        # Solve
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.time_limit.seconds = Cfg.VRP_SEARCH_TIME_LIMIT_SEC
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        
        self.solution = self.routing.SolveWithParameters(search_params)
        return self._extract_routes() if self.solution else []

    def _create_data(self):
        locs = [self.depot_loc] + list(zip(self.stops_df['lat'], self.stops_df['lon']))
        demands = [0] + [int(math.ceil(d)) for d in self.stops_df['demand_kg']]
        ids = ["WAREHOUSE"] + self.stops_df['id'].astype(str).tolist()
        
        if 'tw_open' in self.stops_df.columns:
            tw_list = list(zip(self.stops_df['tw_open'], self.stops_df['tw_close']))
        else:
            tw_list = [(0, 1440)] * len(self.stops_df)
        tw = [Cfg.WAREHOUSE_WINDOW] + tw_list
        
        if 'service_time' in self.stops_df.columns:
            srv_list = self.stops_df['service_time'].tolist()
        else:
            srv_list = [15] * len(self.stops_df)
        service = [0] + srv_list
        
        n = len(locs)
        dm = [[0]*n for _ in range(n)]
        tm = [[0]*n for _ in range(n)]
        
        avg_speed_mpm = (Cfg.SPEED_KMPH * 1000) / 60.0
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    km = GeoUtils.haversine_km(locs[i][0], locs[i][1], locs[j][0], locs[j][1])
                    dist_m = int(km * 1000)
                    dm[i][j] = dist_m
                    tm[i][j] = int((dist_m / avg_speed_mpm) * 1.2)
                    
        return {'dm': dm, 'tm': tm, 'demands': demands, 'tw': tw, 'service': service, 'locs': locs, 'ids': ids}

    def _extract_routes(self):
        routes = []
        time_dim = self.routing.GetDimensionOrDie("Time")
        
        for v_idx in range(len(self.vehicles)):
            index = self.routing.Start(v_idx)
            if self.routing.IsEnd(self.solution.Value(self.routing.NextVar(index))): continue
            
            steps = []; load = 0; dist = 0
            while not self.routing.IsEnd(index):
                node = self.manager.IndexToNode(index)
                load += self.data['demands'][node]
                t_val = self.solution.Min(time_dim.CumulVar(index))
                
                steps.append({
                    "id": self.data['ids'][node], 
                    "lat": self.data['locs'][node][0], 
                    "lon": self.data['locs'][node][1],
                    "arrival_time_min": t_val, 
                    "arrival_time_fmt": f"{int(t_val//60):02d}:{int(t_val%60):02d}",
                    "type": "DEPOT" if node == 0 else ("SUPPLIER" if self.role == "Inbound" else "STORE")
                })
                prev = index; index = self.solution.Value(self.routing.NextVar(index))
                dist += self.data['dm'][self.manager.IndexToNode(prev)][self.manager.IndexToNode(index)]
            
            t_end = self.solution.Min(time_dim.CumulVar(index))
            steps.append({
                "id": "WAREHOUSE", 
                "lat": self.data['locs'][0][0], 
                "lon": self.data['locs'][0][1], 
                "arrival_time_min": t_end, 
                "arrival_time_fmt": f"{int(t_end//60):02d}:{int(t_end%60):02d}",
                "type": "DEPOT"
            })
            
            v = self.vehicles[v_idx]
            km = dist / 1000.0
            cost = v['fixed_cost'] + (km * v['cost_km'])
            
            routes.append({
                "vehicle_id": v_idx,
                "role": self.role,
                "vehicle_type": v['type'], 
                "steps": steps, 
                "distance_km": km, 
                "total_load_kg": load, 
                "cost": cost, 
                "utilization_pct": round(load/v['capacity']*100, 1),
                "end_time": t_end
            })
        return routes

class RouteVisualizer:
    @staticmethod
    def plot_routes(routes, depot_loc, role, save_dir=None):
        if not routes: return
        if save_dir is None: save_dir = Cfg.OUT_DIR_LOGISTICS # Fallback
        plt.figure(figsize=(10, 8))
        plt.scatter(depot_loc[1], depot_loc[0], c='black', s=200, marker='P', label='Depot', zorder=10)
        cmap = plt.get_cmap('tab10')
        
        for i, r in enumerate(routes):
            lats = [s['lat'] for s in r['steps']]
            lons = [s['lon'] for s in r['steps']]
            color = cmap(i % 10)
            plt.plot(lons, lats, color=color, lw=2, alpha=0.7, label=f"{r['vehicle_type']}")
            plt.scatter(lons[1:-1], lats[1:-1], color=color, s=50, edgecolors='white', zorder=5)
            
        plt.title(f"{role} Routes")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"route_map_{role.lower()}.png"))
        plt.close()

class LogisticsManager:
    def __init__(self, override_output_dir=None):
        # [MODIFIED] Support dynamic output path
        if override_output_dir:
            self.out_dir = override_output_dir
            os.makedirs(self.out_dir, exist_ok=True)
        else:
            self.out_dir = Cfg.OUT_DIR_LOGISTICS
            os.makedirs(self.out_dir, exist_ok=True)
            
        self.wh_loc = (Cfg.CENTER_LAT, Cfg.CENTER_LON)

    def run(self):
        agg_path = os.path.join(Cfg.OUT_DIR_PROCUREMENT, "procurement_summary_by_supplier_store.csv")
        if not os.path.exists(agg_path): return
        df_agg = pd.read_csv(agg_path)
        
        df_sup = pd.read_csv(os.path.join(Cfg.ARTIFACTS_DIR, "suppliers_harmonized_v2.csv"))
        df_store = pd.read_parquet(os.path.join(Cfg.ARTIFACTS_DIR, "unified_for_procurement.parquet"))
        
        # INBOUND
        in_dem = df_agg.groupby('supplier_id')['total_kg'].sum().reset_index()
        in_dem['supplier_id'] = in_dem['supplier_id'].astype(int)
        df_sup['supplier_id'] = df_sup['supplier_id'].astype(int)
        in_dem = in_dem.merge(df_sup, on='supplier_id').rename(columns={'supplier_id': 'id', 'sup_lat': 'lat', 'sup_lon': 'lon', 'total_kg': 'demand_kg'})
        
        in_solver = IntegratedVRPSolver(self.wh_loc, in_dem, "Inbound", 240)
        in_routes = in_solver.solve()
        RouteVisualizer.plot_routes(in_routes, self.wh_loc, "Inbound", save_dir=self.out_dir)
        
        last_arr = max([r['steps'][-1]['arrival_time_min'] for r in in_routes]) if in_routes else 240
        out_start = last_arr + Cfg.SERVICE_TIME_CROSSDOCK_MINS
        
        is_shifted_next_day = False
        if out_start > 540: 
            print(f"  -> [Logistics] Arrival at {int(out_start//60)}:{int(out_start%60):02d} > Morning Window. Standard Next-Day Delivery applied.")
            out_start = 360 
            is_shifted_next_day = True
        
        # OUTBOUND
        df_store['store_id'] = df_store['store_id'].astype(str)
        df_agg['store_id'] = df_agg['store_id'].astype(str)
        cols_to_merge = ['store_id','store_lat','store_lon','tw_open','tw_close', 'service_time']
        out_dem = df_agg.groupby('store_id')['total_kg'].sum().reset_index().merge(
            df_store[cols_to_merge].drop_duplicates(), on='store_id'
        ).rename(columns={'store_id': 'id', 'store_lat': 'lat', 'store_lon': 'lon', 'total_kg': 'demand_kg'})
        
        out_solver = IntegratedVRPSolver(self.wh_loc, out_dem, "Outbound", out_start)
        out_routes = out_solver.solve()
        RouteVisualizer.plot_routes(out_routes, self.wh_loc, "Outbound", save_dir=self.out_dir)
        
        all_r = in_routes + out_routes
        total_cost = sum(r['cost'] for r in all_r)
        
        if not all_r: total_cost = 0.0
        
        # [CRITICAL FIX] THIS IS THE MISSING PART THAT SAVES THE ROUTE FILE
        pd.DataFrame(all_r).to_csv(os.path.join(self.out_dir, "vrp_routes_solution.csv"), index=False)            
        pd.DataFrame([{
            'Cost_USD': total_cost, 
            'Crossdock_Ready_Time': f"{int(out_start//60)}:{int(out_start%60):02d}",
            'Is_Next_Day': is_shifted_next_day
        }]).to_csv(os.path.join(self.out_dir, "vrp_summary.csv"), index=False)