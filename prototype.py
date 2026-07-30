import math
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from typing import List, Dict, Tuple, Any

# 수원시  저상버스 노선 
LOW_FLOOR_BUS_ROUTES = {
    '13', '13-4', '13-5', '19', '2-1', '20-1', '25', '27', '3', '30',
    '30-1', '35', '37', '42', '5', '62-1', '7-1', '80', '82-1', '83-1',
    '88', '9', '92', '92-1', '98', '99', '99-2'
}

# 교통 변수
FEATURE_KEYS = [
    'travel_time',              # 1. 소요 시간 
    'transfers',                # 2. 환승 횟수 
    'low_floor_bus_ratio',      # 3. (버스 이용 시) 저상버스 비율 
    'walk_distance',            # 4. 전체 도보 이동 거리 
    'bad_mobility_walk_dist',   # 5. 이동편의 지수 '나쁨' 구간 도보 거리 
    'step_count',               # 6. 도보 중  '단차' 보행 불편 지점 수 
    'obstacle_count',           # 7. 도보 중 '장애물' 보행 불편 지점 수 
    'damaged_tactile_count',    # 8. 도보 중 '점자블록파손' 보행 불편 지점 수
    'slope_count',              # 9. 도보 중 '경사' 보행 불편 지점 수 
    'wheel_trap_count',         # 10. 도보 중 '바퀴 빠짐 주의' 보행 불편 지점 수 
    'narrow_road_count'         # 11. 도보 중 '폭 좁음' 보행 불편 지점 수 
]


# 보행맵 데이터(.KMZ) 파싱
class AccessibilityDataParser:    
    def __init__(self, kmz_file_path: str):
        self.kmz_path = kmz_file_path
        self.obstacles: List[Dict[str, Any]] = []
        self.bad_mobility_lines: List[List[Tuple[float, float]]] = []
        self._parse_kmz()

    def _parse_kmz(self):
        try:
            with zipfile.ZipFile(self.kmz_path, 'r') as kmz:
                kml_data = kmz.read('doc.kml')
                root = ET.fromstring(kml_data)
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}
                for placemark in root.findall('.//kml:Placemark', ns):
                    name_elem = placemark.find('kml:name', ns)
                    desc_elem = placemark.find('kml:description', ns)
                    name = name_elem.text if name_elem is not None else ""
                    desc = desc_elem.text if desc_elem is not None else ""
                    full_text = f"{name} {desc}"
                    point = placemark.find('.//kml:Point/kml:coordinates', ns)
                    if point is not None and point.text:
                        coords = point.text.strip().split(',')
                        lng, lat = float(coords[0]), float(coords[1])
                        obs_type = self._classify_obstacle(full_text)
                        if obs_type:
                            self.obstacles.append({
                                'type': obs_type, 
                                'lat': lat, 
                                'lng': lng
                            })
                    line = placemark.find('.//kml:LineString/kml:coordinates', ns)
                    if line is not None and line.text:
                        # 설명문 내에 '나쁨' 또는 'bad' 키워드가 포함된 선형 경로 추출
                        if '나쁨' in full_text or 'bad' in full_text.lower():
                            coord_pairs = []
                            # 공백 기준으로 좌표 쌍 분할
                            raw_coords = line.text.strip().split()
                            for c in raw_coords:
                                parts = c.split(',')
                                # 내부 다루기 편하게 (위도, 경도) 순서로 변경 저장
                                coord_pairs.append((float(parts[1]), float(parts[0])))
                            
                            self.bad_mobility_lines.append(coord_pairs)
                            
        except Exception as e:
            print(f"[경고] KMZ/KML 파일 파싱 오류 발생: {e}")

    def _classify_obstacle(self, text: str) -> str:
        if '단차' in text: 
            return 'step_count'
        if '장애물' in text: 
            return 'obstacle_count'
        if '점자' in text : 
            return 'damaged_tactile_count'
        if '경사' in text: 
            return 'slope_count'
        if '바퀴' in text: 
            return 'wheel_trap_count'
        if '폭' in text or '좁음' in text: 
            return 'narrow_road_count'
        return None


# Google Route API 호출 후 검색된 경로 좌표화 및 주변 탐색 
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0 
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    index, lat, lng = 0, 0, 0
    coordinates = []
    length = len(polyline_str)

    while index < length:
        b, shift, result = 0, 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat

        shift, result = 0, 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng

        coordinates.append((lat / 1e5, lng / 1e5))

    return coordinates

class GoogleRouteService:
    
    def __init__(self, api_key: str, spatial_db: AccessibilityDataParser):
        self.api_key = api_key
        self.spatial_db = spatial_db

    def get_top5_routes(self, origin: str, destination: str) -> List[Dict[str, float]]:
        url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": origin,
            "destination": destination,
            "mode": "transit",         
            "alternatives": "true",    
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
        except Exception:
            data = {'status': 'ERROR'}


        routes = data.get('routes', [])[:5]  
        processed_routes = []


        # 소요 시간, 환승 횟수
        for route in routes:
            leg = route['legs'][0]
            travel_time = leg['duration']['value'] / 60.0 
            walk_dist = 0.0
            transfers = 0
            bus_lines = []
            walk_polylines = []

            for step in leg['steps']:
                travel_mode = step['travel_mode']
                
                if travel_mode == 'WALKING':
                    walk_dist += step['distance']['value']
                    walk_polylines.append(step['polyline']['points'])
                    
                elif travel_mode == 'TRANSIT':
                    transfers += 1  
                    details = step.get('transit_details', {})
                    vehicle = details.get('line', {}).get('vehicle', {}).get('type', '')
                    line_name = details.get('line', {}).get('short_name', '')
                    
                    if vehicle == 'BUS':
                        bus_lines.append(line_name)

            transfers = max(0, transfers - 1)

            # (버스 이용 시) 저상버스 이용 비율 
            # ------------------------------------------------------------------
            if bus_lines:
                low_floor_count = sum(1 for bus in bus_lines if bus in LOW_FLOOR_BUS_ROUTES)
                low_floor_ratio = low_floor_count / len(bus_lines)
            else:
                # 버스를 타지 않는 경로는 1로 처리 
                low_floor_ratio = 1.0

            # 이동 편의 지수 및 보행 불편 지점 수
            spatial_counts = self._analyze_walk_path(walk_polylines)

            route_features = {
                'travel_time': travel_time,
                'transfers': float(transfers),
                'low_floor_bus_ratio': low_floor_ratio,
                'walk_distance': walk_dist,
                'bad_mobility_walk_dist': spatial_counts['bad_mobility_dist'],
                'step_count': float(spatial_counts['step_count']),
                'obstacle_count': float(spatial_counts['obstacle_count']),
                'damaged_tactile_count': float(spatial_counts['damaged_tactile_count']),
                'slope_count': float(spatial_counts['slope_count']),
                'wheel_trap_count': float(spatial_counts['wheel_trap_count']),
                'narrow_road_count': float(spatial_counts['narrow_road_count'])
            }
            processed_routes.append(route_features)

        return processed_routes

    def _analyze_walk_path(self, polylines: List[str]) -> Dict[str, float]:
        counts = {
            'bad_mobility_dist': 0.0, 'step_count': 0, 'obstacle_count': 0,
            'damaged_tactile_count': 0, 'slope_count': 0, 'wheel_trap_count': 0,
            'narrow_road_count': 0
        }
        
        walk_coords = []
        for poly in polylines:
            walk_coords.extend(decode_polyline(poly))

        if not walk_coords:
            return counts

        for obs in self.spatial_db.obstacles:
            for w_lat, w_lng in walk_coords:
                if haversine_distance(w_lat, w_lng, obs['lat'], obs['lng']) <= 15.0:
                    counts[obs['type']] += 1
                    break  

        return counts




class RouteRecommendationEngine:
    def __init__(self):
        #  초기 가중치 
        # 패널티 점수이기 때문에 저상버스 비율은 가중치가 음수(나머지는 양수)
        self.default_weights = np.array([0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        # 데이터 저장소
        self.user_type_models: Dict[str, LogisticRegression] = {}

   # 교통변수들은 min-max 정규화 후 사용 
    def normalize_features(self, routes_data: List[Dict[str, float]]) -> np.ndarray:
        raw_matrix = np.array([[r[k] for k in FEATURE_KEYS] for r in routes_data])
        min_vals = raw_matrix.min(axis=0)
        max_vals = raw_matrix.max(axis=0)
        range_vals = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        normalized_matrix = (raw_matrix - min_vals) / range_vals
        
        return normalized_matrix

    def rank_routes(self, routes_data: List[Dict[str, float]], user_type: str = None) -> List[Dict[str, Any]]:
        
        # 행렬 정규화 수행 (score 계산을 위해 추가)
        norm_matrix = self.normalize_features(routes_data)
        
        # 이전 동일 유형 사용자의 회귀 계수를 가중치로 사용  
        if user_type and user_type in self.user_type_models:
            weights = self.user_type_models[user_type].coef_[0]
        else:
            weights = self.default_weights

        scores = np.dot(norm_matrix, weights)
        
        results = []
        for idx, (route, score) in enumerate(zip(routes_data, scores)):
            results.append({
                'route_id': idx + 1,
                'score': round(float(score), 4),
                'raw_features': route
            })
            
        # 페널티 점수 기준 오름차순 정렬
        results.sort(key=lambda x: x['score'])
        return results

    # 가중치 갱신
    def update_weights_from_feedback(self, user_type: str, feedback_history: List[Dict]):

        # x = 교통 변수 
        X = np.array([item['norm_features'] for item in feedback_history])
        
        # y = 경로 선택 여부 (1 또는 0)
        y = np.array([item['selected'] for item in feedback_history])

       # 회귀 계수 계산
        model = LogisticRegression(fit_intercept=False, solver='lbfgs')
        model.fit(X, y)

        # 갱신된 가중치 저장 
        self.user_type_models[user_type] = model
        
        print(f"\n[업데이트 완료] '{user_type}' 유형의 새로운 회귀 계수(가중치):")
        for key, weight in zip(FEATURE_KEYS, model.coef_[0]):
            print(f"  - {key:<22}: {weight:+.4f}")


# 실행 
# 실행 구문 (이 부분만 교체하시면 됩니다)
if __name__ == "__main__":
    kmz_parser = AccessibilityDataParser("doc.kmz")
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  
    route_service = GoogleRouteService(GOOGLE_API_KEY, kmz_parser)
    
    # [핵심] 추천 엔진 객체는 반복문 "밖"에서 한 번만 생성해야 가중치 학습 결과가 유지됩니다.
    recommend_engine = RouteRecommendationEngine()

    while True:
        print(" [교통약자 맞춤형 경로 추천 시스템] ")
        print(" 1) 휠체어 이용자 | 2) 시각장애인 | 3) 유모차 이용자 | 4) 고령자")
        print(" (종료하려면 'q' 입력)")

        # 사용자 유형
        choice = input("사용자 유형을 선택하세요 (1~4 / q): ").strip().lower()
        if choice == 'q':
            print("프로그램을 종료합니다.")
            break

        user_type_map = {
            '1': 'wheelchair',
            '2': 'blind',
            '3': 'stroller',
            '4': 'elder'
        }
        current_user_type = user_type_map.get(choice)
        if not current_user_type:
            print("잘못된 번호입니다. 다시 선택해주세요.")
            continue

        # 출발지, 목적지
        origin = input("출발지를 입력하세요 (종료: q): ").strip()
        if origin.lower() == 'q': break
        
        destination = input("목적지를 입력하세요 (종료: q): ").strip()
        if destination.lower() == 'q': break

        routes_5 = route_service.get_top5_routes(origin, destination)
        if not routes_5:
            print("경로를 검색하지 못했습니다. 입력 정보를 확인해주세요.")
            continue

        ranked_routes = recommend_engine.rank_routes(routes_5, user_type=current_user_type)
        
        print(f"\n>>> [{current_user_type.upper()}] 추천 경로 결과 <<<")
        for rank, r in enumerate(ranked_routes, 1):
            f = r['raw_features']
            print(f"\n[{rank}위 추천] 경로 ID: {r['route_id']} (비선호 점수: {r['score']})")
            print(f"  - 소요시간 {f['travel_time']:.1f}분 | 환승 {int(f['transfers'])}회 | 전체 도보 {f['walk_distance']:.0f}m | 저상버스 비율 {f['low_floor_bus_ratio']*100:.0f}%")
            print(f"  - 불편 보도거리 {f['bad_mobility_walk_dist']:.0f}m | 단차 지점 {int(f['step_count'])}개 | 장애물 {int(f['obstacle_count'])}개")
            print(f"  - 파손 점자블록 {int(f['damaged_tactile_count'])}개 | 경사 {int(f['slope_count'])}개 | 바퀴끼임 위험 {int(f['wheel_trap_count'])}개 | 좁은 도로 {int(f['narrow_road_count'])}개")

        # 사용자 피드백 수집
        selected_id = input("\n실제로 이용할 경로의 ID(1~5)를 선택하세요 (학습 없이 넘어가려면 Enter): ").strip()

        if selected_id.isdigit() and 1 <= int(selected_id) <= len(routes_5):
            selected_id = int(selected_id)
            norm_matrix = recommend_engine.normalize_features(routes_5)
            
            feedback_data = []
            for idx, route in enumerate(routes_5):
                # 사용자가 선택한 경로는 1, 아닌 경로들은 0으로 라벨링
                is_selected = 1 if (idx + 1) == selected_id else 0
                feedback_data.append({
                    'norm_features': norm_matrix[idx],
                    'selected': is_selected
                })

            recommend_engine.update_weights_from_feedback(current_user_type, feedback_data)
