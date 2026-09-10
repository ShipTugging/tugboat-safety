"""
TUGGUARD - 예인선 거딩 위험 감지 파이프라인 (강영한 담당 파트)

흐름:
  Mask (YOLO-Seg 출력, 지금은 더미로 생성)
    -> Skeleton 추출
    -> SagRatio 계산
    -> Towline Angle(픽셀 기준, beta) 계산
    -> IMU Roll(alpha)로 보정 -> 실제 3D 각도(gamma)
    -> Confidence 기반 Vision/IMU Fusion
    -> 위험 상태 분류 (Normal -> Loaded -> Girting Risk -> Developing -> Critical)
    -> JSON 출력 (이후 파이프라인/UI로 전달될 형태)

지금은 실제 YOLO-Seg가 아니라 더미 mask 생성기로 전체 흐름이 끊김 없이
동작하는지만 확인한다. 나중에 generate_dummy_mask() 자리에 실제
YOLO-Seg 추론 결과(mask, confidence)를 꽂아 넣으면 된다.
"""

import json
import math
from collections import deque

import numpy as np
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# 0. 더미 Mask 생성기 (실제로는 YOLO-Seg가 이 자리를 대체함)
# ---------------------------------------------------------------------------
def generate_dummy_mask(sag_ratio: float, angle_deg: float,
                         shape=(480, 640), thickness: int = 3) -> np.ndarray:
    """
    처짐 정도(sag_ratio)와 화면상 각도(angle_deg)를 지정해서
    그럴듯한 예인줄 모양의 binary mask를 만든다.
    (실제 파이프라인에서는 이 함수 대신 YOLO-Seg의 출력 mask를 사용)
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # 화면 하단 중앙(예인선 선미)에서 화면 위쪽(상선 쪽)으로 줄이 뻗어나간다고 가정
    start = np.array([w * 0.5, h * 0.95])
    length = h * 0.7
    angle_rad = math.radians(angle_deg)
    end = start + np.array([math.sin(angle_rad), -math.cos(angle_rad)]) * length

    n_points = 200
    t = np.linspace(0, 1, n_points)
    # 직선 보간
    line = start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]
    # 중앙이 sag_ratio 만큼 옆으로 처지는 포물선 형태 추가
    perp = np.array([-(end - start)[1], (end - start)[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-6)
    bow = 4 * t * (1 - t)  # 0->1->0, 중앙에서 최대
    sag_pixels = sag_ratio * length
    curve = line + perp[None, :] * (bow[:, None] * sag_pixels)

    for x, y in curve:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            mask[max(0, yi - thickness):yi + thickness,
                 max(0, xi - thickness):xi + thickness] = 1

    return mask


# ---------------------------------------------------------------------------
# 1. Mask -> Skeleton
# ---------------------------------------------------------------------------
def extract_skeleton_points(mask: np.ndarray) -> np.ndarray:
    """binary mask를 1픽셀 폭 skeleton으로 깎고, 좌표(y,x) 배열로 반환."""
    skeleton = skeletonize(mask.astype(bool))
    ys, xs = np.nonzero(skeleton)
    points = np.stack([xs, ys], axis=1)  # (N, 2) -> (x, y)
    return points


def order_points_along_line(points: np.ndarray) -> np.ndarray:
    """
    skeleton 좌표는 순서가 없으므로, y좌표(세로 위치) 기준으로 정렬해서
    "선을 따라가는 순서"를 근사한다. (예인줄은 대체로 세로 방향으로 뻗어있다고 가정)
    """
    order = np.argsort(points[:, 1])
    return points[order]


# ---------------------------------------------------------------------------
# 2. SagRatio 계산
# ---------------------------------------------------------------------------
def compute_sag_ratio(points: np.ndarray) -> float:
    """
    SagRatio = MaxDeviation / EndpointDistance
    - 양 끝점을 잇는 직선 대비, skeleton이 가장 많이 벗어난 거리(MaxDeviation)
    - EndpointDistance: 양 끝점 사이 거리
    """
    if len(points) < 2:
        return 0.0

    p1, p2 = points[0].astype(float), points[-1].astype(float)
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-6:
        return 0.0
    line_unit = line_vec / line_len

    # 점-직선 거리 (외적의 크기)
    rel = points.astype(float) - p1[None, :]
    proj_len = rel @ line_unit
    proj_point = p1[None, :] + proj_len[:, None] * line_unit[None, :]
    dist = np.linalg.norm(points.astype(float) - proj_point, axis=1)

    max_deviation = float(np.max(dist))
    return max_deviation / line_len


# ---------------------------------------------------------------------------
# 3. Towline Angle 계산 (픽셀 기준, beta)
# ---------------------------------------------------------------------------
def compute_towline_angle_pixel(points: np.ndarray) -> float:
    """
    skeleton 양 끝점을 잇는 방향과, 예인선 중심축(화면 기준 수직 방향, y축)
    사이의 각도(도)를 반환. 0도 = 정중앙(centerline)과 일치.
    """
    if len(points) < 2:
        return 0.0
    # points는 y 오름차순 정렬 상태 -> points[-1]이 화면 하단(예인선 쪽, 시작점),
    # points[0]이 화면 상단(상선 쪽, 끝점)
    p1, p2 = points[-1].astype(float), points[0].astype(float)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # 화면 기준 수직(위쪽) 방향을 0도 기준으로 잡음
    beta = math.degrees(math.atan2(dx, -dy))
    return beta


# ---------------------------------------------------------------------------
# 4. IMU Roll 보정 (JMSE 2020 논문 공식 기반)
#    tan(gamma) = (y0*tan(alpha) - y0*tan(beta)) / (z0 + z0*tan(beta)*tan(alpha))
# ---------------------------------------------------------------------------
def correct_roll(beta_deg: float, roll_deg: float,
                  y0: float = 10.0, z0: float = 70.0) -> float:
    """
    beta_deg: 화면(픽셀)에서 잰 예인줄 각도
    roll_deg: IMU에서 받은 배의 Roll 각도 (alpha)
    y0, z0  : 카메라 설치 높이/거리 (캘리브레이션 값, 논문 예시 기본값)
    반환값  : Roll 왜곡을 보정한 3D 세계좌표계 상의 실제 예인줄 각도(gamma, 도)
    """
    beta = math.radians(beta_deg)
    alpha = math.radians(roll_deg)

    denom = z0 + z0 * math.tan(beta) * math.tan(alpha)
    if abs(denom) < 1e-6:
        return beta_deg  # 특이점 방지, 보정 없이 원값 반환

    tan_gamma = (y0 * math.tan(alpha) - y0 * math.tan(beta)) / denom
    gamma = math.degrees(math.atan(tan_gamma))
    return gamma


# ---------------------------------------------------------------------------
# 5. Confidence 기반 Vision/IMU Fusion 모드 결정
# ---------------------------------------------------------------------------
def decide_fusion_mode(confidence: float) -> str:
    if confidence > 0.7:
        return "vision_imu_fused"      # 정상 융합
    elif confidence > 0.3:
        return "imu_primary"           # 비전은 보조 신호만
    else:
        return "imu_only"              # 비전 배제, IMU 단독


# ---------------------------------------------------------------------------
# 6. 위험 상태 분류 (최근 프레임 이력 기반)
# ---------------------------------------------------------------------------
class RiskStateClassifier:
    """
    최근 N프레임의 SagRatio / Towline Angle(gamma) / Roll rate를 보고
    Normal -> Loaded -> GirtingRisk -> Developing -> Critical 단계를 판단.

    지금은 단순 threshold + 변화율 기반. 실측 데이터가 쌓이면
    임계값을 재조정해야 함 (지금 값은 초기 추정치).
    """

    STATES = ["Normal", "Loaded", "GirtingRisk", "Developing", "Critical"]

    def __init__(self, window: int = 5):
        self.window = window
        self.sag_history: deque = deque(maxlen=window)
        self.angle_history: deque = deque(maxlen=window)

    def update(self, sag_ratio: float, gamma_deg: float,
               roll_rate_deg_s: float, fusion_mode: str) -> str:
        self.sag_history.append(sag_ratio)
        self.angle_history.append(abs(gamma_deg))

        sag_now = self.sag_history[-1]
        angle_now = self.angle_history[-1]

        sag_rate = 0.0
        if len(self.sag_history) >= 2:
            # 처짐이 급격히 "줄어드는" 속도 = 줄이 급격히 팽팽해지는 속도
            sag_rate = self.sag_history[-2] - self.sag_history[-1]

        # IMU 단독 모드일 때는 비전 값(sag, angle) 대신 Roll rate만으로 보수적으로 판단
        if fusion_mode == "imu_only":
            if abs(roll_rate_deg_s) > 5.0:
                return "Developing"
            elif abs(roll_rate_deg_s) > 2.0:
                return "GirtingRisk"
            return "Normal"

        # --- 초기 threshold (실측 데이터로 추후 보정 필요) ---
        if angle_now > 40 and sag_rate > 0.05:
            return "Critical"
        if angle_now > 30 and sag_rate > 0.02:
            return "Developing"
        if angle_now > 20 or sag_rate > 0.01:
            return "GirtingRisk"
        if sag_now < 0.05:
            return "Loaded"
        return "Normal"


# ---------------------------------------------------------------------------
# 7. 프레임 단위 처리 함수 (전체 파이프라인 한 번 실행)
# ---------------------------------------------------------------------------
def process_frame(mask: np.ndarray, confidence: float, roll_deg: float,
                   roll_rate_deg_s: float, classifier: RiskStateClassifier,
                   timestamp: float) -> dict:
    fusion_mode = decide_fusion_mode(confidence)

    if fusion_mode == "imu_only":
        # 비전 값을 신뢰하지 않고 IMU만으로 판단
        sag_ratio, beta, gamma = None, None, None
    else:
        points = extract_skeleton_points(mask)
        points = order_points_along_line(points)
        sag_ratio = compute_sag_ratio(points)
        beta = compute_towline_angle_pixel(points)
        gamma = correct_roll(beta, roll_deg)
        if fusion_mode == "imu_primary":
            # 비전 값은 참고만 하고, 각도 신뢰도를 낮춰 표시 (예: 절반 가중)
            gamma = gamma * 0.5

    risk_state = classifier.update(
        sag_ratio if sag_ratio is not None else 0.0,
        gamma if gamma is not None else 0.0,
        roll_rate_deg_s,
        fusion_mode,
    )

    result = {
        "timestamp": timestamp,
        "fusion_mode": fusion_mode,
        "confidence": round(confidence, 3),
        "sag_ratio": round(sag_ratio, 4) if sag_ratio is not None else None,
        "towline_angle_pixel_deg": round(beta, 2) if beta is not None else None,
        "towline_angle_corrected_deg": round(gamma, 2) if gamma is not None else None,
        "roll_deg": round(roll_deg, 2),
        "roll_rate_deg_s": round(roll_rate_deg_s, 2),
        "risk_state": risk_state,
    }
    return result


# ---------------------------------------------------------------------------
# 8. 더미 시나리오로 전체 파이프라인 끊김 없이 도는지 확인
# ---------------------------------------------------------------------------
def run_dummy_scenario():
    classifier = RiskStateClassifier(window=5)

    # 시간에 따라 줄이 점점 팽팽해지고(sag 감소) 각도가 커지는(거딩 진행) 시나리오
    n_frames = 12
    sag_values = np.linspace(0.25, 0.02, n_frames)      # 점점 팽팽해짐
    angle_values = np.linspace(5, 45, n_frames)          # 점점 횡방향으로 꺾임
    roll_values = np.linspace(0, 12, n_frames)           # 배가 점점 기움
    confidences = [0.9] * 8 + [0.5, 0.5, 0.2, 0.2]       # 후반부에 시야 나빠짐(안개 가정)

    print(f"{'t':>3} | {'mode':<16} | {'conf':>5} | {'sag':>6} | "
          f"{'beta':>7} | {'gamma':>7} | {'roll':>5} | risk")
    print("-" * 80)

    for i in range(n_frames):
        mask = generate_dummy_mask(sag_values[i], angle_values[i])
        roll_rate = roll_values[i] - (roll_values[i - 1] if i > 0 else 0)

        result = process_frame(
            mask=mask,
            confidence=confidences[i],
            roll_deg=roll_values[i],
            roll_rate_deg_s=roll_rate,
            classifier=classifier,
            timestamp=i * 0.5,
        )

        print(f"{i:>3} | {result['fusion_mode']:<16} | "
              f"{result['confidence']:>5} | "
              f"{str(result['sag_ratio']):>6} | "
              f"{str(result['towline_angle_pixel_deg']):>7} | "
              f"{str(result['towline_angle_corrected_deg']):>7} | "
              f"{result['roll_deg']:>5} | {result['risk_state']}")

    # 마지막 프레임 결과를 JSON 형태로 출력 (UI/파이프라인으로 넘길 실제 형태)
    print("\n마지막 프레임 JSON 출력 예시:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run_dummy_scenario()