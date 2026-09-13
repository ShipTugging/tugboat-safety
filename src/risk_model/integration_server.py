"""
TUGGUARD - 통합 파이프라인 서버 (강영한 담당)

역할: Three.js 시뮬레이터가 보낸 이미지+IMU 데이터를 받아서,
     (지금은 더미 mask로 대체된) YOLO-Seg 결과와 risk_pipeline.py를
     거쳐 위험판단 JSON을 만들고, 다시 Three.js로 돌려준다.

[Three.js] --POST(image, imu)--> [이 서버] --JSON--> [Three.js UI]

지금 단계 목표: 구조가 끊김 없이 도는지 검증 (더미 mask 사용).
나중에 할 일: run_yolo_seg() 자리에 제원이의 실제 YOLO-Seg 추론 코드 연결.
"""

import base64
import io
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

from risk_pipeline import (
    generate_dummy_mask,
    process_frame,
    RiskStateClassifier,
)

app = FastAPI(title="TUGGUARD Integration Server")

# Three.js(브라우저)에서 다른 포트로 요청이 오니까 CORS 허용 필요
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 데모 단계라 전체 허용, 배포 시엔 좁혀야 함
    allow_methods=["*"],
    allow_headers=["*"],
)

# 위험 상태는 "최근 프레임 이력"을 기억해야 하므로, 서버가 켜져있는 동안
# 하나의 시나리오만 돈다고 가정하고 전역으로 하나만 둔다.
# (여러 클라이언트/시뮬레이션을 동시에 돌리려면 session_id별로 나눠야 함 - 지금은 범위 밖)
classifier = RiskStateClassifier(window=5)
_start_time = time.time()


class FrameRequest(BaseModel):
    """Three.js가 매 프레임 보내는 데이터 형태."""
    image_base64: str | None = None   # data:image/jpeg;base64,... (지금은 안 씀, 더미 mask 사용)
    roll_deg: float = 0.0
    roll_rate_deg_s: float = 0.0
    confidence: float = 0.9           # 실제로는 YOLO가 뱉어주는 값 (지금은 클라이언트가 임의로 넣어 테스트)
    # 더미 mask 생성용 (실전에서는 YOLO가 실제 이미지로 계산하므로 불필요해짐)
    sag_ratio_hint: float = 0.15
    angle_hint_deg: float = 30.0


def run_yolo_seg_dummy(sag_ratio_hint: float, angle_hint_deg: float) -> np.ndarray:
    """
    지금은 제원이의 실제 YOLO-Seg 모델이 없으므로, 힌트값으로 그럴듯한
    mask를 만들어서 파이프라인 구조만 검증한다.

    TODO(제원 모델 연동 시): 이 함수를 아래처럼 교체
        def run_yolo_seg(image: np.ndarray) -> tuple[np.ndarray, float]:
            results = yolo_model.predict(image)
            mask = results[0].masks.data[0].cpu().numpy()
            confidence = float(results[0].boxes.conf[0])
            return mask, confidence
    """
    return generate_dummy_mask(sag_ratio_hint, angle_hint_deg)


@app.get("/health")
def health_check():
    """서버가 살아있는지 확인하는 용도. Three.js에서 연결 전 핑 날려보기 좋음."""
    return {"status": "ok", "uptime_s": round(time.time() - _start_time, 1)}


@app.post("/analyze")
def analyze_frame(req: FrameRequest):
    """
    Three.js가 프레임마다 호출할 메인 엔드포인트.
    지금은 image_base64를 실제로 디코딩해서 YOLO에 넣지 않고,
    sag_ratio_hint/angle_hint_deg로 더미 mask만 만들어서 구조를 검증한다.
    """
    mask = run_yolo_seg_dummy(req.sag_ratio_hint, req.angle_hint_deg)
    confidence = req.confidence

    result = process_frame(
        mask=mask,
        confidence=confidence,
        roll_deg=req.roll_deg,
        roll_rate_deg_s=req.roll_rate_deg_s,
        classifier=classifier,
        timestamp=round(time.time() - _start_time, 3),
    )
    return result


@app.post("/reset")
def reset_classifier():
    """새 시나리오를 처음부터 돌릴 때, 위험 상태 이력을 초기화."""
    global classifier
    classifier = RiskStateClassifier(window=5)
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)