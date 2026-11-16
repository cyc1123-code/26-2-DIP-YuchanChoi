import streamlit as st
import cv2
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import os

# 1. 페이지 레이아웃 설정 (와이드 모드)
st.set_page_config(layout="wide", page_title="Object Counting & Filtering System")

st.title("📦 Filtering 시스템")
st.write("사이드바의 슬라이더를 조절하며 배경 노이즈를 잡고 물체들을 검출해 보세요.")

# =====================================================================
# 🎛️ 사이드바 영역: 이미지 선택 및 필터링 파라미터 조절
# =====================================================================
st.sidebar.header("📁 이미지 및 필터 설정")

# 현재 .py 파일이 위치한 실제 폴더 경로를 기준으로 설정 (배포 환경 경로 문제 해결)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1) 현재 .py 파일이 있는 폴더 안의 이미지(.jpg, .jpeg, .png) 자동 검색
image_extensions = (".jpg", ".jpeg", ".png")
image_files = [f for f in os.listdir(current_dir) if f.lower().endswith(image_extensions)]

if image_files:
    selected_image_name = st.sidebar.selectbox("📂 이미지 선택하기", image_files)
    # 파일 절대 경로 지정
    target_image_path = os.path.join(current_dir, selected_image_name)
else:
    st.sidebar.warning("현재 폴더에 이미지 파일이 없습니다. 사진들을 코드와 같은 폴더에 넣어주세요!")
    target_image_path = None

# 2) 필터링 파라미터 슬라이더
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 필터링 파라미터 조절")

# 가우시안 스무딩 강도 (Sigma)
sigma_val = st.sidebar.slider("Gaussian Sigma (스무딩 강도)", 0.1, 5.0, 1.5, 0.1)

# 수동 임계값 슬라이더
threshold_val = st.sidebar.slider("Threshold (이진화 임계값)", 0, 255, 140)

# 최소 객체 크기 슬라이더 (컨투어 필터링용)
min_area = st.sidebar.slider("Minimum Contour Area (최소 객체 크기)", 1, 3000, 30)

# =====================================================================
# 🖼️ 이미지 로드 및 처리 파이프라인 함수
# =====================================================================
def image_filtering_pipeline(image_path, sigma=1.5, threshold_val=140):
    """
    디지털 이미지 처리 객체 카운팅을 위한 단일 필터링 파이프라인
    """
    # 1. 이미지 로드 및 그레이스케일 변환
    img = Image.open(image_path).convert('L')
    f = np.array(img, dtype=np.float32)
    
    # 2. 필터링 단계 (가우시안 블러를 통한 노이즈 제거 및 평탄화)
    smoothed = nd.gaussian_filter(f, sigma=sigma)
    
    # 3. 이진화(Thresholding) 및 객체 분할 준비
    binary = (smoothed > threshold_val).astype(np.uint8) * 255
    
    # 4. 형태학적 연산(Morphological cleanup)을 통한 작은 노이즈 제거
    binary_cleaned = nd.binary_opening(binary, structure=np.ones((3,3))).astype(np.uint8) * 255

    return f, smoothed, binary_cleaned

# =====================================================================
# 🚀 메인 실행 로직
# =====================================================================
if target_image_path is not None:
    # 파이프라인 호출 (사이드바 슬라이더 값 연동)
    orig_img, processed_img, result_binary = image_filtering_pipeline(
        target_image_path, 
        sigma=sigma_val, 
        threshold_val=threshold_val
    )
    
    # 📍 OpenCV 컨투어(Contour) 분석을 통한 객체 카운팅 수행
    contours, _ = cv2.findContours(result_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 지정한 최소 크기(min_area) 이상의 객체만 필터링하여 카운트
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    num_objects = len(valid_contours)
    
    # 객체 시각화를 위해 원본 이미지 로드 후 컨투어 드로잉
    color_img_cv = cv2.cvtColor(cv2.imread(target_image_path), cv2.COLOR_BGR2RGB)
    cv2.drawContours(color_img_cv, valid_contours, -1, (0, 255, 0), 2)

    # -----------------------------------------------------------------
    # 🖥️ 화면 3분할 출력 레이아웃
    # -----------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📥 원본 이미지 (Grayscale)")
        st.image(orig_img.astype(np.uint8), use_column_width=True, channels="GRAY")
        
    with col2:
        st.subheader("🔍 필터링된 창 (Smoothing)")
        st.image(processed_img.astype(np.uint8), use_column_width=True, channels="GRAY")
        
    with col3:
        st.subheader("📤 최종 타겟 구분 및 카운팅")
        st.image(color_img_cv, use_column_width=True)
        
    st.markdown("---")
    st.success(f"🎯 **Counting Result**: 현재 선택된 파일(**{selected_image_name}**)에서 검출된 객체(나뭇잎) 개수는 총 **{num_objects}개**입니다.")
    
else:
    st.info("👈 사이드바에 표시될 이미지 파일들을 코드 파일과 같은 폴더에 넣어주세요!")
