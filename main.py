import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. 기본 설정
# ==========================================
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "it-teacher-go/data-share-/main/kobis_movies.csv"
)

FEATURES = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]
CLUSTER_NAMES = ["㉮", "㉯", "㉰"]

LABELS = {
    "스크린 수": "스크린 수(개)",
    "누적 관객": "누적 관객(명)",
    "10위권 일수": "10위권 일수(일)",
    "롱런 지수": "롱런 지수",
}

st.title("🎬 영화 유형 나누기")
st.caption(
    "영화의 흥행 관련 속성을 바탕으로 "
    "K-평균 알고리즘을 이용해 비슷한 영화끼리 세 유형으로 나눕니다."
)

# ==========================================
# 2. 데이터 불러오기 및 전처리
# ==========================================
@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    original_count = len(df)

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV에 필요한 열이 없습니다: " + ", ".join(missing)
        )

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 첫 주 관객이 없거나 0인 영화 제외
    df = df[
        df["first_week_audi"].notna()
        & (df["first_week_audi"] > 0)
    ].copy()

    # 네 속성 중 결측치가 있는 영화 제외
    df = df.dropna(
        subset=[
            "first_scrn",
            "total_audi",
            "days_in_top10",
        ]
    )

    # 음수 값 제외
    df = df[
        (df["first_scrn"] >= 0)
        & (df["total_audi"] >= 0)
        & (df["days_in_top10"] >= 0)
    ].copy()

    # 롱런 지수 = 누적 관객 / 첫 주 관객
    df["long_run_index"] = (
        df["total_audi"] / df["first_week_audi"]
    ).clip(upper=20)

    # 군집화용 속성
    # 0도 처리할 수 있도록 log10(x + 1) 적용
    df["스크린 수"] = np.log10(df["first_scrn"] + 1)
    df["누적 관객"] = np.log10(df["total_audi"] + 1)
    df["10위권 일수"] = df["days_in_top10"]
    df["롱런 지수"] = df["long_run_index"]

    return df.reset_index(drop=True), original_count


try:
    movies, total_movie_count = load_data()
except Exception as e:
    st.error("영화 데이터를 불러오지 못했습니다.")
    st.write("데이터 주소와 CSV 파일을 확인해 주세요.")
    st.exception(e)
    st.stop()

usable_count = len(movies)

# 요청한 전체 편수와 군집화 가능한 편수 표시
st.markdown(
    f"**전체 편수:** {total_movie_count:,}편"
    f"　 |　 **묶을 수 있는 편수:** {usable_count:,}편"
)

if usable_count < 3:
    st.error("유형을 세 개로 나누려면 영화가 최소 3편 필요합니다.")
    st.stop()

# ==========================================
# 3. 군집화에 사용할 속성 선택
# ==========================================
st.subheader("1. 유형을 나눌 속성 선택")

selected_features = st.multiselect(
    "군집화에 사용할 속성을 두 개 이상 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    help=(
        "스크린 수와 누적 관객은 상용로그 변환값을 사용합니다. "
        "롱런 지수는 최대 20으로 제한합니다."
    ),
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

# ==========================================
# 4. 표준화 및 K-평균 군집화
# ==========================================
feature_matrix = movies[selected_features].astype(float)

scaler = StandardScaler()
scaled_values = scaler.fit_transform(feature_matrix)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

movies["군집번호"] = kmeans.fit_predict(scaled_values
