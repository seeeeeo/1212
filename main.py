import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as px_go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 페이지 설정 (앱 제목, 아이콘, 브라우저 탭 제목)
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

# 화면 상단 제목 및 아이콘
st.title("🎬 영화 유형 나누기")
st.write("KOBIS 영화 데이터를 활용하여 K-Means 알고리즘으로 영화 유형을 군집화합니다.")

# 2. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    url = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    total_raw_count = len(df)
    
    # 첫 주 관객 수가 0 이하이거나 결측인 영화 제외
    valid_df = df.copy()
    valid_df = valid_df[valid_df['first_week_audi'] > 0]
    
    # 롱런 지수 생성: 누적 관객 / 첫 주 관객 (20 초과 시 20으로 큼)
    valid_df['long_run_index'] = valid_df['total_audi'] / valid_df['first_week_audi']
    valid_df['long_run_index'] = valid_df['long_run_index'].clip(upper=20)
    
    # 상용로그 취하기 (log10)
    valid_df['log_first_scrn'] = np.log10(valid_df['first_scrn'])
    valid_df['log_total_audi'] = np.log10(valid_df['total_audi'])
    
    # 필수 4개 속성 결측치 제거
    required_cols = ['log_first_scrn', 'log_total_audi', 'days_in_top10', 'long_run_index']
    valid_df = valid_df.dropna(subset=required_cols)
    
    used_count = len(valid_df)
    return df, valid_df, total_raw_count, used_count

try:
    raw_df, df, total_raw_count, used_count = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 전체 편수와 묶은 편수를 화면에 한 줄로 작성
st.markdown(f"**전체 영화 편수:** {total_raw_count:,}편 | **분석에 사용된 묶은 영화 편수:** {used_count:,}편")
st.divider()

# 3. 속성 및 묶음 수 선택
feature_options = {
    '스크린 수 (상용로그)': 'log_first_scrn',
    '누적 관객 (상용로그)': 'log_total_audi',
    '10위권 일수': 'days_in_top10',
    '롱런 지수': 'long_run_index'
}

col_sel1, col_sel2 = st.columns([2, 1])

with col_sel1:
    st.subheader("1. 묶는 데 사용할 속성 선택")
    selected_feature_labels = st.multiselect(
        "군집화에 사용할 속성을 2개 이상 선택하세요 (기본값: 4개 전체):",
        options=list(feature_options.keys()),
        default=list(feature_options.keys())
    )

with col_sel2:
    st.subheader("2. 묶음 수(K) 선택")
    n_clusters = st.slider(
        "묶음 수(K)를 선택하세요:",
        min_value=2,
        max_value=7,
        value=3,
        step=1
    )

if len(selected_feature_labels) < 2:
    st.warning("속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_options[label] for label in selected_feature_labels]

# 4. 표준화 및 K-Means 군집화
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster_raw'] = cluster_labels

# 기호 정의 (최대 7개까지 확장: ㉮, ㉯, ㉰, ㉱, ㉲, ㉳, ㉴)
all_labels_kor = ['㉮', '㉯', '㉰', '㉱', '㉲', '㉳', '㉴']
current_labels_kor = all_labels_kor[:n_clusters]

# 누적 관객(total_audi) 평균이 큰 묶음부터 순서대로 기호 부여
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
cluster_mapping = {}
for rank, cluster_id in enumerate(cluster_audi_mean.index):
    cluster_mapping[cluster_id] = current_labels_kor[rank]

df['묶음'] = df['cluster_raw'].map(cluster_mapping)

st.divider()

# 5. 2차원 산점도
st.subheader("3. 2차원 산점도")
col2
