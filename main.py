import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

# 3. 속성 선택
feature_options = {
    '스크린 수 (상용로그)': 'log_first_scrn',
    '누적 관객 (상용로그)': 'log_total_audi',
    '10위권 일수': 'days_in_top10',
    '롱런 지수': 'long_run_index'
}

st.subheader("1. 묶는 데 사용할 속성 선택")
selected_feature_labels = st.multiselect(
    "군집화에 사용할 속성을 2개 이상 선택하세요 (기본값: 4개 전체):",
    options=list(feature_options.keys()),
    default=list(feature_options.keys())
)

if len(selected_feature_labels) < 2:
    st.warning("속성을 최소 2개 이상 선택해 주세요.")
    st.stop()

selected_features = [feature_options[label] for label in selected_feature_labels]

# 4. 표준화 및 K-Means 군집화 (세 묶음, 난수 고정 random_state=42)
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df['cluster_raw'] = cluster_labels

# 누적 관객(total_audi) 평균이 큰 묶음부터 ㉮, ㉯, ㉰ 부여
cluster_audi_mean = df.groupby('cluster_raw')['total_audi'].mean().sort_values(ascending=False)
cluster_mapping = {}
labels_kor = ['㉮', '㉯', '㉰']
for rank, cluster_id in enumerate(cluster_audi_mean.index):
    cluster_mapping[cluster_id] = labels_kor[rank]

df['묶음'] = df['cluster_raw'].map(cluster_mapping)

st.divider()

# 5. 2차원 산점도
st.subheader("2. 2차원 산점도")
col2d_1, col2d_2 = st.columns(2)
with col2d_1:
    x_axis_2d_label = st.selectbox("가로축 (X축) 속성:", options=selected_feature_labels, index=0, key='2d_x')
with col2d_2:
    y_axis_2d_label = st.selectbox("세로축 (Y축) 속성:", options=selected_feature_labels, index=min(1, len(selected_feature_labels)-1), key='2d_y')

fig_2d = px.scatter(
    df,
    x=feature_options[x_axis_2d_label],
    y=feature_options[y_axis_2d_label],
    color='묶음',
    hover_name='movieNm',
    category_orders={'묶음': ['㉮', '㉯', '㉰']},
    labels={
        feature_options[x_axis_2d_label]: x_axis_2d_label,
        feature_options[y_axis_2d_label]: y_axis_2d_label
    },
    title=f"2차원 산점도 ({x_axis_2d_label} vs {y_axis_2d_label})"
)
st.plotly_chart(fig_2d, use_container_width=True)

st.divider()

# 6. 3차원 산점도
st.subheader("3. 3차원 산점도")
if len(selected_feature_labels) < 3:
    st.info("💡 3차원 산점도를 그리려면 위에서 속성을 3개 이상 선택해 주세요.")
else:
    col3d_1, col3d_2, col3d_3 = st.columns(3)
    with col3d_1:
        x_axis_3d_label = st.selectbox("X축 속성:", options=selected_feature_labels, index=0, key='3d_x')
    with col3d_2:
        y_axis_3d_label = st.selectbox("Y축 속성:", options=selected_feature_labels, index=1, key='3d_y')
    with col3d_3:
        z_axis_3d_label = st.selectbox("Z축 속성:", options=selected_feature_labels, index=2, key='3d_z')

    fig_3d = px.scatter_3d(
        df,
        x=feature_options[x_axis_3d_label],
        y=feature_options[y_axis_3d_label],
        z=feature_options[z_axis_3d_label],
        color='묶음',
        hover_name='movieNm',
        category_orders={'묶음': ['㉮', '㉯', '㉰']},
        labels={
            feature_options[x_axis_3d_label]: x_axis_3d_label,
            feature_options[y_axis_3d_label]: y_axis_3d_label,
            feature_options[z_axis_3d_label]: z_axis_3d_label
        },
        title=f"3차원 산점도 ({x_axis_3d_label} / {y_axis_3d_label} / {z_axis_3d_label})"
    )
    # 점 크기 작게 조절
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

st.divider()

# 7. 묶음별 편수 및 네 속성 평균 (원래 단위)
st.subheader("4. 묶음별 편수 및 속성 평균 (원래 단위)")

summary_df = df.groupby('묶음').agg(
    편수=('movieNm', 'count'),
    스크린_수_평균=('first_scrn', 'mean'),
    누적_관객_평균=('total_audi', 'mean'),
    십위권_일수_평균=('days_in_top10', 'mean'),
    롱런_지수_평균=('long_run_index', 'mean')
).reindex(['㉮', '㉯', '㉰'])

# 형식 정리
formatted_summary = summary_df.copy()
formatted_summary['편수'] = formatted_summary['편수'].apply(lambda x: f"{x:,}편")
formatted_summary['스크린_수_평균'] = formatted_summary['스크린_수_평균'].apply(lambda x: f"{x:,.1f}개")
formatted_summary['누적_관객_평균'] = formatted_summary['누적_관객_평균'].apply(lambda x: f"{x:,.1f}명")
formatted_summary['십위권_일수_평균'] = formatted_summary['십위권_일수_평균'].apply(lambda x: f"{x:,.1f}일")
formatted_summary['롱런_지수_평균'] = formatted_summary['롱런_지수_평균'].apply(lambda x: f"{x:,.2f}")

formatted_summary.columns = ['편수', '스크린 수 평균', '누적 관객 평균', '10위권 일수 평균', '롱런 지수 평균']
st.dataframe(formatted_summary, use_container_width=True)

st.divider()

# 8. 묶음별 누적 관객 수 상위 5개 영화
st.subheader("5. 묶음별 누적 관객 수가 많은 대표 영화 TOP 5")

cols_top5 = st.columns(3)
for idx, group_name in enumerate(['㉮', '㉯', '㉰']):
    with cols_top5[idx]:
        st.markdown(f"### 묶음 {group_name}")
        top5_movies = (
            df[df['묶음'] == group_name]
            .sort_values(by='total_audi', ascending=False)
            .head(5)
        )
        top5_list = []
        for i, (_, row) in enumerate(top5_movies.iterrows(), 1):
            top5_list.append(f"{i}. **{row['movieNm']}** ({row['total_audi']:,}명)")
        st.markdown("\n".join(top5_list))
