kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
movies["군집번호"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 높은 군집부터 ㉮, ㉯, ㉰ 지정
cluster_order = (
    movies.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
name_map = {
    cluster_id: CLUSTER_NAMES[i]
    for i, cluster_id in enumerate(cluster_order)
}
movies["유형"] = movies["군집번호"].map(name_map)

st.markdown(
    f"**전체 편수:** {total_movie_count:,}편　 |　 "
    f"**묶은 편수:** {len(movies):,}편"
)

# 그래프는 원래 단위로 표시
plot_df = movies.copy()
plot_df["스크린 수"] = plot_df["first_scrn"]
plot_df["누적 관객"] = plot_df["total_audi"]
plot_df["10위권 일수"] = plot_df["days_in_top10"]
plot_df["롱런 지수"] = plot_df["long_run_index"]

st.divider()
st.subheader("2. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("가로축 속성", FEATURES, index=0, key="x_feature")
with col2:
    y_feature = st.selectbox("세로축 속성", FEATURES, index=1, key="y_feature")

fig2d = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    category_orders={"유형": CLUSTER_NAMES},
    labels={**LABELS, "유형": "영화 유형"},
    title="속성별 영화 유형 분포",
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.78})
fig2d.update_layout(height=560, legend_title_text="영화 유형")
st.plotly_chart(fig2d, use_container_width=True)

st.divider()
st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 군집화 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("3차원 x축", FEATURES, index=0, key="x3")
    with c2:
        y3 = st.selectbox("3차원 y축", FEATURES, index=1, key="y3")
    with c3:
        z3 = st.selectbox("3차원 z축", FEATURES, index=2, key="z3")

    fig3d = px.scatter_3d(
        plot_df,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        hover_name="movieNm",
        category_orders={"유형": CLUSTER_NAMES},
        labels={**LABELS, "유형": "영화 유형"},
        title="마우스로 회전하며 살펴보는 3차원 영화 유형",
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(
        height=680,
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": LABELS[x3],
            "yaxis_title": LABELS[y3],
            "zaxis_title": LABELS[z3],
        },
    )
    st.plotly_chart(fig3d, use_container_width=True)

st.divider()
st.subheader("4. 유형별 편수와 네 속성의 평균")

summary_rows = []
for group_name in CLUSTER_NAMES:
    group = movies[movies["유형"] == group_name]
    if group.empty:
        continue
    summary_rows.append({
        "유형": group_name,
        "편수": len(group),
        "스크린 수 평균(개)": group["first_scrn"].mean(),
        "누적 관객 평균(명)": group["total_audi"].mean(),
        "10위권 일수 평균(일)": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["long_run_index"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df["스크린 수 평균(개)"] = summary_df["스크린 수 평균(개)"].round(1)
summary_df["누적 관객 평균(명)"] = summary_df["누적 관객 평균(명)"].round(0)
summary_df["10위권 일수 평균(일)"] = summary_df["10위권 일수 평균(일)"].round(1)
summary_df["롱런 지수 평균"] = summary_df["롱런 지수 평균"].round(2)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for group_name in CLUSTER_NAMES:
    group = (
        movies[movies["유형"] == group_name]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )
    if group.empty:
        continue

    count_in_group = len(movies[movies["유형"] == group_name])
    with st.expander(f"{group_name} 유형 · {count_in_group:,}편", expanded=True):
        top_five = group[["movieNm", "total_audi"]].copy()
        top_five.columns = ["영화 제목", "누적 관객(명)"]
        top_five["누적 관객(명)"] = top_five["누적 관객(명)"].map(
            lambda value: f"{int(value):,}"
        )
