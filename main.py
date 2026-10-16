left, right = st.columns(2)
with left:
    x_feature = st.selectbox("가로축 속성", FEATURES, index=0, key="x_feature")
with right:
    y_default = 1 if len(FEATURES) > 1 else 0
    y_feature = st.selectbox("세로축 속성", FEATURES, index=y_default, key="y_feature")

# 산점도는 사용자가 고른 속성의 실제 해석이 가능하도록 로그 전 값/원래 값으로 표시
plot_df = raw_df.copy()
plot_df["스크린 수"] = plot_df["first_scrn"]
plot_df["누적 관객"] = plot_df["total_audi"]
plot_df["10위권 일수"] = plot_df["days_in_top10"]
plot_df["롱런 지수"] = plot_df["long_run_index"]

fig2d = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    hover_data={
        "유형": True,
        "first_scrn": False,
        "total_audi": False,
        "days_in_top10": False,
        "long_run_index": False,
    },
    category_orders={"유형": CLUSTER_NAMES},
    labels={
        "스크린 수": "스크린 수(개)",
        "누적 관객": "누적 관객(명)",
        "10위권 일수": "10위권 일수(일)",
        "롱런 지수": "롱런 지수",
        "유형": "영화 유형",
    },
    title="선택한 속성으로 살펴보는 영화 유형"
)
fig2d.update_traces(marker={"size": 8, "opacity": 0.78})
fig2d.update_layout(legend_title_text="영화 유형", height=560)
st.plotly_chart(fig2d, use_container_width=True)

# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.divider()
st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 위에서 유형을 나눌 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        z_x = st.selectbox("3D x축", FEATURES, index=0, key="z_x")
    with c2:
        z_y = st.selectbox("3D y축", FEATURES, index=1, key="z_y")
    with c3:
        z_z = st.selectbox("3D z축", FEATURES, index=2, key="z_z")

    fig3d = px.scatter_3d(
        plot_df,
        x=z_x,
        y=z_y,
        z=z_z,
        color="유형",
        hover_name="movieNm",
        category_orders={"유형": CLUSTER_NAMES},
        labels={
            "스크린 수": "스크린 수(개)",
            "누적 관객": "누적 관객(명)",
            "10위권 일수": "10위권 일수(일)",
            "롱런 지수": "롱런 지수",
            "유형": "영화 유형",
        },
        title="마우스로 회전하며 살펴보는 3차원 영화 유형"
    )
    fig3d.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3d.update_layout(
        height=680,
        legend_title_text="영화 유형",
        scene={
            "xaxis_title": z_x,
            "yaxis_title": z_y,
            "zaxis_title": z_z,
        }
    )
    st.plotly_chart(fig3d, use_container_width=True)

# --------------------------------------------------
# 유형별 요약 통계 및 대표 영화
# --------------------------------------------------
st.divider()
st.subheader("4. 유형별 요약")
summary_rows = []
for group_name in CLUSTER_NAMES:
    group = raw_df[raw_df["유형"] == group_name]
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
for col in ["스크린 수 평균(개)", "누적 관객 평균(명)", "10위권 일수 평균(일)", "롱런 지수 평균"]:
    if col == "롱런 지수 평균":
        summary_df[col] = summary_df[col].round(2)
    else:
        summary_df[col] = summary_df[col].round(0).astype("Int64")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.subheader("5. 유형별 누적 관객 상위 영화 5편")
for group_name in CLUSTER_NAMES:
    group = raw_df[raw_df["유형"] == group_name].sort_values("total_audi", ascending=False).head(5)
    if group.empty:
        continue
    with st.expander(f"{group_name} 유형 · {len(raw_df[raw_df['유형'] == group_name]):,}편", expanded=True):
        top_five = group[["movieNm", "total_audi"]].rename(
            columns={"movieNm": "영화 제목", "total_audi": "누적 관객(명)"}
        ).copy()
        top_five["누적 관객(명)"] = top_five["누적 관객(명)"].map(
            lambda x: f"{int(x):,}" if pd.notna(x) else "-"
        )
        st.dataframe(top_five, use_container_width=True, hide_index=True)

st.caption("군집 번호는 누적 관객 평균이 높은 유형부터 ㉮·㉯·㉰로 표시합니다. 산점도 축은 원래 단위로 표시하고, 군집화 과정에서는 선택한 속성을 표준화해 사용합니다.")
