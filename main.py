import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 앱 설정
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")
st.markdown("영화 데이터를 분석하여 비슷한 특성을 가진 영화들을 묶어보고, 최적의 묶음 수를 탐색해 봅니다.")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None

df = load_data()

if df is not None:
    # 결측치 처리 및 0 나누기 방지
    original_count = len(df)
    
    # 사용할 4가지 속성이 존재하고, 첫 주 관객수가 0이 아닌 데이터만 필터링
    df_filtered = df.dropna(subset=['first_scrn', 'total_audi', 'days_in_top10', 'first_week_audi'])
    df_filtered = df_filtered[df_filtered['first_week_audi'] > 0].copy()
    
    # 롱런 지수 계산 및 20 초과 값 자르기
    df_filtered['long_run_index'] = df_filtered['total_audi'] / df_filtered['first_week_audi']
    df_filtered['long_run_index'] = df_filtered['long_run_index'].clip(upper=20)
    
    # 스크린 수와 누적 관객에 상용로그 취하기
    df_filtered['log_first_scrn'] = np.log10(df_filtered['first_scrn'])
    df_filtered['log_total_audi'] = np.log10(df_filtered['total_audi'])
    
    used_count = len(df_filtered)
    st.info(f"전체 영화 편수: {original_count}편 / 분석에 사용된 영화 편수: {used_count}편")

    st.sidebar.header("분석 설정")
    
    # 클러스터링에 사용할 속성 매핑
    features_mapping = {
        '스크린 수 (로그)': 'log_first_scrn',
        '누적 관객 (로그)': 'log_total_audi',
        '10위권 일수': 'days_in_top10',
        '롱런 지수': 'long_run_index'
    }
    
    selected_features_labels = st.sidebar.multiselect(
        "클러스터링에 사용할 속성을 고르세요 (2개 이상):",
        options=list(features_mapping.keys()),
        default=list(features_mapping.keys())
    )
    
    num_clusters = st.sidebar.slider(
        "묶음 수 (2~7):", 
        min_value=2, 
        max_value=7, 
        value=3,
        step=1
    )
    
    selected_features = [features_mapping[label] for label in selected_features_labels]

    if len(selected_features) < 2:
        st.warning("분석을 위해 최소 2개 이상의 속성을 선택해주세요.")
    else:
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered[selected_features])
        
        # 선택된 묶음 수로 K-평균 클러스터링 진행 (난수 고정)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df_filtered['cluster'] = kmeans.fit_predict(scaled_data)
        
        # 클러스터 이름 매핑 (누적 관객 평균순으로 ㉮~㉴ 부여)
        cluster_audi_mean = df_filtered.groupby('cluster')['total_audi'].mean().sort_values(ascending=False)
        cluster_symbols = ['㉮', '㉯', '㉰', '㉱', '㉲', '㉳', '㉴']
        
        cluster_names = {}
        ordered_categories = []
        for i, original_cluster_idx in enumerate(cluster_audi_mean.index):
            symbol = cluster_symbols[i]
            cluster_names[original_cluster_idx] = symbol
            ordered_categories.append(symbol)
            
        df_filtered['cluster_name'] = df_filtered['cluster'].map(cluster_names)
        
        # 시각화를 위해 클러스터 이름을 정렬된 범주형 변수로 변환
        df_filtered['cluster_name'] = pd.Categorical(df_filtered['cluster_name'], categories=ordered_categories, ordered=True)

        st.header("1. 2차원 산점도")
        col1, col2 = st.columns(2)
        with col1:
            x_axis_2d = st.selectbox("X축 속성:", options=selected_features_labels, index=0, key='2d_x')
        with col2:
            y_axis_2d = st.selectbox("Y축 속성:", options=selected_features_labels, index=1 if len(selected_features_labels) > 1 else 0, key='2d_y')
        
        fig_2d = px.scatter(
            df_filtered,
            x=features_mapping[x_axis_2d],
            y=features_mapping[y_axis_2d],
            color='cluster_name',
            hover_name='movieNm',
            title=f"{x_axis_2d} vs {y_axis_2d}",
            color_discrete_sequence=px.colors.qualitative.Set1,
            labels={features_mapping[x_axis_2d]: x_axis_2d, features_mapping[y_axis_2d]: y_axis_2d, 'cluster_name': '묶음'}
        )
        st.plotly_chart(fig_2d, use_container_width=True)

        st.header("2. 3차원 산점도")
        if len(selected_features) < 3:
            st.info("💡 3차원 산점도를 그리려면 사이드바에서 3개 이상의 속성을 선택해야 합니다.")
        else:
            col3, col4, col5 = st.columns(3)
            with col3:
                x_axis_3d = st.selectbox("X축 속성:", options=selected_features_labels, index=0, key='3d_x')
            with col4:
                y_axis_3d = st.selectbox("Y축 속성:", options=selected_features_labels, index=1, key='3d_y')
            with col5:
                z_axis_3d = st.selectbox("Z축 속성:", options=selected_features_labels, index=2, key='3d_z')
            
            fig_3d = px.scatter_3d(
                df_filtered,
                x=features_mapping[x_axis_3d],
                y=features_mapping[y_axis_3d],
                z=features_mapping[z_axis_3d],
                color='cluster_name',
                hover_name='movieNm',
                title=f"3D: {x_axis_3d}, {y_axis_3d}, {z_axis_3d}",
                color_discrete_sequence=px.colors.qualitative.Set1,
                labels={
                    features_mapping[x_axis_3d]: x_axis_3d, 
                    features_mapping[y_axis_3d]: y_axis_3d, 
                    features_mapping[z_axis_3d]: z_axis_3d, 
                    'cluster_name': '묶음'
                }
            )
            # 점 크기 작게 설정
            fig_3d.update_traces(marker=dict(size=3))
            st.plotly_chart(fig_3d, use_container_width=True)

        st.header("3. 묶음 요약")
        summary_df = df_filtered.groupby('cluster_name', observed=False).agg(
            편수=('movieCd', 'count'),
            스크린수_평균=('first_scrn', 'mean'),
            누적관객_평균=('total_audi', 'mean'),
            탑10일수_평균=('days_in_top10', 'mean'),
            롱런지수_평균=('long_run_index', 'mean')
        ).reset_index()
        
        summary_df = summary_df.rename(columns={'cluster_name': '묶음'})
        
        # 숫자 형식 지정
        summary_df['스크린수_평균'] = summary_df['스크린수_평균'].round(1)
        summary_df['누적관객_평균'] = summary_df['누적관객_평균'].round(1)
        summary_df['탑10일수_평균'] = summary_df['탑10일수_평균'].round(1)
        summary_df['롱런지수_평균'] = summary_df['롱런지수_평균'].round(2)
        
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        st.header(f"4. 묶음별 누적 관객 상위 5편")
        
        # 선택된 클러스터 수만큼 컬럼 생성
        cols = st.columns(num_clusters)
        
        for i, cluster_sym in enumerate(ordered_categories):
            with cols[i]:
                st.subheader(f"묶음 {cluster_sym}")
                top_movies = df_filtered[df_filtered['cluster_name'] == cluster_sym].sort_values(by='total_audi', ascending=False).head(5)
                for _, row in top_movies.iterrows():
                    st.write(f"- {row['movieNm']}\n  ({row['total_audi']:,}명)")

        st.divider()

        st.header("5. 최적의 묶음 수 찾기 (현재 선택 속성 기준)")
        
        # 실루엣 점수 계산 및 표시
        sil_score = silhouette_score(scaled_data, df_filtered['cluster'])
        st.info(f"✨ **현재 묶음 수({num_clusters}개)의 실루엣 점수:** `{sil_score:.4f}` (점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷하다는 뜻입니다.)")

        # 1부터 7까지의 Inertia 계산
        inertias = []
        k_range = range(1, 8)
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled_data)
            inertias.append(km.inertia_)
        
        # 꺾은선 그래프 생성
        fig_elbow = px.line(
            x=list(k_range), 
            y=inertias, 
            markers=True,
            title="묶음 수에 따른 거리 제곱합 (엘보우 방법)",
            labels={'x': '묶음 수', 'y': '거리의 제곱합 (Inertia)'}
        )
        
        # 현재 선택한 묶음 수에 세로선 추가
        fig_elbow.add_vline(
            x=num_clusters, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"현재 선택 ({num_clusters}개) "
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)

        # 표 생성
        elbow_df = pd.DataFrame({
            '묶음 수': list(k_range),
            '거리 제곱합': inertias
        })
        
        # 바로 앞 값과 비교하여 줄어든 양 계산
        elbow_df['전 단계 대비 감소량'] = elbow_df['거리 제곱합'].shift(1) - elbow_df['거리 제곱합']
        
        # 보기 좋게 포맷팅 (첫 줄 빈칸 처리)
        display_df = elbow_df.copy()
        display_df['거리 제곱합'] = display_df['거리 제곱합'].apply(lambda x: f"{x:,.2f}")
        display_df['전 단계 대비 감소량'] = display_df['전 단계 대비 감소량'].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
        
        st.write("📊 **묶음 수별 거리 제곱합 및 감소량 표**")
        st.dataframe(display_df, hide_index=True, use_container_width=True)
