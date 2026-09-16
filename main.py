import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 앱 설정
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬", layout="wide")

st.title("🎬 영화 유형 나누기")
st.markdown("영화 데이터를 분석하여 비슷한 특성을 가진 영화들을 묶어봅니다.")

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
    df_filtered = df_filtered[df_filtered['first_week_audi'] > 0]
    
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
    
    selected_features = [features_mapping[label] for label in selected_features_labels]

    if len(selected_features) < 2:
        st.warning("분석을 위해 최소 2개 이상의 속성을 선택해주세요.")
    else:
        # 데이터 표준화
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_filtered[selected_features])
        
        # K-평균 클러스터링 (난수 고정)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_filtered['cluster'] = kmeans.fit_predict(scaled_data)
        
        # 클러스터 이름 매핑 (누적 관객 평균순으로 ㉮, ㉯, ㉰)
        cluster_audi_mean = df_filtered.groupby('cluster')['total_audi'].mean().sort_values(ascending=False)
        cluster_names = {cluster_audi_mean.index[0]: '㉮', cluster_audi_mean.index[1]: '㉯', cluster_audi_mean.index[2]: '㉰'}
        df_filtered['cluster_name'] = df_filtered['cluster'].map(cluster_names)
        
        # 시각화를 위해 클러스터 이름을 정렬된 순서(㉮, ㉯, ㉰)로 범주형 변수로 변환
        df_filtered['cluster_name'] = pd.Categorical(df_filtered['cluster_name'], categories=['㉮', '㉯', '㉰'], ordered=True)

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
            st.info("3차원 산점도를 그리려면 3개 이상의 속성을 선택해야 합니다.")
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
        
        # 요약 통계 계산을 위해 그룹화
        summary_df = df_filtered.groupby('cluster_name', observed=False).agg(
            편수=('movieCd', 'count'),
            스크린수_평균=('first_scrn', 'mean'),
            누적관객_평균=('total_audi', 'mean'),
            탑10일수_평균=('days_in_top10', 'mean'),
            롱런지수_평균=('long_run_index', 'mean')
        ).reset_index()
        
        # 열 이름 변경
        summary_df = summary_df.rename(columns={'cluster_name': '묶음'})
        
        # 숫자 형식 지정
        summary_df['스크린수_평균'] = summary_df['스크린수_평균'].round(1)
        summary_df['누적관객_평균'] = summary_df['누적관객_평균'].round(1)
        summary_df['탑10일수_평균'] = summary_df['탑10일수_평균'].round(1)
        summary_df['롱런지수_평균'] = summary_df['롱런지수_평균'].round(2)
        
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
        
        st.header("4. 묶음별 누적 관객 상위 5편")
        cols = st.columns(3)
        
        for i, cluster in enumerate(['㉮', '㉯', '㉰']):
            with cols[i]:
                st.subheader(f"묶음 {cluster}")
                top_movies = df_filtered[df_filtered['cluster_name'] == cluster].sort_values(by='total_audi', ascending=False).head(5)
                for _, row in top_movies.iterrows():
                    st.write(f"- {row['movieNm']} ({row['total_audi']:,}명)")
