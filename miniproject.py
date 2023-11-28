
# 필요 패키지 추가
import time
import datetime
import pickle
import glob

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.font_manager as fm

#한글깨짐 방지코드 
font_location = '/home/sagemaker-user/gsc/NanumGothic.ttf'
fm.fontManager.addfont(font_location)
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)
matplotlib.rc('axes', unicode_minus=False)

pd.options.display.float_format = '{:,.0f}'.format

# 웹 페이지 기본 설정
# page title: 데이터 분석 및 모델링 대시보드
st.set_page_config(
    page_title="보스턴 주택 가격 데이터 분석 및 모델링 대시보드", # page 타이틀
    page_icon="🧊", # page 아이콘
    layout="wide", # wide, centered
    initial_sidebar_state="auto", # 사이드 바 초기 상태
    menu_items={
        'Get Help': 'https://streamlit.io',
        'Report a bug': None,
        'About': '2023 GS CDS Class',

    }
)

# 실습 소개 페이지 출력 함수
# 소개 페이지는 기본으로 제공됩니다.
def front_page():
    st.title('데이터 분석 및 모델링 실습')
    st.write('이 실습은 보스턴 주택 가격 데이터 분석, 학습 및 서빙을 대시보드를 생성하는 실습입니다.')
    st.markdown(' 1. EDA 페이지 생성')
    st.markdown('''
        - 데이터 로드 (파일 업로드)
        - 데이터 분포 확인 (시각화)
        - 데이터 관계 확인 (개별 선택, 시각화)
    ''')
    st.markdown(' 2. Modeling 페이지 생성')
    st.markdown('''
        - 변수 선택 및 데이터 분할
        - 모델링 (하이퍼 파라미터 설정)
        - 모델링 결과 확인 (평가 측도, 특성 중요도)
    ''')
    st.markdown(' 3. Model Serving 페이지 생성')
    st.markdown('''
        - 입력 값 설정 (메뉴)
        - 추론 
    ''')    
    
# 1. file load 함수
# 2. 파일 확장자에 맞게 읽어서 df으로 리턴하는 함수
# 3. 성능 향상을 위해 캐싱 기능 이용
@st.cache_data
def load_file(file):
    
    # 확장자 분리
    ext = file.name.split('.')[-1]
    
    # 확장자 별 로드 함수 구분
    if ext == 'csv':
        return pd.read_csv(file)
    elif 'xls' in ext:
        return pd.read_excel(file, engine='openpyxl')

# file uploader 
# session_state에 다음과 같은 3개 값을 저장하여 관리함
# 1. st.session_state['eda_state'] = {}
#  1.1 : st.session_state['eda_state']['current_file']  / st.session_state['eda_state']['current_data']
# 2. st.session_state['modeling_state'] = {}
# 3. st.session_state['serving_state'] = {}
def file_uploader():
    # 파일 업로더 위젯 추가
    # file = st.file_uploader("파일 선택(csv or excel)", type=['csv', 'xls', 'xlsx'], accept_multiple_files=True)
    file = st.file_uploader("파일 선택(csv or excel)", type=['csv', 'xls', 'xlsx'])
    print(f'files:{file}')
    if file is not None:
        # 새 파일이 업로드되면 기존 상태 초기화
        st.session_state['eda_state'] = {}
        st.session_state['modeling_state'] = {}
        st.session_state['serving_state'] = {}
        
        # 새로 업로드된 파일 저장
        st.session_state['eda_state']['current_file'] = file
    
    # 새로 업로드한 파일을 df로 로드
    if 'current_file' in st.session_state['eda_state']:
        st.write(f"Current File: {st.session_state['eda_state']['current_file'].name}")
        st.session_state['eda_state']['current_data'] = load_file(st.session_state['eda_state']['current_file'])

    # 새로 로드한 df 저장
    if 'current_data' in st.session_state['eda_state']:
        print(st.dataframe(st.session_state['eda_state']['current_data']))
        st.dataframe(st.session_state['eda_state']['current_data'])

# get_info 함수
@st.cache_data
def get_info(col, df):
    # 독립 변수 1개의 정보와 분포 figure 생성 함수
    plt.figure(figsize=(1.5,1))
    
    # 수치형 변수(int64, float64)는 histogram : sns.histplot() 이용
    if df[col].dtype in ['int64', 'float64']:
        ax = sns.histplot(x=df[col], bins=30)
        plt.grid(False)
    # 범주형 변수는 seaborn.barplot 이용
    else:
        s_vc = df[col].value_counts().sort_index()
        ax = sns.barplot(x=s_vc.index, y=s_vc.values)

    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('count')
    sns.despine(bottom = True, left = True)
    fig = ax.get_figure()
    
    # 사전으로 묶어서 반환
    return {'name': col, 'total': df[col].shape[0], 'na': df[col].isna().sum(), 'type': df[col].dtype, 'distribution':fig }
        
# variables 함수
def variables():
    # 각 변수 별 정보와 분포 figure를 출력하는 함수
    
    # 저장된 df가 있는 경우에만 동작
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.columns

        # 열 정보를 처음 저장하는 경우 초기 사전 생성
        if 'column_dict' not in st.session_state['eda_state']:
            st.session_state['eda_state']['column_dict'] = {}
            
        # 모든 열에 대한 정보 생성 후 저장
        for col in cols:
            st.session_state['eda_state']['column_dict'][col] = get_info(col, df)

        # 각 열의 정보를 하나씩 출력
        for col in st.session_state['eda_state']['column_dict']:
            with st.expander(col, expanded=True):
                left, center, right = st.columns((1, 1, 1.5))
                right.pyplot(st.session_state['eda_state']['column_dict'][col]['distribution'], use_container_width=True)
                left.subheader(f"**:blue[{st.session_state['eda_state']['column_dict'][col]['name']}]**")
                left.caption(st.session_state['eda_state']['column_dict'][col]['type'])
                cl, cr = center.columns(2)
                cl.markdown('**Missing**')
                cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']}")
                cl.markdown('**Missing Rate**')
                cr.write(f"{st.session_state['eda_state']['column_dict'][col]['na']/len(df):.2%}")
            

# corr 계산 함수
@st.cache_data
def get_corr(options, df):
    # 전달된 열에 대한 pairplot figure 생성
    pairplot = sns.pairplot(df, vars=options)
    return pairplot.fig
            
# correlation tab 출력 함수
def correlation():
    cols = []
    
    # 저장된 df가 있는 경우에만 동작
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.select_dtypes(['int64', 'float64']).columns
    
    # 상관 관계 시각화를 할 변수 선택 (2개 이상)
    options = st.multiselect(
        '변수 선택',
        cols,
        [],
        max_selections=len(cols))
    
    # 선택된 변수가 2개 이상인 경우 figure를 생성하여 출력
    if len(options)>=2:
        st.pyplot(get_corr(options, df))

        
def missing_data():
    pass
            
# EDA 페이지 출력 함수
def eda_page():
    st.title('Exploratory Data Analysis')
    
    # eda page tab 설정
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2, t3 = st.tabs(['File Upload', 'Variables', 'Correlation'])
    
    with t1:
        file_uploader()
    
    with t2:
        variables()
    
    with t3:
        correlation()
        
        
# 독립 변수 선택 및 데이터 분할 함수
def select_split():
    cols = []
    selected_features = []
    selected_label = []
    split_rate = 0
    
    # 저장된 df가 있는 경우에만 실행
    if 'current_data' in st.session_state['eda_state']:
        df = st.session_state['eda_state']['current_data']
        cols = df.columns
    
    # 이미 저장된 선택된 독립 변수가 있으면 그대로 출력
    if 'selected_features' in st.session_state['modeling_state']:
        selected_features = st.session_state['modeling_state']['selected_features']

    # 이미 저장된 선택된 종속 변수가 있으면 그대로 출력
    if 'selected_label' in st.session_state['modeling_state']:
        selected_label = st.session_state['modeling_state']['selected_label']
        
    # 이미 설정된 분할 비율이 있으면 그대로 출력
    if 'split_rate' in st.session_state['modeling_state']:
        split_rate = st.session_state['modeling_state']['split_rate']
    
    # 이미 설정된 랜덤 시드 값이 있으면 그대로 출력
    if 'split_rs' in st.session_state['modeling_state']:
        split_rs = st.session_state['modeling_state']['split_rs']
    
    # 독립 변수 선택
    with st.form('feature_selection'):
        selected_features = st.multiselect(
            '독립 변수 선택',
            cols,
            selected_features,
            max_selections=len(cols))
        
        submitted = st.form_submit_button('Select')
        if submitted:
            st.session_state['modeling_state']['selected_features'] = selected_features
        st.write(f'선택된 독립 변수: {selected_features}')
    
    # 독립 변수로 선택된 변수 제외
    cols = list(set(cols)-set(selected_features))
    
    # 종속 변수 선택
    with st.form('label_selection'):
        selected_label = st.multiselect(
            '종속 변수 선택',
            cols,
            selected_label,
            max_selections=1)
        
        submitted = st.form_submit_button('Select')
        if submitted:
            st.session_state['modeling_state']['selected_label'] = selected_label
        st.write(f'선택된 종속 변수: {selected_label}')
    
    # 분할 비율(test_size) 및 랜덤 시드 설정
    with st.form('Split Rate'):
        split_rate = st.slider('Test Rate', 0.1, 0.9, 0.25, 0.01)
        split_rs = st.slider('random_state', 0, 100000, 0, 1)
        
        submitted = st.form_submit_button('Confirm')
        if submitted:
            st.session_state['modeling_state']['split_rate'] = split_rate
            st.session_state['modeling_state']['split_rs'] = split_rs
        st.write(f'분할 비율 - 학습: {(1-split_rate):.2%}, 평가: {split_rate:.2%}')
        st.write(f'랜덤 시드: {split_rs}')

# 하이퍼 파라미터 설정
def set_hyperparamters(model_name):
    param_list = {
        'Random Forest Regressor':{
            'n_estimators':[1, 3000, 100, 1], 
            'min_samples_leaf':[1, 100, 1, 1],
            'min_samples_split':[2, 100, 2, 1],
            'random_state':[0, 100000, 0, 1]},
        'Gradient Boosting Regressor':{
            'n_estimators':[1, 3000, 100, 1],
            'max_depth':[1, 100, 3, 1],
            'min_samples_leaf':[1, 100, 1, 1],
            'min_samples_split':[2, 100, 2, 1],
            'subsample':[0.0, 1.0, 1.0, 0.01],
            'learning_rate':[0.0, 1.0, 0.1, 0.01],
            'random_state':[0, 100000, 0, 1]}
    }
    ret = {}
    with st.form('hyperparameters'):
        for key, item in param_list[model_name].items():
            ret[key] = st.slider(key, *item)
        
        submitted = st.form_submit_button('Run')
        
        if submitted:
            st.write(ret)
            return ret

# split data
def split_data():
    df = st.session_state['eda_state']['current_data']
    X = df.loc[:, st.session_state['modeling_state']['selected_features']].values
    Y = df.loc[:, st.session_state['modeling_state']['selected_label']].values.reshape(-1)
    
    test_size = st.session_state['modeling_state']['split_rate']
    seed = st.session_state['modeling_state']['split_rs']
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=seed, test_size=test_size)
    
    return x_train, x_test, y_train, y_test
       
# train_model
def train_model(selected_model, model_name):
    with st.spinner('데이터 분할 중...'):
        x_train, x_test, y_train ,y_test = split_data()
        time.sleep(1)
    st.success('분할 완료')
    time.sleep(1)

    with st.spinner('학습 중...'): 
        model = selected_model(**st.session_state['modeling_state']['hyperparamters'])
        model.fit(x_train, y_train)
    st.success('학습 완료')

    with st.spinner('예측 값 생성 중...'):
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
    st.success('예측 값 생성 완료')
    
    file_name = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    
    # 모델 파일 저장
    with open(f'./models/model_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
        pickle.dump(model, f)
    
    # 학습에 사용된 독립 변수 목록 저장 (순서)
    with open(f'./models/meta_{model_name.replace(" ", "_")}_{file_name}.dat', 'wb') as f:
        pickle.dump(st.session_state['modeling_state']['selected_features'], f)
        
    return model, y_train, train_pred, y_test, test_pred
        
# modeling 함수
def modeling():
    # 모델링 tab 출력 함수
    model_list = ['Select Model', 'Random Forest Regressor', 'Gradient Boosting Regressor']
    model_dict = {'Random Forest Regressor': RandomForestRegressor, 'Gradient Boosting Regressor':GradientBoostingRegressor}
    selected_model = ''
    
    if 'selected_model' in st.session_state['modeling_state']:
        selected_model = st.session_state['modeling_state']['selected_model']
    if 'hyperparamters' in st.session_state['modeling_state']:
        hps = st.session_state['modeling_state']['hyperparamters']
    
    selected_model = st.selectbox(
        '학습에 사용할 모델을 선택하세요.',
        model_list, 
        index=0)

    if selected_model in model_list[1:]:
        st.session_state['modeling_state']['selected_model'] = selected_model
        hps = set_hyperparamters(selected_model)
        st.session_state['modeling_state']['hyperparamters'] = hps
        
        if hps != None:
            model, y_train, train_pred, y_test, test_pred = train_model(model_dict[selected_model], selected_model)
            st.session_state['modeling_state']['model'] = model
            st.session_state['modeling_state']['y_train'] = y_train
            st.session_state['modeling_state']['y_test'] = y_test
            st.session_state['modeling_state']['train_pred'] = train_pred
            st.session_state['modeling_state']['test_pred'] = test_pred
            st.success('학습 종료')

# 결과 tab 함수
def results():
    with st.expander('Metrics', expanded=True):
        if 'y_train' in st.session_state['modeling_state']:
            st.divider()
            st.caption('Train Results')
            c1, c2, c3 = st.columns(3)
            left, right = c1.columns(2)
            mse = mean_squared_error(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            left.write('**:blue[MSE]**')
            right.write(f'{mse: 10.5f}')

            left, right = c2.columns(2)
            mse = mean_absolute_error(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            left.write('**:blue[MAE]**')
            right.write(f'{mse: 10.5f}')

            left, right = c3.columns(2)
            mse = r2_score(st.session_state['modeling_state']['y_train'], st.session_state['modeling_state']['train_pred'])
            left.write('**:blue[$R^2$]**')
            right.write(f'{mse: 10.5f}')
        if 'y_test' in st.session_state['modeling_state']:
            st.divider()
            st.caption('Test Results')
            c1, c2, c3 = st.columns(3)
            left, right = c1.columns(2)
            mse = mean_squared_error(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            left.write('**:blue[MSE]**')
            right.write(f'{mse: 10.5f}')

            left, right = c2.columns(2)
            mse = mean_absolute_error(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            left.write('**:blue[MAE]**')
            right.write(f'{mse: 10.5f}')

            left, right = c3.columns(2)
            mse = r2_score(st.session_state['modeling_state']['y_test'], st.session_state['modeling_state']['test_pred'])
            left.write('**:blue[$R^2$]**')
            right.write(f'{mse: 10.5f}')
        
    st.divider()
    
    with st.expander('Result Analysis', expanded=False):
        if 'y_train' in st.session_state['modeling_state']:

            data = {
                'real': st.session_state['modeling_state']['y_train'],
                'prediction': st.session_state['modeling_state']['train_pred']
            }
            result = pd.DataFrame(data)
            plot = sns.lmplot(x='real', y='prediction', data=result, line_kws={'color':'red'})
            plt.title('Train Results')
            fig = plot.fig
            st.pyplot(fig, use_container_width=True)

            plt.figure()

            data = {
                'real': st.session_state['modeling_state']['y_test'],
                'prediction': st.session_state['modeling_state']['test_pred']
            }
            result = pd.DataFrame(data)
            plot = sns.lmplot(x='real', y='prediction', data=result, line_kws={'color':'red'})
            plt.title('Test Results')
            fig = plot.fig
            st.pyplot(fig, use_container_width=True)
     
    st.divider()
    
    with st.expander('Feature Importances', expanded=False):
        if 'model'  in st.session_state['modeling_state']:
            plt.figure()
            plot = sns.barplot(x=st.session_state['modeling_state']['selected_features'],
                               y=st.session_state['modeling_state']['model'].feature_importances_)
            plt.title('Feature Importances')
            plt.xticks(rotation=90)
            fig = plot.get_figure()
            st.pyplot(fig, use_container_width=True)


# Modeling 페이지 출력 함수
def modeling_page():
    st.title('ML Modeling')
    
    # tabs를 추가하세요.
    # tabs에는 File Upload, Variables (type, na, 분포 등), Correlation(수치)이 포함됩니다.
    t1, t2, t3 = st.tabs(['Data Selection and Split', 'Modeling', 'Results'])

    # file upload tab 구현
    with t1:
        select_split()
    
    with t2:
        modeling()
    
    with t3:
        results()

# 추론 함수
def inference():
    model = st.session_state['serving_state']['model']
    model_name = st.session_state['serving_state']['model_name']
    model_name = model_name.removeprefix('model_').replace('_', ' ')
    
    if 'meta' in st.session_state['serving_state']:
        placeholder = ', '.join(st.session_state['serving_state']['meta'])
    else:
        placeholder = ''
    
    with st.expander('Inference', expanded=True):
        st.caption(model_name)
        input_data = st.text_input(
        label='예측하려는 값을 입력하세요.',
        placeholder=placeholder)

        if input_data:
            input_data = [[float(s) for s in input_data.split(',')]]

            left, center, right = st.columns(3)
            left.write('**:blue[입력]**')
            center.write(input_data)

            left, center, right = st.columns(3)
            left.write('**:blue[출력]**')
            center.write(model.predict(np.array(input_data)))

# Serving 페이지 출력 함수
def serving_page():
    st.title('ML Serving')
    
    with st.form('select pre-trained model'):
        # 모델 파일 목록 불러오기
        model_paths = glob.glob('./models/model_*')
        model_paths.sort(reverse=True)
        model_list = [s.removeprefix('./models/').removesuffix('.dat') for s in model_paths]
        model_dict = {k:v for k, v in zip(model_list, model_paths)}
        model_list = ['Select Model'] + model_list
        
        # 추론에 사용할 모델 선택
        
        selected_inference_model = st.selectbox('추론에 사용할 모델을 선택하세요.', model_list, index=0)
        checked = st.checkbox('독립 변수 정보')
        
        submitted = st.form_submit_button('Confirm')
        
        if submitted:
            st.session_state['serving_state'] = {}
            with open(model_dict[selected_inference_model], 'rb') as f_model:
                inference_model = pickle.load(f_model)
                st.session_state['serving_state']['model'] = inference_model
                st.session_state['serving_state']['model_name'] = selected_inference_model
                if checked:
                    with open(model_dict[selected_inference_model].replace('model_', 'meta_'), 'rb') as f_meta:
                        metadata = pickle.load(f_meta)
                        st.session_state['serving_state']['meta'] = metadata
                placeholder = st.empty()
                placeholder.success('모델 불러오기 성공')
                time.sleep(2)
                placeholder.empty()
                
    if 'model' in st.session_state['serving_state']:
        inference()
        
                
# session_state에 사전 sidebar_state, eda_state, modeling_state, serving_state를 추가하세요.
if 'sidebar_state' not in st.session_state:
    st.session_state['sidebar_state'] = {}
    st.session_state['sidebar_state']['current_page'] = front_page
if 'eda_state' not in st.session_state:
    st.session_state['eda_state'] = {}
if 'modeling_state' not in st.session_state:
    st.session_state['modeling_state'] = {}
if 'serving_state' not in st.session_state:
    st.session_state['serving_state'] = {}
    
# sidebar 추가
with st.sidebar:
    st.subheader('Dashboard Menu')
    b1 = st.button('Front Page', use_container_width=True)
    b2 = st.button('EDA Page', use_container_width=True)
    b3 = st.button('Modeling Page', use_container_width=True)
    b4 = st.button('Serving Page', use_container_width=True)
    
if b1:
    st.session_state['sidebar_state']['current_page'] = front_page
#     st.session_state['sidebar_state']['current_page']()
    front_page()
elif b2:
    st.session_state['sidebar_state']['current_page'] = eda_page
#     st.session_state['sidebar_state']['current_page']()
    eda_page()
elif b3:
    st.session_state['sidebar_state']['current_page'] = modeling_page
#     st.session_state['sidebar_state']['current_page']()
    modeling_page()
elif b4:
    st.session_state['sidebar_state']['current_page'] = serving_page
#     st.session_state['sidebar_state']['current_page']()
    serving_page()
else:
    st.session_state['sidebar_state']['current_page']()
