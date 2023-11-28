
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import datetime, time

#==== SideBar

# 커스텀 CSS를 정의하고 스트림릿에 삽입
st.markdown(
    """
    <style type='text/css'>
    
    .reportview-container .main .block-container {
        max-width: 800px;
    }
    .reportview-container .main {
        color: blue;
        background-color: red;
    }
    /* 중앙 컬럼의 CSS 클래스 이름을 찾아서 지정해야 합니다. 
    .st-emotion-cache-5rimss {
        background-color: lightgreen;
    }
    */
    .st-emotion-cache-6qob1r{
        background-color: lightyellow;
    }
    

    .st-emotion-cache-1629p8f e1nzilvr2 h3{
        font-size: 50px;
        color:pink
    }/*
    
    h3 {
       여기에 원하는 스타일 속성을 추가하세요 */
      font-size: 40px; /* 예: 글자 크기를 20px로 설정 */
      color: orange; /* 예: 글자 색상을 파란색으로 설정 */
      margin: 0; /* 예: 외부 여백을 없앰 */
      padding: 10px; /* 예: 내부 여백을 10px로 설정 */
      /* 추가적으로 원하는 스타일 속성을 계속해서 추가할 수 있습니다. */
    }
    
    /*
    .st-emotion-cache-1629p8f e1nzilvr2 h3 {
        font-size: 2.0rem;
        font-weight: 600;
        color: orange !important;
    }
    */
    
    </style>
    """,
    unsafe_allow_html=True
)

#st-emotion-cache-5rimss e1nzilvr5
#st-emotion-cache-5rimss e1nzilvr5

st.title('sidebar 실습')
st.header('첫번째 방법')

lang = st.sidebar.selectbox(
    '언어를 선택하세요',
    ['한국어', '영어'],
    index=0,
    label_visibility='collapsed'
)

if lang == '한국어':
    st.header(':red[한국어]')
else:
    st.header(':blue[영어]')

#===== 두번째 방법

with st.sidebar:
    st.sidebar.subheader('날짜를 선택하세요')
    date = st.date_input(
        '날짜를 선택하세요',
        datetime.datetime(2023,10,27),
        label_visibility='collapsed'
    )
    
    st.sidebar.subheader('목적지를 선택하세요')
    dest = st.selectbox(
        '목적지를 선택하세요',
        ['순천만','여수'],
        index=1,
        label_visibility='collapsed'
    )
    
    if date and dest :
        st.subheader(f'당신은 :red[{date}]에 :blue[{dest}]로 출발할 예정입니다.')
    
    
st.divider()

col1, col2, col3 = st.columns(3)
col1.subheader(':red[왼쪽]입니다.')
col2.subheader(':blue[중앙]입니다.')
col3.subheader(':green[오른쪽]입니다.')

col1, col2, col3 = st.columns((1, 8, 1))
col1.subheader(':red[왼쪽]입니다.')
col2.subheader(':blue[중앙]입니다.')
col3.subheader(':green[오른쪽]입니다.')

col1, col2, col3 = st.columns((1, 8, 1), gap='large')
col1.subheader(':red[왼쪽]입니다.')
col2.subheader(':blue[중앙]입니다.')
col3.subheader(':green[오른쪽]입니다.')

st.divider()

st.header('마크 다운 정렬하기')
st.markdown("<h3 style='text-align: center; color: blue;'>GS 칼텍스</h3>", unsafe_allow_html=True)
st.text('')

st.divider()

st.header('tab 실습')

tab1, tab2, tab3 = st.tabs([':smile_cat: Cat',':guide_dog: Dog',':owl: Owl'])

with tab1:
    st.header('A Cat :smile_cat:')
with tab2:
    st.header('A Dog :guide_dog:')
with tab3:
    st.header('A Owl :owl:')
    
st.divider()

st.header('Expander')
#st.expander : 열기, 접기

st.bar_chart({'data':[1,5,6,3,4]})

with st.expander('설명보기'):
    st.write('''
        이 데이터는 
        가상 데이터로써 어쩌구 저쩌구.....
    ''')
    st.image('https://static.streamlit.io/examples/dice.jpg')


st.divider()        

st.header('Container')
with st.container():
    st.write("컨테이너 내부입니다.")
    st.subheader('컨테이너 내부 서브헤더')
    # 커스텀 컴포넌트를 포함한 모든 streamlit command 사용 가능
    st.bar_chart(np.random.randn(50, 3))

st.write("컨테이너 외부입니다.")


st.divider()
st.subheader('out of order 테스트')

# out of order
container = st.container()
container.write("컨테이너 내부 write입니다.")
st.write("컨테이너 외부 write입니다.")

# 컨테이너에 컴포넌트 추가
container.write("두 번째 컨테이너 내부 write입니다.")



st.header('Empty')
st.subheader('in-place overwrite')

# st.empty
# empty는 순차적으로 수행
# overwrite in-place
with st.empty():
    for seconds in range(10):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
    st.write("✔️ 1 minute over!")

    
st.divider()
st.subheader('in-place replacing element')

placeholder = st.empty()
time.sleep(2)
placeholder.text("Hello")
time.sleep(2)
placeholder.line_chart({"data": [1, 5, 2, 6]})
time.sleep(2)

# container도 1개 element로 취급
with placeholder.container():
    st.write("This is one element")
    st.write("This is another")
time.sleep(2)

placeholder.write('완료')