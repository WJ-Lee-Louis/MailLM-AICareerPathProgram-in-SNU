#####################################################
## MAIL PRIORIY AND SUMMARIZATION LLM(gpt-4o-mini) ##
#####################################################
from openai import OpenAI
import openai
import os
import streamlit as st
import backend

#######################################
## streamlit 인터페이스 웹서비스 구현 ##
#######################################
if "client" not in st.session_state:
    api_key = os.getenv('OPENAI_API_KEY')
    openai.api_key = api_key
    st.session_state['client']  = OpenAI()

def mail_priority_summarization_LLM_gpt(role, prompt):
    # Step1. 역할지정/ 사전정의한 role 변수를 인자로 받아 사용
    prompt_list = [{'role':'system', 'content':role}]

    # Step2. 구체적인 prompt 입력/ 사전정의한 prompt 변수를 인자로 받아 사용
    prompt_list.append({'role':'user', 'content':prompt})

    # Step3. LLM 답변 생성
    completion = st.session_state['client'].chat.completions.create(
        model='gpt-4o-mini',
        messages=prompt_list,
        stream=True,
        max_tokens=10000
    )
    for c in completion:
        # 마지막에 None 값이 delta content로 들어가 있어서 None은 제외하고 출력하기 위해 if문 설정
        if c.choices[0].delta.content:
            # 한 글자씩 출력하되, print 자동개행 없이 출력
            # print(c.choices[0].delta.content ,end='')
            yield c.choices[0].delta.content # 1개의 문자를 지속적으로 return

st.set_page_config(page_title="Team2 Proj.", layout="wide")
st.title('📧 LLM 기반 신규메일 중요도 분류 및 요약 웹서비스')
st.markdown("안녕하세요, 신속한 업무 처리를 위해 중요 메일부터 요약까지 확인하실 수 있습니다.")

# '우선순위 정렬' 버튼을 클릭하면 바로 답변 생성을 시작합니다.
priority_clicked = st.button('메일 분석 시작 👆', use_container_width=True)

if priority_clicked:
    st.subheader('✅이메일 분석 결과')
    with st.spinner('⏳ 당신의 메일함📁에 접근하여 분석을 진행 중입니다...'):
        output_container = st.empty()
        # mail_priority_summarization_LLM_gpt 함수는 generator를 반환합니다.
        # st.write_stream이 이 generator를 받아 스트리밍 답변을 자동으로 화면에 출력해줍니다.
        response_generator = mail_priority_summarization_LLM_gpt(backend.role, backend.prompt)
        output_container.write_stream(response_generator)