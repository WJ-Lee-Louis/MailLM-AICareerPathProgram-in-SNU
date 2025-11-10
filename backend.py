##########################################################
## [1] Enron Mail Dataset                               ##
## - 개인 'parsed_enron_all.csv' 파일 경로에 맞추어 설정 ##
## - 변수명: enron_dataset                              ##
## - To 항목 공란(NaN) 삭제                             ##
## - 자기자신에게 보낸 메일 삭제                         ##
## - Date 정보 utc 기준으로 통일                         ##
##########################################################
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv('parsed_enron_all.csv')

    def clean_enron_emails(df):
        df_processed = df.copy()
        initial_count = len(df_processed)

        # 1. 'To' 항목이 공란(NaN)인 메일 개수 계산 (삭제 대상)
        to_nan_mask = df_processed['To'].isna()
        to_nan_count = to_nan_mask.sum()

        # 2 & 3. process_recipients 함수를 적용하여 'To' 필드 처리
        def process_recipients(row):
            from_email = str(row['From']).strip().lower()
            to_field = row['To']

            # To 필드가 문자열이 아닌 경우(NaN 등)는 변경하지 않음
            if not isinstance(to_field, str) or not to_field:
                return to_field

            recipients = {email.strip().lower() for email in to_field.split(',')}

            if from_email in recipients:
                if len(recipients) == 1:
                    return None  # 자기 자신에게만 보낸 경우 -> 삭제 대상
                else:
                    recipients.remove(from_email)
                    return ', '.join(recipients) # 자기 자신과 타인에게 보낸 경우 -> 수정
            else:
                return to_field # 그 외 -> 유지
        df_processed['To_processed'] = df_processed.apply(process_recipients, axis=1)

        # 2. 자기 자신에게만 보낸 메일 개수 계산 (삭제 대상)
        # To_processed가 None이면서, 원래 To는 NaN이 아니었던 경우
        self_only_mask = df_processed['To_processed'].isna() & df_processed['To'].notna()
        self_only_count = self_only_mask.sum()

        # 3. 자기 자신과 다른 사람에게 보낸 메일 개수 계산 (수정 대상)
        modified_mask = (df_processed['To_processed'].notna()) & (df_processed['To_processed'] != df_processed['To'])
        modified_count = modified_mask.sum()

        # 4. 최종 데이터셋 생성
        # 삭제 대상(To가 NaN이거나, 자기 자신에게만 보낸 메일)을 모두 제외
        final_df = df_processed[~to_nan_mask & ~self_only_mask].copy()

        # 처리된 'To_processed' 컬럼을 'To' 컬럼에 반영
        final_df['To'] = final_df['To_processed']
        # 원본 컬럼 목록을 유지하기 위해 불필요한 임시 컬럼 삭제
        final_df = final_df[df.columns]

        final_count = len(final_df)
        total_removed = initial_count - final_count

        return final_df

    enron_dataset = clean_enron_emails(df)
    return enron_dataset

enron_dataset = load_data()

# Date 정보를 utc 기준으로 통일
enron_dataset['Date'] = pd.to_datetime(
    enron_dataset['Date'],
    format='%a, %d %b %Y %H:%M:%S %z',
    errors='coerce',
    utc=True
)

#############################################################
## [2] 특정 메일 주소의 보낸 메일함                         ##
## - 변수명: personal_sent_dataset                         ##
## - 필요한 사전 정의 데이터셋: [1] enron_dataset           ##
## - 필요 파라미터: [1] enron_dataset 과 특정메일 주소(str) ##
#############################################################
@st.cache_data
def create_outbox(df, email_address):
    # 이메일 주소 정규화 (공백 제거, 소문자 변환)
    email_address = email_address.strip().lower()

    # 'From' 컬럼이 해당 이메일 주소와 일치하는 행을 필터링
    outbox_df = df[df['From'].str.strip().str.lower() == email_address].copy()
    outbox_df = outbox_df[['Message-ID', 'From', 'To', 'Date', 'Body']]

    return outbox_df

personal_sent_dataset = create_outbox(enron_dataset, "jeff.dasovich@enron.com")


#############################################################
## [3] 특정 메일 주소의 받은 메일함                         ##
## - 변수명: personal_received_dataset                     ##
## - 필요한 사전 정의 데이터셋: [1] enron_dataset           ##
## - 필요 파라미터: [1] enron_dataset 과 특정메일 주소(str) ##
#############################################################
@st.cache_data
def create_inbox(df, email_address):
    # 이메일 주소 정규화 (공백 제거, 소문자 변환)
    email_address = email_address.strip().lower()

    # 'To' 컬럼에 해당 이메일 주소를 포함하는 행을 필터링 (NaN 값은 제외)
    inbox_df = df[df['To'].str.lower().str.contains(email_address, na=False)].copy()
    inbox_df = inbox_df[['Message-ID', 'From', 'To', 'Date', 'Body']]

    return inbox_df

personal_received_dataset = create_inbox(enron_dataset, "jeff.dasovich@enron.com")


#################################################################
## [4] 특정 메일 주소의 통합 메일함                             ##
## - 변수명: personal_combined_dataset                         ##
## - 필요한 사전 정의 데이터셋: [1] enron_dataset,              ##
##   [2] personal_sent_dataset, [3] personal_received_dataset  ##
## - 필요 파라미터: [1] enron_dataset 과 특정메일 주소(str)     ##
## - 메일ID 기준으로 혹시 모를 중복된 메일까지 제거              ##
##   (확인결과, 실제 제거되는 메일 수는 극소수)                  ##
## - date 기준으로 데이터셋 재정렬                              ##
#################################################################
@st.cache_data
def create_combined_box(df, email_address):
    # 위에서 정의한 함수들을 재사용하여 발신 및 수신 메일함 생성
    outbox_df = create_outbox(df, email_address)
    inbox_df = create_inbox(df, email_address)

    # 두 데이터프레임을 하나로 합침
    combined_df = pd.concat([outbox_df, inbox_df])

    # Message-ID 컬럼을 기준으로 실제 중복 메일만 제거
    combined_df_deduped = combined_df.drop_duplicates(subset=['Message-ID']).reset_index(drop=True)

    return combined_df_deduped

personal_combined_dataset = create_combined_box(enron_dataset, "jeff.dasovich@enron.com")
# date 기준으로 데이터셋 재정렬
personal_combined_dataset = personal_combined_dataset.sort_values(by='Date').reset_index(drop=True)
# date 기준으로 정렬된 통합메일함(보낸메일함+받은메일함)


##################################################################
## [5] 최근 n일(입력) 기간 동안의 메일함(Train Dataset, n=90)     ##
## - 변수명: recent_ndays_dataset                                ##
## - 필요한 사전 정의 데이터셋: [4] personal_combined_dataset     ##
## - 필요 파라미터: [4] personal_combined_dataset,               ##
##   현재날짜(str), 최근n일에 대한 n값(int)                       ##
## - 인자로 입력한 현재날짜 기준으로 최근 n일 동안의 메일함을 구축 ##
##################################################################
@st.cache_data
def get_recent_emails(df, current_date, n):
    # 1. 입력받은 날짜 문자열을 datetime 객체로 변환
    end_date = pd.to_datetime(current_date)

    # 2. n일 전의 시작 날짜를 계산
    start_date = end_date - pd.Timedelta(days=n)

    # 3. 데이터셋의 'Date' 컬럼(UTC)과 비교하기 위해 기준 날짜들도 UTC로 통일
    start_date_utc = start_date.tz_localize('UTC')
    end_date_utc = end_date.tz_localize('UTC')

    # 4. 시작 날짜와 종료 날짜 사이의 데이터를 필터링
    # [start_date <= email_date < end_date] 범위의 메일을 선택
    mask = (df['Date'] >= start_date_utc) & (df['Date'] < end_date_utc)
    recent_ndays_dataset = df[mask].copy()

    return recent_ndays_dataset

personal_combined_dataset = create_combined_box(enron_dataset, "jeff.dasovich@enron.com")
# date 기준으로 데이터셋 재정렬
personal_combined_dataset = personal_combined_dataset.sort_values(by='Date').reset_index(drop=True)

date = '2002-01-01'
n = 90
recent_ndays_dataset = get_recent_emails(personal_combined_dataset, date, n)


#################################################################
## [6] Jeff의 받은메일함(Test Dataset)                         ##
## - 변수명: future_ndays_dataset                              ##
## - 필요한 사전 정의 데이터셋: [3] personal_received_dataset   ##
## - 필요 파라미터: [3] personal_received_dataset,             ##
##                 현재날짜(str), 이후n일에 대한 n값(int)       ##
## - 인자로 입력한 현재날짜 기준으로 n일 이후까지의 받은메일함   ##
#################################################################
@st.cache_data
def get_emails_for_ndays_after(df, start_date_str, n):
    # 1. 입력받은 시작 날짜 문자열을 datetime 객체로 변환
    start_date = pd.to_datetime(start_date_str)

    # 2. n일 후의 종료 날짜를 계산
    end_date = start_date + pd.Timedelta(days=n + 1)

    # 3. 데이터셋의 'Date' 컬럼(UTC)과 비교하기 위해 기준 날짜들도 UTC 기준으로 통일
    start_date_utc = start_date.tz_localize('UTC')
    end_date_utc = end_date.tz_localize('UTC')

    # 4. 시작 날짜와 종료 날짜 사이의 데이터를 필터링
    mask = (df['Date'] >= start_date_utc) & (df['Date'] < end_date_utc)
    future_ndays_dataset = df.loc[mask].copy()

    return future_ndays_dataset

personal_received_dataset = create_inbox(enron_dataset, "jeff.dasovich@enron.com")
# date 기준으로 데이터셋 재정렬
personal_received_dataset = personal_received_dataset.sort_values(by='Date').reset_index(drop=True)

date = '2002-01-01'
n = 20
future_ndays_dataset = get_emails_for_ndays_after(personal_received_dataset, date, n)


role = "You are a highly efficient and accurate email prioritization and summarization expert. Your primary goal is to help the user quickly grasp the most important information from a batch of emails. Be precise and concise in your analysis."

prompt_format="""
The following information is provided:

(1) Network Summary:
This shows the user's most frequent contacts over the past n days(user parameter), including the number of received and sent emails per address:
{recent_network}

(2) Importance Keywords:
The user has indicated the following keywords as important. Emails containing these keywords or similar words should be prioritized:
{important_keywords}

(3) Received Emails:
Here are the new {N} emails to be analyzed. Each email includes From, To, Date, and Body and separated by the string "\n\n===separatrix===\n\n":
The total number of received emails is {N}:
{received_emails}

(4) Priority Evaluation Criteria:
Use the following logic to evaluate the priority of each email.
Criterion 1: Emails from addresses in recent_network are more likely to be important.
Criterion 2: Emails containing any of the important_keywords in their body should be considered more important.
For example,
Critical: Requires immediate attention (within hours). High impact if ignored. Often sent from key contacts or includes urgent, deadline-sensitive language.
High: Requires action within 1–2 days. Moderate-to-high impact. Often includes important keywords or comes from important/recent contacts.
Low: Informational only. No immediate action required. Includes newsletters, announcements, or non-urgent content.

(5) Step-by-Step Instructions:
Step 1: Count the total number of received emails and check whether it is equal to {N}. You must check the "\n\n===separatrix===\n\n" delimiter as the only splitting method.
Step 2: For each email, analyze its priority based on the two criteria described above.
Step 3: For each email, provide a concise reason (up to 3 lines) why the email was assigned its priority.
Step 4: For each email, provide a concise summary (up to 3 lines) of the email. Focus on the core message, key actions, people, dates, and places. Do not interpret—just extract factual content.
Step 5: Before you present the analysis outputs, count the number of mails assigned to each priority level. The exact sum of each priority level should be {N}. Please check this again and again, and if the exact sum is not equal to {N}, retry the analysis process more precisely.
Step 6: After you strictly check whether the above steps are processed correctly, present the analysis in decreasing order of priority: Critical → High → Low.

(6) Output Constraints:
For each email, output the following format.
Priority: One of [Critical / High / Low]
Reason: The reason why the email was assigned this priority
From: The sender of the email
Date: The date of the email
Summary: The summary of the email

(7) Warning(You must strictly follow):
Please pay more attention to counting the numbers I requested in the above.
You must check again and again whether the total number of received emails is {N}.
Analyze all emails with care. Consider sender frequency, keyword relevance, and specific user-defined instructions.
For reply emails, focus on the new message content, not the quoted text from the original email.
You must check again and again whether the number of your analysis outputs is {N}.
If the number of your analysis outputs is not equal to {N}, you must retry the entire analysis process of received emails more precisely.
Again, you must retry the whole process until the total number of your analysis outputs is equal to {N}.

(8) Output Examples:
<I will give three emails like the below.>

Original Email #1 Content:
From: susan.mara@enron.com
To: me@example.com
Date: 2025-07-13
Body: Hi, this is an urgent request. We need your final approval on Project X's budget by 5 PM KST today. Without it, we cannot proceed, which will significantly delay the launch. Please respond immediately. This is a critical item.


===separatrix===


Original Email #2 Content:
From: teammate@example.com
To: me@example.com
Date: 2025-07-11
Body: Could you please provide the Q2 sales report data by end of this week? I need it to finalize the presentation for next Monday's management review. Your prompt response would be greatly appreciated.


===separatrix===


Original Email #3 Content:
From: newsletter@example.com
To: me@example.com
Date: 2025-07-12
Body: Here's your weekly digest of industry news. No specific action is required from your side. It contains general updates on market trends.

<Then, you should response like the below.>
Total Emails Analyzed: 3

Priority: Critical
Reason: Sent by a frequent contact (susan.mara@enron.com), contains 'Urgent', 'Immediate Action Needed', 'Final Approval', 'critical' keywords and mentions a tight deadline (by 5 PM KST today) with significant impact (delay launch).
From: susan.mara@enron.com
Date: 2025-07-13
Summary: Urgent final approval for Project X budget due by 5 PM KST today. Lack of approval will significantly delay project launch.

Priority: High
Reason: Contains 'Request for Data' (related to 'request for information' concept), 'report' keyword, and has a clear request with a deadline (by end of this week) for an important presentation.
From: teammate@example.com
Date: 2025-07-11
Summary: Request for Q2 sales report data needed by end of the week to finalize the presentation for next Monday's management review.

Priority: Low
Reason: General informational email (newsletter), no keywords indicating urgency or action required.
From: newsletter@example.com
Date: 2025-07-12
Summary: Weekly digest of industry news, providing general updates on market trends with no specific action needed.
"""

def analyze_email_network(recent_ndays_dataset: pd.DataFrame, target_email: str):
    target_email = target_email.strip().lower()
    df = recent_ndays_dataset.copy()

    # 소문자 정규화 및 날짜 파싱
    df['From'] = df['From'].astype(str).str.lower()
    df['To']   = df['To'].astype(str).str.lower()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # From/To 분리 리스트화
    recent_ndays_dataset.loc[:, 'from_list'] = (
        recent_ndays_dataset['From'].str.split(',').apply(lambda lst: [s.strip() for s in lst])
    )
    recent_ndays_dataset.loc[:, 'to_list'] = (
        recent_ndays_dataset['To'].str.split(',').apply(lambda lst: [s.strip() for s in lst])
    )

    # 발신 분석
    sent_df = recent_ndays_dataset[recent_ndays_dataset['from_list'].apply(lambda lst: target_email in lst)]
    sent_counts = (
        sent_df.explode('to_list')['to_list']
        .value_counts()
        .rename_axis('Contact')
        .reset_index(name='emails_sent')
    )

    # 수신 분석
    received_df = recent_ndays_dataset[recent_ndays_dataset['to_list'].apply(lambda lst: target_email in lst)]
    received_counts = (
        received_df.explode('from_list')['from_list']
        .value_counts()
        .rename_axis('Contact')
        .reset_index(name='emails_received')
    )

    # 발신/수신 합치기 및 정리
    connectivity_df = pd.merge(sent_counts, received_counts, on='Contact', how='outer').fillna(0)
    connectivity_df['emails_sent'] = connectivity_df['emails_sent'].astype(int)
    connectivity_df['emails_received'] = connectivity_df['emails_received'].astype(int)
    connectivity_df['total_interactions'] = (connectivity_df['emails_sent'] + connectivity_df['emails_received'])
    connectivity_df = (
        connectivity_df[connectivity_df['Contact'] != target_email]
        .sort_values('total_interactions', ascending=False)
        .reset_index(drop=True)
    )

    # connectivity_df에서 상위 5명의 정보를 추출
    top_5_list = connectivity_df.head(5).rename(columns={
        'Contact': 'adress',
        'emails_received': 'received',
        'emails_sent': 'sent'
    })[['adress', 'received', 'sent']].to_dict('records')

    formatted_lines = [
        f"{{'adress': '{item['adress']}', 'received': {item['received']}, 'sent': {item['sent']}}}"
        for item in top_5_list
    ]
    recent_network = ",\n".join(formatted_lines)

    return recent_network

# 타겟 이메일에 대한 네트워크 분석
target = 'jeff.dasovich@enron.com'
recent_network = analyze_email_network(recent_ndays_dataset, target)

important_keywords = """
회신 요청: ['reply', 'response', 'respond', 'feedback'],
자료 요청: ['document', 'file', 'report', 'data', 'information', 'materials'],
마감: ['deadline', 'due', 'date', 'end', 'eod', 'asap'],
긴급: ['urgent', 'asap', 'immediate', 'critical', 'emergency'],
중요: ['important', 'key', 'major', 'essential'],
확인 부탁: ['review', 'check', 'confirm', 'look', 'examine'],
피드백 요청: ['feedback', 'comments', 'thoughts', 'opinion', 'suggestion'],
결재: ['approval', 'approve', 'sign', 'signature', 'authorize'],
보고: ['report', 'update', 'status', 'summary', 'briefing']
"""

email_text_list = []

for index, row in future_ndays_dataset.iterrows():
    # 데이터셋에서 모든 내용을 텍스트로 전환
    email_block = (
        f"From: {row['From']}\n"
        f"To: {row['To']}\n"
        f"Date: {row['Date']}\n"
        f"Body: {row['Body']}"
    )
    email_text_list.append(email_block)

# 각 이메일에 대한 텍스트 구분 ("\n\n===separatrix===\n\n")
received_emails = "\n\n===separatrix===\n\n".join(email_text_list)
total_num_of_received_emails = len(email_text_list)

#####################################################
## (1) target 이메일 지정: target                  ##
## (2) 네트워크 분석 결과: recent_network          ##
## (3) 중요 키워드 리스트: important_keywords      ##
## (4) 분석대상 이메일 개수: N                     ##
## (4) LLM 분석 대상 이메일 내용: received_emails  ##
## (최종) LLM 답변 생성: completion                ##
#####################################################
target = 'jeff.dasovich@enron.com'
recent_network = analyze_email_network(recent_ndays_dataset, target)
important_keywords = important_keywords
received_emails = received_emails
prompt = prompt_format.format(
    recent_network=recent_network,
    important_keywords=important_keywords,
    N=total_num_of_received_emails,
    received_emails=received_emails
)