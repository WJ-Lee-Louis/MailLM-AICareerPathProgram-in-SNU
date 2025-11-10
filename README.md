# MailLM-AICareerPathProgram-in-SNU
Built a web-based service that classifies and ranks incoming emails into three levels of importance by prompting an LLM with sender–receiver network summaries, keyword priorities, and email contents.

---

## [1] Project Background and Differentiation Strategy
Despite the rapid advancement and widespread adoption of IT technologies, email remains one of the most trusted and secure communication methods within organizations. While numerous AI-based email analysis models have been developed—such as those for spam filtering, importance classification, and automated responses—there is still a lack of **LLM-based systems** that incorporate **both content understanding and sender–receiver network analysis** for enhanced email prioritization and summarization.

### (1) Differentiation from Rule-Based Systems
- Existing classification tools (e.g., SaneBox, Clean Email) primarily rely on rule-based approaches, which limit their ability to interpret context or provide personalized analysis.  
- Our differentiation: By integrating **network information** with **LLM-driven semantic understanding**, the system enables **personalized and context-aware importance classification** beyond simple rule sets.

### (2) Differentiation from Existing LLM-Based Summarization Services
- Current LLM-powered tools (e.g., Microsoft 365 Copilot, Spark) mainly focus on summarization and reply suggestion.  
- Our differentiation: The proposed model extends beyond summarization, providing **importance-based classification, automatic organization, and network- and context-aware prioritization**, enabling a more intelligent and structured email management experience.

---

## [2] Analysis Framework / Pipeline

<img width="910" height="496" alt="image" src="https://github.com/user-attachments/assets/dea02b87-cf22-495b-8249-34a5416f50f2" />

- **Operation**
  - Provide the LLM with: (1) **recent sender–receiver network information** from the existing mailbox and (2) **predefined importance keywords**.
  - Through a **prompting web interface** (click-to-prompt), automatically supply the classification criteria and output examples to the LLM.
  - The LLM **sorts target emails by importance** and **produces concise summaries**.

- **Service Implementation**
  - Implement a web interface using Streamlit to connect the user and the LLM.

- **Performance Evaluation**
  - Verify performance and compute evaluation metrics using the **Enron Mail Dataset**.
 
---

## [3] Dataset Construction and Preprocessing
- Parse the Enron Mail Dataset (extract 12 columns)
- Preprocess emails with blank To fields (NaN)
- Preprocess self-sent emails
- Normalize Date information to UTC
- Build three mailbox datasets (sender / receiver / combined) filtered by a specific email address and time period
- Select the email address with the largest volume of mail as the analysis target

---

## [4] Prior Information Setup

### (1) Network Analysis

<img width="896" height="782" alt="image" src="https://github.com/user-attachments/assets/c4bd3e87-4d2f-4a04-97dd-ed8eb199683f" />

- Build a combined mailbox dataset for the target email address over the most recent n days.  
- Extract the top 5 counterpart email addresses and their message counts during that period.

### (2) Keyword List

<img width="896" height="748" alt="image" src="https://github.com/user-attachments/assets/d903d65e-01dc-4222-a4f8-5d32a37eba1c" />

| Category | Example Keywords |
|-----------|------------------|
| **Reply Request** | reply, response, respond, feedback |
| **Document / Material Request** | document, file, report, data, information, materials |
| **Deadline** | deadline, due, date, end, eod, asap |
| **Urgent** | urgent, asap, immediate, critical, emergency |
| **Important** | important, key, major, essential |
| **Please Confirm / Review** | review, check, confirm, look, examine |
| **Feedback Request** | feedback, comments, thoughts, opinion, suggestion |
| **Approval** | approval, approve, sign, signature, authorize |
| **Report / Update** | report, update, status, summary, briefing |

- Hard-code the important keyword categories.  
- Prompt the LLM to also recognize words semantically similar to these keywords using its learned embeddings.  
- Scan entire email bodies to verify the presence of important or semantically related terms.

---

## [5] LLM Prompt Engineering

### (0) Role Assignment Prompting
- Assign the LLM the role of an expert in email importance classification and summarization.

### (1) Network Analysis
- Format and provide network analysis results using the variable {recent_network}.

### (2) Important Keyword List
- Format the predefined keyword list using the variable {important_keywords}.

### (3) Target Emails and Email Count
- Extract From, To, Date, and Body information from each target email.  
- Format the extracted content using the variable {received_emails}.  
- Count the total number of target emails to ensure a 1:1 match between analyzed items and output results.  
- Store this number as {N}.

### (4) Importance Classification Criteria and Examples
- **Criterion 1:** Emails from addresses found in the recent network analysis are classified as more important.  
- **Criterion 2:** Emails containing important keywords in their body text are classified as more important.  
- Classify emails into **three levels: Critical / High / Low**, providing short reasoning examples for each level.

### (5) Step-by-Step Analysis (CoT Prompting)
- Instruct the LLM to perform the analysis in six steps.  
- Separate each email with a custom delimiter `\n\n===separatrix===\n\n` to clearly distinguish and count them.  
- Guide the model to follow the given criteria and limit each reasoning and summary to within three lines.  
- Verify that the number of analyzed emails matches the total number of inputs; if not, re-analyze.  
- Output results in descending order of importance.

### (6) Output Format Restriction
- Instruct the LLM to output results in a consistent predefined format, such as JSON, for stable parsing and post-processing.

### (7) Error-Handling Instructions
- Include supplementary prompts emphasizing the need to verify and align the number of input emails and output results.  
- Reiterate that the final output must reflect a complete and consistent analysis.

### (8) Example Outputs (Few-Shot Prompting)
- Provide example input emails ({received_emails}) and corresponding labeled outputs as few-shot examples to guide the LLM’s reasoning and output structure.

---

## [6] LLM Model Evaluation and Comparison

<img width="585" height="81" alt="image" src="https://github.com/user-attachments/assets/057b6e40-a43c-46f1-a742-b480d97df3a4" />

### (1) Model Selection
- **Open-source models:** Gemma 7B IT, Mistral 7B *(analysis failed; requires further tuning and optimization)*  
- **Closed-source API-based models:** Gemini 1.5 Flash, GPT-4o-mini

### (2) Error Analysis and Prompt Refinement
- Issue: Reply emails often contain quoted original messages, reducing the model’s ability to correctly identify the actual analysis target.  
- Issue: When the reply content is short, the model sometimes misclassifies the original quoted message as the primary email body.  
- Improvement:   
  - Add explicit guidance to recheck consistency between the number of analyzed emails** and number of results.  
  - Introduce clear delimiters to separate original and quoted email content for precise recognition.

### (3) Evaluation Metrics
- Task: Three-class classification — [Critical / High / Low]  
- Labeled each email manually for ground truth comparison.  
- Quantitatively evaluated model performance using:  
  - Accuracy  
  - F1 Score (weighted)  
  - Cohen’s Kappa Score  
  - Matthews Correlation Coefficient (MCC)

---

## [7] Project Significance and Future Directions

### (1) Project Significance and Expected Impact
- Serves as a virtual assistant specialized in email classification and summarization, contributing to time efficiency and workflow improvement.  
- Goes beyond simple keyword-based analysis by leveraging LLM-driven logical reasoning for qualitative email interpretation.  
- Introduces a creative and personalized approach by incorporating user network analysis, enabling individualized importance assessment.

### (2) Potential for Development and Future Improvements

#### ① System-Level Limitations
- Sustainability: Requires integration with real-time mailbox updates.  
- Personalization vs. Generalization: Needs restructuring for commercialization; establishing API-based access to personal mailboxes while managing security trade-offs.  
- Permanent Hosting: The current Streamlit-based local prototype must evolve into a cloud-hosted web service for stable and continuous operation.

#### ② Analytical Framework Limitations
- Dataset Limitations: The use of the relatively outdated Enron Email Dataset restricts generalization. Future evaluations should utilize modern corporate email datasets for higher representativeness.

#### ③ Model Implementation Limitations
- Token Limitations: Must enhance flexibility regarding the number and length of processed emails to avoid performance degradation.  
- Lack of Fine-Tuning: Importance classification and summarization performance could be improved through fine-tuning on labeled datasets.  
- Models such as Mistral 7B showed performance degradation and analysis failure, indicating the need for model-specific optimization.
