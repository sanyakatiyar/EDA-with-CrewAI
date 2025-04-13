import crewai
import pandas as pd
import streamlit as st
import openai
import langchain

from crewai import Agent, Task, Crew, Process

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")
print(df.columns.tolist(), len(df))

llm_fast = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_smart = ChatOpenAI(model="gpt-4", temperature=0)

#Defining agents and tasks their respective tasks

# A. data ingestion agent
ingestion_agent = Agent(
    name="Data Ingestion Agent",
    role="Metadata summarizer",
    goal="Summarize dataset column names, types, and basic stats for EDA",
    backstory="You are skilled in summarizing a dataset and have years of experience in doing this task where you can efficiently provide a summary after ingestion.",
    llm=llm_fast
)

#task defined for the agent
ingestion_task = Task(
    name="Summarize Metadata",
    description="""
        Read the dataset at '{datapath}' and output a summary of:
        1. Column names
        2. Data types
        3. Missing values
        4. Example values
        5. Basic statistics (mean, unique count, etc.)
        Return this in markdown format.
    """,
    expected_output="Metadata summary in markdown format",
    agent=ingestion_agent
)

# B. business analyst agent
business_agent = Agent(
    name="Business Analyst",
    role="Business Consultant skilled at asking insightful questions from the data provided", 
    goal="Generate insightful questions based on dataset metadata and make no assumptions",
    backstory="You are a business analyst with years of experience exploring business data to find actionable insights. You excel at forming smart, relevant questions based on what a dataset contains.",
    llm=llm_smart
)

#task for generating questions
question_task_prompt = """
Below is a description of a dataset, including its context and fields.
Based on this information, generate **{how_many}** insightful questions for exploratory analysis.
1. Focus on questions about individual variables or relationships between two variables and do not make assumptions.
2. Use basic EDA concepts (distributions, correlations, trends, outliers) - do use other concepts which could be relevant.
3. Return the questions as a Python list of strings.
"""
question_task = Task(
    name="Generate EDA Questions",
    description=question_task_prompt,
    expected_output="A list of strings with insightful questions about the dataset.",
    agent=business_agent,
    context=[ingestion_task]  # Add context from previous task
)

#C. Data Scientist agent

ds_agent = Agent(
    name="Data Scientist",
    role="Python-savvy Data Scientist proficient in data analysis and visualization",
    goal="Analyze data to answer questions and summarize insights",
    backstory = (
    "You are a seasoned data scientist with years of experience in analyzing complex datasets across various industries. "
    "Your expertise lies in uncovering meaningful patterns, generating actionable insights, and communicating findings effectively. "
    "You approach each analytical task with a critical eye, leveraging Python and data visualization to answer business questions clearly and accurately."),
    llm=llm_smart
)

#task for data analysis
analysis_task_prompt = """
You are a data scientist answering the following question using the dataset provided.
Question: {question_str}
Follow these steps:
1. Write Python code using pandas, numpy, matplotlib/seaborn (and others if needed like scipy).
   a. Load the dataset from "{datapath}" (CSV file path).
   b. Focus only on columns relevant to the question and do not detail from the question.
   c. Perform analysis as required (compute stats or create plot) to answer the question.
   d. Print relevant results (e.g., statistics or sample values).
   e. For relevant pl 
    - Do **not** use `plt.show()`
    - Use `plt.tight_layout()` before saving
    - Save the plot to: "{image_dir}/plot.png" using `plt.savefig(...)`
2. Ensure the code is properly formatted and commented.
3. Execute the code (this will be done by the tool).
4. After execution, summarize the results:
   a. State key findings, referring to the output/plot.
   b. Keep the explanation concise and clear.
Format the output as markdown with sections:
### Question
{question_str}
### Code
```python
<your code here>
```
### Results and Analysis
<your explanation here>
"""

analysis_task = Task(
    name="Answer EDA Question",
    description=analysis_task_prompt,
    expected_output="A markdown-formatted string with question, code, output, and analysis.",
    agent=ds_agent
)

# D. Summarizer/narrator agent

narrator_agent = Agent(
    name="Narrator Agent",
    role="Executive summarizer",
    goal="Review the EDA results and summarize key insights in 3-5 concise bullet points",
    backstory="You are a skilled narrator who presents the EDA results and summarize the findings concisely while making sure that most relevant and important information is provided in the form of a summary which can be used by the users.",
    llm=llm_fast
)
summary_task = Task(
    name="Summarize Insights",
    description="""
Review the EDA results and write a 3-5 bullet summary of the most important findings. Highlight patterns, outliers, or business-relevant insights found in the analysis.

The analyses are:
{analyses_str}
""",
    expected_output="Bullet-point summary in markdown",
    agent=narrator_agent
)



#setting up the crews
prep_crew = Crew(
    agents=[ingestion_agent, business_agent],
    tasks=[ingestion_task, question_task],
    process=Process.sequential,
    verbose=True  # Added for debugging
)
eda_crew = Crew(
    agents=[ds_agent],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True  # Added for debugging
)

# function defined for collaborating the agents and running the crew to perform the EDA 
def run_two_crew_eda(datapath, image_root, how_many):
    import os
    import json
    output_dir = os.path.abspath(image_root)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run prep_crew (Ingestion Agent + Business Analyst Agent)
    prep_result = prep_crew.kickoff(inputs={
        "datapath": datapath,
        "how_many": how_many
    })
    
    # Extract questions from result
    try:
        last_task_output = prep_result
    
        if isinstance(last_task_output, list):
            questions = last_task_output
        elif isinstance(last_task_output, str):
            try:
                questions = eval(last_task_output)
                if not isinstance(questions, list):
                    questions = [last_task_output]
            except Exception as e:
                print(f"Warning: Could not parse questions: {e}")
                questions = [last_task_output]
        else:
            questions = [str(last_task_output)]
    except Exception as e:
        print(f"Error extracting questions: {e}")
        print("Debug - Result structure:", prep_result)
        questions = []

        
    if not questions:
        print("No questions generated.")
        return [], [], "No questions were generated."
    
    print(f"Generated questions: {questions}")
    
    # Run analysis for each question
    analyses = []
    for idx, q in enumerate(questions):
        img_dir = os.path.join(output_dir, f"q{idx+1}")
        os.makedirs(img_dir, exist_ok=True)
        
        try:
            print(f"\nAnalyzing question {idx+1}: {q}")
            analysis_result = eda_crew.kickoff(inputs={
                "question_str": q,
                "datapath": datapath,
                "image_dir": img_dir
            })
            
            # Extract the analysis from the result
            analysis_text = analysis_result
            
            analyses.append(analysis_text)
            print(f"Completed analysis {idx+1}")
        except Exception as e:
            print(f"Error analyzing question {idx+1}: {e}")
            analyses.append(f"Analysis failed: {str(e)}")
    
    # Run the summary task with all analyses
    analyses_str = "\n\n".join([f"Analysis {i+1}:\n{a}" for i, a in enumerate(analyses)])
    summary_crew = Crew(
        agents=[narrator_agent],
        tasks=[summary_task],
        process=Process.sequential,
        verbose=True
    )
    
    try:
        print("\nGenerating summary...")
        summary_result = summary_crew.kickoff(inputs={"analyses_str": analyses_str})
        final_summary = summary_result  # Direct output
        print("Summary generation complete")
    except Exception as e:
        print(f"Error generating summary: {e}")
        final_summary = "Error generating summary."
    
    # Done
    print("\nFINAL SUMMARY:\n")
    print(final_summary)
    
    return questions, analyses, final_summary

# Run the two-crew EDA pipeline
questions, analyses, summary = run_two_crew_eda(
    datapath="train.csv",      
    image_root="plots",            
    how_many=3                    
)

# Print results to confirm everything worked
print("Generated Questions:")
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")
print("\nEDA Analyses:")
for i, md in enumerate(analyses, 1):
    print(f"\n--- Analysis for Question {i} ---\n")
    print(md)
print("\nExecutive Summary:")
print(summary)
