# Automated EDA with AI Agents (CrewAI + GPT-4)

This project automates exploratory data analysis (EDA) using an AI agent team built with [CrewAI](https://docs.crewai.com). It mimics a human data science workflow by coordinating multiple role-based agents powered by GPT-4 and Python.

---

## 🚀 Project Overview

The system accepts a tabular dataset (CSV) and runs a multi-step AI-driven EDA process that includes:

- **Data Ingestion Agent**: Summarizes column types, nulls, basic statistics.
- **Business Analyst Agent**: Asks insightful EDA questions based on metadata.
- **Data Scientist Agent**: Writes and executes Python code to answer each question (including plots).
- **Narrator Agent**: Generates an executive summary of findings.

The result is a markdown-based EDA report with charts, code, insights, and summary - ready to support business decision-making.

---

## 💡 Features

- 🧠 **Multi-Agent AI Orchestration** via [CrewAI](https://github.com/joaomdmoura/crewai)
- 🤖 GPT-4-powered agents (via OpenAI) for reasoning and execution
- 🧾 Natural language EDA questions & summaries
- 📊 Code generation for data analysis and matplotlib visualizations
- ⚙️ Prompt engineering for agent role alignment
- 📁 Organized output folder with plots and markdown files
