# Telecom KPI & Customer Performance Tracking System

📊 Project Overview

This project is an end-to-end AI-powered system for real-time telecom KPI monitoring, customer churn prediction, and network performance tracking. It automates data ingestion, ML model training, and analytics visualization using modern MLOps & Data Engineering best practices.

🔹 Use Case: Helps telecom providers monitor network uptime, call drop rates, data usage, and customer churn in real-time.
🔹 Key Outcome: Automates ML model retraining & KPI tracking to optimize network performance.

🛠 Tech Stack

✅ MLOps & Automation: Docker, Apache Airflow, MLflow, CI/CD
✅ Cloud & Infrastructure: AWS (EC2), Snowflake
✅ Machine Learning & API: Scikit-Learn, FastAPI, Model Monitoring
✅ Data Engineering & Analytics: SQL, Pandas, Streamlit, ETL Pipelines
✅ Visualization: Streamlit, Plotly

📌 Features

🔹 📡 Real-Time Telecom KPI Monitoring – Tracks call drop rates, latency, and data usage trends
🔹 🔮 Customer Churn Prediction – Uses ML models to predict customers likely to churn
🔹 📈 AI Model Performance Tracking – MLflow logs model metrics and versions
🔹 ⚙️ Automated Data Pipeline – Airflow schedules & manages data ingestion from Snowflake
🔹 🌐 Interactive Dashboard – Streamlit provides real-time insights for business users
🔹 🚀 Scalable Deployment – CI/CD, Docker, and AWS for enterprise-grade scalability

![Screenshot_25-3-2025_145133_localhost](https://github.com/user-attachments/assets/5b32a575-46b0-4eec-8a4e-00aa6d100856)
![Screenshot_25-3-2025_145715_localhost](https://github.com/user-attachments/assets/4336ab6b-3267-4690-b11b-382181e0d5d9)
![Screenshot_25-3-2025_145737_localhost](https://github.com/user-attachments/assets/93ff8491-3382-4074-a75f-d833223937d2)


💻 Setup & Installation
1️⃣ Clone the Repository

git clone https://github.com/your-username/telecom-mlops.git
cd telecom-mlops

2️⃣ Install Dependencies
```
pip install -r requirements.txt
```

3️⃣ Set Up Airflow & MLflow with Docker
```
docker-compose up -d
```
4️⃣ Run the Streamlit Dashboard
```
cd dashboard
streamlit run app.py
```
🔗 Access the dashboard at http://localhost:8501
5️⃣ Run the API for Model Predictions
```
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
🔗 API available at http://localhost:8000/docs
📌 How It Works

1️⃣ Data Pipeline (Airflow & Snowflake)

    Automates data extraction & transformation
    Stores cleaned telecom data in Snowflake

2️⃣ Machine Learning Model (Scikit-Learn & MLflow)

    Predicts customer churn & network failures
    MLflow tracks model versions & performance

3️⃣ Dashboard & API Deployment

    Streamlit visualizes real-time KPIs
    FastAPI serves ML predictions

  🚀 Future Improvements

✅ Add anomaly detection for network issues
✅ Optimize ML models for better accuracy
✅ Enhance API with authentication & security
🤝 Contributing

🔹 Pull requests are welcome! Feel free to submit feature requests or report issues in the GitHub Issues section.


📞 Contact & Support

📩 Email: oluwafemi.ezra@gmail.com
🔗 LinkedIn: www.linkedin.com/in/oluwafemi-oloye-a3b772353
📜 License

This project is open-source and available under the MIT License.
