C:/Users/ashek/AppData/Local/Microsoft/WindowsApps/python3.12.exe -m streamlit run app.py

Streamlit Cloud deployment note:
1. Open Manage app -> Settings -> Advanced settings.
2. Set Python version to 3.10 or 3.11.
3. Save and Reboot app.

Reason: TensorFlow does not provide wheels for Python 3.14, so dependency installation fails on the default environment.