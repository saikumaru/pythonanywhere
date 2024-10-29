Register with your personal email account and have the tokens, keys ready.
1. Create your token on Huggingface
 https://huggingface.co/settings/tokens 

2. Create API key on Pinecone
https://app.pinecone.io/

3. Create an account to run your Python code on Pythonanywhere
https://www.pythonanywhere.com/


Building the app on Pythonanywhere
1. Get into web tab and create your first python app. Choose below options
(a)Flask framework
(b) Python version 3.10
With rest of the default options

2. Navigate to files tab and under "mysite" directory you will find the pre-initialized python file "flask_app.py"

3. Copy over the files to respective directorys in Pythonanywhere files tab

4. Install dependencies from your "console" tab
```
cd mysite/
pip install -r requirements.txt
```
5. Reload the python app from the "Web" tab

For any questions please raise an issue