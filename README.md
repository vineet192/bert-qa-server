# bert-qa-server
The documentation for the live server can be found [here](https://app.swaggerhub.com/apis/vineet192/bert-qa-server/1.0.0)
## Steps To Run
> * Create a [Python virtual environment](https://docs.python.org/3/library/venv.html)
> * Once in the environment, run ```pip install -r requirements.txt```
> * Download the [BERT model](https://drive.google.com/drive/folders/1DPGuYdPh1NBPzPc3Q-XnOQOyM5HJcj7b) and save it into a folder named mymodel in the project directory
> * Create a file named .env in the project folder and in it type (the vocab file will be in the assest folder) :
```
PATH_TO_VOCAB_FILE=<path_to_vocab.txt>
PATH_TO_MODEL=<path_to_model>
``` 
> * Run the command ```flask run``` in the terminal, the server should be running on http://localhost:5000
