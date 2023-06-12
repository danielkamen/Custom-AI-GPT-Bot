
 Custom-AI-GPT-Bot
====
<p> 
Builds a chatbot built off openAI api, trained on website data that was scrapped from all subpages. The text files are then converted into CSV files, and the information is embedded to associate tokens with expected next words. Make sure you have a working and paid API account with OpenAI before starting any of this. 
</p>




This project was built off the OpenAI WebQA Tutorial. https://platform.openai.com/docs/tutorials/web-qa-embeddings Below are the steps I personally took to make my bot.

Config
------
```
python -m venv env

source env/bin/activate

pip install -r requirements.txt
```
### What to add to the chatbotCustom.py
1. Navigate to:
```
domain = "www.websitename.com"
```
2. Switch out websitename with the main page of the site you wish to scrape.
3. Below that is:
```
full_url = "https://www.websitename.com/"
```
4. Switch in your website for websitename
5. Insert your OpenAI API key @
```
openai.api_key = openAPIKeyHere
```
6. Update ``` prompt``` in ``` def answer_question ``` to give tailored responses based on the chatbot's goal. Do you want it to be better at selling items? Add at the beginning of the prompt that its goal is to inform the user as much as possible about the benefits of your products.
> If this is your first time scrapping the website, comment out:
``` 
#crawl(full_url) 
```
## **Once the website has been crawled once, it is very important to comment this out again to reduce runtime**

File Structure
-------

![Alt text](../fileStructure.PNG)

### If you wish to maintain a different file structure, go to chatbotCustom.py. Search for instances of:
``` 
processed/embeddings.csv
```
### and replace it with your file structure.
### Do the same for:
``` 
text/
```
