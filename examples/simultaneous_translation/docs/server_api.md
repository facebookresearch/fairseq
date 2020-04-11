# Introduction to the evaluation interface

## **Server RESTful API**

Assuming the server address is *server_url*

---

## Get number of sentences in test set

* **Method:** : `GET`
  
*  **Endpoint** : *server_url*
  
*  **URL Params:**
 
* **Success Response:**
    
    ```
    {
        "num_sentences": N
    }
    ``` 


Where `N` is the number of sentences in test set

---
## Start a new evaluation session
* **Method:** : `POST`
  
*  **Endpoint** : *server_url*
  
*  **URL Params:** 

Should be used everytime starting to evaluate a new model.

___
## Obtain a source word from server

* **Method:** : `GET`
  
*  **Endpoint** : *server_url*/src
  
*  **URL Params:** 
  
    ```
    {
        "sent_id": i
    }
    ``` 
 
* **Success Response:**
    
    ```
    {
        "sent_id": i,
        "segment_id": segment_id,
        "segment": word
    }
    ```

A source word in sentence *i* will be received from server. 
Notice that the word is not tokenized. 
```segment_id``` is the position of the word in sentence, starting from zero.
The server will automatically move the pointer to next word. 
When the pointer reach the end of sentence, "<\/s>" will be sent. 
For example, for sentence "A B C". 
The first time the request happens, "A" will be received,  the second time will be "B", the third time will be "C",  and after that, it will be always "<\/s>". The request should be made everytime the model decides to read a new word.

---
## Obtain a segment of speech from server

* **Method:** : `GET`
  
*  **Endpoint** : *server_url*/src
  
*  **URL Params:** 
  
    ```
    {
        "sent_id": i,
        "segment_size": t_ms
    }
    ``` 
 
* **Success Response:**
    
    ```
    {
        "sent_id": i,
        "segment_id": segment_id,
        "segment": list_of_numbers
    }
    ```

The wav form for the segment of a speech utterance will be received in the format of list of numbers.
The sample rate on the server is 16000Hz. By default, the length of list is 160, or 10 ms in time.
There is an optional `segment_size` parameter in the request, the unit is ms. A customized length of segment can be requested. However, the length of a segment in time can only be the a multiple of 10ms. It will be rounded if not. For example, if `t_ms = 301` is requested, the returned segment will be 300ms long.

Again, the request should be made every time the model decides to read a new segment of speech utterence. 

---

## Send a translated token to the server
* **Method:** : `PUT`
  
*  **Endpoint** : *server_url*/hypo
  
*  **URL Params:** 
  
    ```
    {
        "sent_id": i,
    }
    ``` 
*  **Body** (raw text)

    ```
    WORD
    ```

After the token is sent, the server will record the delay (length of source context) the model used to predict the token. Notice that the content should be detokenized. If there is a space in `WORD`, it will be considered as multiple words split by space. In order to end a translation hypothesis, an end of sentence token "<\/s>" should be sent to the server.

---
## Get evaluation scores from the server
* **Method:** : `GET`
  
*  **Endpoint** : *server_url*/result
  
*  **URL Params:**
 
* **Success Response:**
  ```
    {
        "BLEU": BLEU, 
        "TER": TER, 
        "METEOR": METEOR,
        "DAL": DAL,
        "AL": AL,
        "AP": AP
    }
    ```
Make sure to make this request after finishing all the translations.

---

## **The structure of the evaluation client**
Here is example pseudocode for a client. 
In practice, evaluation can be done in parallel
```
POLICY <- The function gives decision of read or write
MODEL <- The translation model

Start evaluation

N <- Request to get number of sentences in test set

Request to start a new evaluation session

For id in 0,..,N-1 
    Do
        if POLICY is read
        Then
            Request to obtain a token or a speech utterence of sentence i
        Else
            W <- prediction of MODEL
            Request to send W of target sentence i to server
        EndIf
    While W is not <\s>

Request to get evaluation scores from server
```
