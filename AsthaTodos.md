
1. Install the python environment- using the commands in setup_script should do it. 


1. Create BOW embedder in `src/bag_of_words.py` (I think I did this for you using your old code)
    - Create a function called initialize_bow that accepts a dataset and returns an initialized embedder

    - Create the embedder as a custom pytorch class. This needs the following methods:
        1. __init__ 
            - Here you want to create the vocabulary for your BOW embedder. Be sure to handle unkown words
            - Save the vocabulary in an internal self.vocab variable
            - Save the size of the vocabulary +1 (for the criteria number) as self.embedding_size
        2. preprocess_data:
            - inputs: 
                - a tuple of (x, y, problem_id, student_id)
                - the hyperparameter dictionary if you need anything from that
            - outputs:
                - a tuple of (x, y, problem_id, student_id) where X has been transformed in the relevent way
            - This method should do any stemming/tokenization you wish to do to the words        
        3. forward:
            - This accepts the input features (a tensor of strings) and returns the embeddings
            - Use the vocab you created earlier here.

2. Create a copy of `create_finetune_yamls.py` for both frozen bert and BOW. Change the relevant details

3. Run submit_jobs.py for both frozen bert and bow.