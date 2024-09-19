from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class MatchFinderChatbot:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Initialize the language model
        self.llm = ChatOpenAI(model_name=model_name)
        self.template = """
        You are a match-finding assistant. We have to find a possible benificiary of a deceased person's insurance policy. 
        We have two dataset. One if the matched dataset which contains probable matches and another one is input dataset which is a deceased person's details. This can be vice-versa. 
         
        You have been given two datasets: an input dataset and a dataset containing probable matches with distance scores.
        The goal is to find the closest match from the matched dataset by matching each data of tha matched dataset with the input dataset based on a distance metric provided.
        The input dataset has the following columns:
        Age, latitude, longitude, Ethnicity, Diabetic, Religion, Height, last_name
        Whereas, the matched dataset contains one more additional column which is Distance. 
        The distance column represents the similarity between rows. Your task is to identify the perfect match and explain why this row is the best match based on the data.

        Below is the information for the row of input data and the matched data, as well as the distance:
        
        Input Data: {input_data}
        Matches Data: {matched_data}
        

        Explain why the matched data is a good match for the input data.
        Below are the reasons why we have to match the data:
        The input data is of a deceased person, and the matched df contains the probable benificiary matches.
        
        You have to answer few questions along with the explainations:
        - Why this person looks similar to the person who is represented in the input data?
        - What are the key points to refer when we are saying this person is a better match to the person who is represented in the input data
        - Why others in the matches dataset are not considered
        """

        self.prompt = PromptTemplate(input_variables=["input_data", "matched_data"], template=self.template)
    

    def find_best_match(self, input_data, matched_data,):
        """
        Given a row from the input dataframe and a distances array for that row, 
        find the best match from the training dataframe and use the LLM to explain it.
        """
       
        
        # Prepare data for the prompt
        input_data = input_data.to_dict()
        matched_data = matched_data.to_dict()
        
        # Create the chain and generate the response
        chain = self.prompt | self.llm | StrOutputParser()
        response = chain.invoke(
            {"input_data":input_data, 
             "matched_data":matched_data})
        
        return response

    def find_matches(self, input_df, matched_df):
        """
        For each row in the input_df, find the best match from training_df and return explanations.
        """

        response = self.find_best_match(input_df, matched_df)

        return response
