from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    import os
    import cohere
    os.environ['COHERE_API_KEY'] = 'yARQQpDOyath4wS5cphHJooAbgUQslMTaT6gEmyP'
    co = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())

    def get_completion(prompt, temp=0):
        response = co.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=200,
            temperature=temp)
          return response.generations[0].text
        
    Apex_Data = f"""Feature,Status,Point_of_contact,Comment
    Decimal Precision,In Progress,Murali,50% Complete
    CREF Description Update,Completed,Hamesh,Completed Last week
    Apex Project,Not Started,Sangeetha,Will start next week"""
    User_Inquiry = input("Enter your query : ")
    prompt = f"""{User_Inquiry} from {Apex_Data}
    Give in the following format
    Decimal Precision Feature is 'In Progress'
    Point of Contact for this feature is Hamesh
    As of now 50% is complete
    """

    response = get_completion(prompt, temp=0.5)
    print(response)
    return(response)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

    
