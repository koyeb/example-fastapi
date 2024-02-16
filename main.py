from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/restaurent/{restaurent_id}")
def read_item(restaurent_id: int, q: Union[str, None] = None):
    return {"restaurent_id": restaurent_id, "q": q}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

    
