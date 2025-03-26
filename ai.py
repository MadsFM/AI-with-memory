from fastapi import APIRouter, HTTPException
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, MongoDBChatMessageHistory
from pymongo import MongoClient
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from mongo_memory import MongoMemory

router = APIRouter()

#BaseModel
class UserPrompt(BaseModel):
    prompt: str = Field(..., description="Prompt message from user")
    temperature: float = Field(default=0.8, ge=0, le=2, description="The creativity of the answer.")

class Response(BaseModel):
    result: str = Field(..., description="The answer from the AI.")

#Client
client = MongoClient("mongodb://localhost:27017/")
db = client["chat_memory"]
collection = db["chat"]
model = "gemma2:9b"

#Model
llm = ChatOllama(model="gemma2:9b", temperature=0.8)

#Template
template = """
{history}
User: {input}
"""

#Prompt
prompt_temp = PromptTemplate(input_variables=["history", "input"], template=template)

#Init DB
db_memory = MongoMemory(collection)

#LangChain memory
memory = ConversationBufferMemory()

previous_messages = db_memory.load_messages()
for msg in previous_messages:
    parts = msg.split("\n")
    memory.chat_memory.add_user_message(parts[0].replace("User", ""))
    memory.chat_memory.add_user_message(parts[1].replace("Bot", ""))

conversation = LLMChain(llm=ChatOllama(model=model), memory=memory, prompt=prompt_temp, verbose=True)

#Functions
def answer_writer(prompt: str, temperature: float) -> str:
    llm.temperature = temperature
    response = conversation.predict(input=prompt)
    db_memory.save_message(prompt, response)
    return response.content if hasattr(response, "content") else str(response)

#Endpoint
@router.post("/conversation", response_model=Response)
async def answer(request: UserPrompt):
    try:
        ai_answer = answer_writer(request.prompt, request.temperature)
        return Response(result=ai_answer)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error making an answer. {str(ex)}")