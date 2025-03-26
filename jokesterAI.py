from fastapi import APIRouter, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

router = APIRouter()

class UserPrompt(BaseModel):
    prompt_tmp: str = Field(..., description="Prompt message from user")
    temperature: float = Field(default=0.8, ge=0, le=2, description="The creativity of the answer.")

class Response(BaseModel):
    result: str = Field(..., description="The answer from the AI.")

#Model
llm = ChatOllama(model="gemma2:9b", temperature=0.8)

#Chat memory
short_mem = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=short_mem,
    verbose=True)

#Prompt Template
prompt_tmp = ChatPromptTemplate.from_messages(
    [("user", "{prompt}")]
)

#Chain
llm_chain = prompt_tmp | llm

#Functions
def answer_writer(prompt: str, temperature: float) -> str:
    llm.temperature = temperature
    response = conversation.predict(input=prompt)
    return response.content if hasattr(response, "content") else str(response)

#Endpoint
@router.post("/conversation", response_model=Response)
async def answer(request: UserPrompt):
    try:
        ai_answer = answer_writer(request.prompt, request.temperature)
        return Response(result=ai_answer)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error making an answer. {str(ex)}")