from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field("default", pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Session id: same id = same memory and same knowledge base")


class AgentStep(BaseModel):
    tool: str
    args: dict
    result: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    charts: list[str] = Field(default_factory=list, description="Chart URLs served under /charts")


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    size: int


class KBDocument(BaseModel):
    filename: str
    title: str
    doc_type: str
    summary: str
    keywords: list[str]
    n_pages: int
    ocr_pages: int
    n_sections: int
    n_chunks: int


class KBResponse(BaseModel):
    overview: str | None
    keywords: list[str]
    documents: list[KBDocument]
    prompt_block: str = Field(description="Exactly what the agent sees about this knowledge base")
