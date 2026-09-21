import config
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from address_alignment import address_alignment
from transformers import BertForTokenClassification, BertTokenizerFast


class AddressAlignmentRequest(BaseModel):
    message: str = Field(..., description="地址文本信息")


class AddressAlignmentResponse(BaseModel):
    prov: str | None = Field(..., description="省份")
    city: str | None = Field(..., description="城市")
    district: str | None = Field(..., description="区/县")
    town: str | None = Field(..., description="街道/街道")
    detail: str | None = Field(..., description="详细地址")
    name: str | None = Field(..., description="姓名")
    phone: str | None = Field(..., description="电话")


# 创建 FastAPI 实例
app = FastAPI()
# 加载模型
model_path = config.FINETUNED_PATH / "best"
model = BertForTokenClassification.from_pretrained(model_path).to(config.DEVICE)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


@app.get("/")
async def homepage():
    return FileResponse("templates/index.html")


@app.post("/address_alignment")
async def handle_message(request: AddressAlignmentRequest) -> AddressAlignmentResponse:
    user_message = request.message
    address = address_alignment(
        user_message, model, tokenizer, config.LABELS, config.MYSQL_CONFIG
    )
    return AddressAlignmentResponse(
        prov=address["prov"],
        city=address["city"],
        district=address["district"],
        town=address["town"],
        detail=address["detail"],
        name=address["name"],
        phone=address["phone"],
    )


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8089)
