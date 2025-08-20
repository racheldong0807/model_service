from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import inference  # 之前写好的推理逻辑


app = FastAPI(title="MNIST 推理服务")

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    接口说明：
    - 接收上传的图片文件
    - 调用推理函数
    - 返回预测结果 JSON
    """
    # 调用推理函数
    result = inference.predict(file.file)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)