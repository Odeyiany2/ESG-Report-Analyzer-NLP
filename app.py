import uvicorn
import gradio as gr
from src.backend.fastapi_logic import app

demo = gr.Blocks()
with demo:
    gr.Markdown("ESG Report Analyzer API is running. See /docs for the endpoints.")

app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)