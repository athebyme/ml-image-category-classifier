from modal import App, Image

image_with_source = Image.debian_slim().pip_install("numpy").add_local_python_source("datalore", "sitecustomize")

app = App()

@app.function(image=image_with_source)
async def some_function():
    import numpy as np
    print(np.random.rand())