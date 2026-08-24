import asyncio
import websockets
import base64
import cv2
import numpy as np

async def test_websocket():
    # Create a dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buffer = cv2.imencode('.jpg', img)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    data_url = "data:image/jpeg;base64," + jpg_as_text

    try:
        async with websockets.connect("ws://127.0.0.1:7860/ws") as websocket:
            await websocket.send(data_url)
            response = await websocket.recv()
            print(f"Server response: {response}")
    except Exception as e:
        print(f"Error connecting or communicating with WS: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())
