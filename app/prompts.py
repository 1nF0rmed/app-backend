import base64
from openai import OpenAI
from xml.etree import ElementTree as ET

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prompt_vision(image_path, user_prompt):
    client = OpenAI()
    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
      model="gpt-4-vision-preview",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": user_prompt},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              },
            },
          ],
        }
      ],
      max_tokens=500,
    )

    return response.choices[0]

def parse_vision_items(xml_string):
    root = ET.fromstring(xml_string)
    items = [item.get('name') for item in root.findall('items/item')]
    return {"items": items}
