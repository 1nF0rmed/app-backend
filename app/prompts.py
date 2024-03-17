import base64
from typing import List
from openai import OpenAI
from xml.etree import ElementTree as ET

from pydantic import BaseModel

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
    items = [item.get('name') for item in root.findall('wasteitems/wasteitem')]
    return {"items": items}

class Product(BaseModel):
    name: str
    items: List[str]
    steps: List[str]

def parse_products(xml_string: str) -> List[Product]:
    root = ET.fromstring(xml_string)
    products = []
    for product in root.findall('products/product'):
        name = product.get('name')
        items = [item.get('name') for item in product.findall('wasteitems/wasteitem')]
        steps = [step.get('desc') for step in product.findall('steps/step')]
        products.append(Product(name=name, items=items, steps=steps))
    return products

def get_items_in_image(image_path: str) -> dict:
    vis_prompt = """
You are a trash recycling and reuse vision system for HackPSU. Your goal is to take any given image of trash and identify
the various items that are present in it. These items will then later be segregated by another system and prompted to reuse.

Please follow the following rules when identifying items, as mentioned in between <rules></rules> tag.
<rules>
<rule>
The user will be familiar with the general purpose names of the items
</rule>
<rule>
Do not identify items that cannot be reused or repurposed by the user
</rule>
</rules>

Please only respond in the example response format provided in between <example></example> tags.
<example>
<content>
<wasteitems>
<wasteitem name="paper"/>
<wasteitem name="fruit" />
</wasteitems>
</content>
</example>

Now, start your response:
"""
    response = prompt_vision(image_path, vis_prompt)

    return parse_vision_items(response.message.content)

def determine_products_from_items(items: list[str]) -> List[Product]:

    item_strings = [f"<wasteitem name='{item}'/>" for item in items]
    items_string = "<wasteitems>\n" + "\n".join(item_strings) + "\n</wasteitems>"

    products_prompt = f"""
You are a recycling and reuse supportive system for HackPSU. Your goal is to take a list of items that will be provided to you
and then determine the various ways these items could be repurposed or recycled by the user. Along with each product, you will
also provide steps to produce it assuming that the person has functional hands, basic tools and knowledge of a high schooler.


Using the above examples, provide a response of products, items that can be used to make product and steps for the same as described in the <response></response> tags.
<response>
<products>
<product name="Product Name here">
<wasteitems>
<wasteitem name="item name here">
<-- Next item here -->
</wasteitems>
<steps>
<step desc="Step of the process to make product using items" />
<-- Next step here -->
</steps>
</product>
<-- Next product here -->
</products>
</response>

Please following the given rules:
1. Only use the given waste items and do not assume any other items.
2. Tools are not items, consider that the user has a basic toolkit.

Now, here are the available waste items:
{items_string}

Your response:
"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output XML."},
            {"role": "user", "content": products_prompt}
        ],
        max_tokens=1000
    )

    print(f"Items: {items_string}")
    print(f"Products: {response.choices[0].message.content}")

    return parse_products(response.choices[0].message.content)


def generate_products_from_image(file_path: str):

    items_string = get_items_in_image(file_path)["items"]
    products = determine_products_from_items(items_string)

    return products
