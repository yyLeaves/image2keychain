import os
import io
import base64
import json
from PIL import Image
from openai import OpenAI
from prompts import PROMPTS

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageService:
    def __init__(self, upload_dir="static/uploads", processed_dir="static/processed"):
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def convert_to_png(self, file_content: bytes, filename: str) -> str:
        """Converts uploaded image bytes to PNG and saves it."""
        try:
            image = Image.open(io.BytesIO(file_content))
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(self.upload_dir, f"{name_no_ext}.png")
            
            image.convert('RGBA').save(save_path)
            print(f"Converted: {filename} -> {save_path}")
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze_proportions(self, image_path: str, model="gpt-4o") -> dict:
        """Analyzes keychain image to extract dimensional proportions."""
        image_base64 = self.encode_image(image_path)
        
        function_schema = [
            {
                'name': 'extract_keychain_proportions',
                'description': 'Extract the physical dimensional proportions and shape information from a keychain image',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'width': {
                            'type': 'number',
                            'description': 'Width dimension (longest vertical extent). Use 1.0 as baseline for normalization.'
                        },
                        'length': {
                            'type': 'number',
                            'description': 'Length dimension (longest horizontal extent). Relative to width.'
                        },
                        'thickness': {
                            'type': 'number',
                            'description': 'Thickness/depth from front to back surface.'
                        },
                        'complexity': {
                            'type': 'string',
                            'enum': ['simple', 'moderate', 'complex'],
                            'description': 'Visual complexity of the shape.'
                        },
                    },
                    'required': ['width', 'length', 'thickness', 'complexity']
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPTS.RATIO_ANALYSIS},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                }
                            }
                        ]
                    }
                ],
                functions=function_schema,
                function_call={"name": "extract_keychain_proportions"}
            )

            message = response.choices[0].message
            
            if message.function_call:
                args = json.loads(message.function_call.arguments)
                return {
                    "success": True,
                    "data": args,
                    "ratio_string": f"{args['length']}:{args['width']}:{args['thickness']}"
                }
            
            return {"success": False, "error": "No function call returned"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_silhouette(self, image_path: str, model="gpt-image-1") -> str:
        """Generates a silhouette from the original image."""
        try:
            result = client.images.edit(
                model=model,
                image=open(image_path, "rb"),
                prompt=PROMPTS.SILHOUETTE_EXTRACTION,
                n=1,
                # size="1024x1024"
            )
            
            # Handle Base64 response
            image_data = result.data[0]
            
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_bytes = base64.b64decode(image_data.b64_json)
            elif hasattr(image_data, 'url'):
                # If API returns URL, download it (simplified here, assumes b64 request)
                # To force b64, usually need response_format="b64_json" in call
                # For this demo, we will assume the prompt implies we want the image data
                # If using standard DALL-E 2, add response_format="b64_json" to the call above
                raise ValueError("Please configure OpenAI call to return b64_json")

            filename = os.path.basename(image_path).split('.')[0] + '_silhouette.png'
            output_path = os.path.join(self.processed_dir, filename)
            
            return self._download_or_save_image(image_data, output_path)

        except Exception as e:
            print(f"Error generating silhouette: {e}")
            raise e

    def edit_silhouette(self, image_path: str, model="gpt-image-1", instructions: str="") -> str:
        """Edits a silhouette based on red marks and instructions."""
        combined_prompt = PROMPTS.SILHOUETTE_EDIT_PROMPT.format(user_instruction=instructions)
        
        try:
            result = client.images.edit(
                model="gpt-image-1",
                image=open(image_path, "rb"),
                prompt=combined_prompt,
                n=1,
            )
            
            filename = os.path.basename(image_path).split('.')[0] + '_updated.png'
            output_path = os.path.join(self.processed_dir, filename)
            
            image_bytes = base64.b64decode(result.data[0].b64_json)
            with open(output_path, "wb") as f:
                f.write(image_bytes)
                
            return output_path
        except Exception as e:
            raise ValueError(f"Error editing silhouette: {str(e)}")

    def _download_or_save_image(self, image_data, output_path):
        """Helper to handle OpenAI image response (URL or B64)."""
        image_b64 = image_data.b64_json
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(image_b64))
        return output_path
        